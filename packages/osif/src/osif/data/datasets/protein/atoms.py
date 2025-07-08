"""Atom neighborhood dataset."""

import os
from collections.abc import Callable, Collection, Generator
from glob import glob
from multiprocessing import Pool

import pandas as pd
import prody
import torch
from torch_geometric.data import Data, Dataset

from osif.data.utils.protein.atom_tensors import AtomTensors
from osif.data.utils.protein.vocabs import ATOM_NAMES_TO_INDEX, AtomPropertyName


def _process_file(task: tuple[str, str, bool]) -> None:
    pdb_path, torch_path, remove_failed = task
    try:
        AtomTensors.from_atom_group(prody.parsePDB(pdb_path)).save(torch_path)
    except Exception as e:
        if remove_failed:
            os.remove(pdb_path)
        else:
            raise e  # noqa: TRY201


class ProteinAtomsDataset(Dataset):
    """A protein dataset.

    Each subgraph sampled is an ego graph centered at a protein atom and
    includes its neighborhood (by proximity in 3D space).
    """

    def __init__(
        self,
        root: str,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pdb_folder: str | None = None,
        central_atom_names: Collection[str] = ("CA",),
        min_neighborhood_size: float = 7.1,
        max_neighborhood_size: float = 14.1,
        preprocess_workers: int = 1,
        remove_failed: bool = True,
    ):
        """Initialize the dataset.

        Args:
            root (str): root directory.
            transform (Callable[[Data], Data] | None, optional): transform
                to be applied to the data during loading. Defaults to None.
            pre_transform (Callable[[Data], Data] | None, optional): transform
                to be applied to the data before saving. NOT IMPLEMENTED. Defaults to None.
            pdb_folder (str | None, optional): folder with pdb files. Defaults to None.
            central_atom_names (tuple[str], optional): names of the possible central atoms. Defaults to ("CA",).
            min_neighborhood_size (float, optional): minimal size of the neighborhood to sample in Angstroms. Defaults to 7.1.
            max_neighborhood_size (float, optional): maximal size of the neighborhood to sample in Angstroms. Defaults to 14.1.
            preprocess_workers (int, optional): number of workers for dataset preprocessing. Defaults to 1.
            remove_failed (bool, optional): whether to remove broken PDB files or skip them.
                Existing failed PDB files will trigger the preprocessing step. Defaults to True.

        """
        self.central_atom_names = central_atom_names
        assert (  # noqa: S101
            len(self.central_atom_names) >= 1
        ), "At least one central atom type must be specified."
        self.min_neighborhood_size = min_neighborhood_size
        self.max_neighborhood_size = max_neighborhood_size
        self.preprocess_workers = preprocess_workers
        self.remove_failed = remove_failed

        self.pdb_folder = pdb_folder if pdb_folder else os.path.join(root, "raw")
        self._raw_file_names = sorted(
            [x.split("/")[-1] for x in glob(os.path.join(self.pdb_folder, "*.pdb.gz"))],
        )

        self._processed_subfolders = [self.extract_pdb_id_from_path(path) for path in self._raw_file_names]
        self._processed_files = [os.path.join(folder, "pos.pt") for folder in self._processed_subfolders]
        self._processed_files.extend([os.path.join(folder, "labels.pt") for folder in self._processed_subfolders])
        self._processed_files.append("index.parquet")
        super().__init__(root, transform, pre_transform)
        self.index = pd.read_parquet(self.processed_paths[-1])

    @property
    def raw_dir(self) -> str:
        """Raw directory of the dataset."""
        return self.pdb_folder

    @property
    def raw_file_names(self) -> list[str]:
        """Raw data file names."""
        return self._raw_file_names

    @property
    def processed_file_names(self) -> list[str]:
        """Processed data file names."""
        return self._processed_files

    def len(self) -> int:
        """Number of processed samples."""
        return len(self.index)

    @staticmethod
    def _get_neighborhood_mask(
        center_pos_idx: int, atom_tensors: AtomTensors, neighborhood_size: float
    ) -> torch.Tensor:
        distances = torch.norm(atom_tensors.pos - atom_tensors.pos[center_pos_idx], dim=1)
        return distances <= neighborhood_size

    @staticmethod
    def _extend_mask_to_residues(mask: torch.Tensor, atom_tensors: AtomTensors) -> torch.Tensor:
        """Extend the mask to include whole residues."""
        selected_residues = atom_tensors.labels[mask][:, [AtomPropertyName.chain, AtomPropertyName.resindex]].unique(
            dim=0
        )
        mask = torch.zeros(atom_tensors.labels.shape[0], dtype=torch.bool)
        for selected_chain, selected_residue in selected_residues:
            mask |= (atom_tensors.labels[:, AtomPropertyName.resindex] == selected_residue) & (
                atom_tensors.labels[:, AtomPropertyName.chain] == selected_chain
            )
        return mask

    def _get_target_atoms_mask(self, atom_tensors: AtomTensors) -> torch.Tensor:
        mask = torch.zeros(atom_tensors.labels.shape[0], dtype=torch.bool)
        for atom_name in self.central_atom_names:
            mask |= atom_tensors.labels[:, AtomPropertyName.atom_type] == ATOM_NAMES_TO_INDEX[atom_name]
        return mask

    def _construct_data(self, atom_tensors: AtomTensors, central_atom_idx: int) -> Data:
        y = atom_tensors.labels[:, AtomPropertyName.residue_type].clone()
        if self.min_neighborhood_size == self.max_neighborhood_size:
            neihgborhood_size = self.min_neighborhood_size
        else:
            neihgborhood_size = self.min_neighborhood_size + torch.rand(1).item() * (
                self.max_neighborhood_size - self.min_neighborhood_size
            )
        neihgborhood_mask = self._get_neighborhood_mask(central_atom_idx, atom_tensors, neihgborhood_size)
        neihgborhood_mask = self._extend_mask_to_residues(neihgborhood_mask, atom_tensors)

        central_atom_mask = torch.zeros(atom_tensors.labels.shape[0], dtype=torch.bool)
        central_atom_mask[central_atom_idx] = True

        return Data(
            pos=atom_tensors.pos[neihgborhood_mask].float(),
            labels=atom_tensors.labels[neihgborhood_mask].long(),
            query_mask=central_atom_mask[neihgborhood_mask],
            y=y[neihgborhood_mask].long(),
        )

    def _load_data(self, pdb_id: str, central_atom_idx: int) -> Data:
        """Load the data from a file."""
        path = os.path.join(self.processed_dir, pdb_id)
        atom_tensors = AtomTensors.load(path)
        return self._construct_data(atom_tensors, central_atom_idx)

    def get(self, idx: int) -> Data:
        """Get the object at the given index."""
        row = self.index.loc[idx]
        return self._load_data(row.pdb_id, row.central_atom_idx)

    def _calculate_index(self) -> None:
        """Calculate the index of the dataset."""
        index: dict[int, tuple[str, int]] = {}
        current_idx = 0
        for _, pdb_id in enumerate(self._processed_subfolders):
            atom_tensors = AtomTensors.load(os.path.join(self.processed_dir, pdb_id))
            # calculate the mask for the central atoms
            mask = self._get_target_atoms_mask(atom_tensors)
            target_indices = mask.nonzero().squeeze()
            if target_indices.dim() > 0:
                for idx in target_indices:
                    index[current_idx] = (pdb_id, idx.item())
                    current_idx += 1
        pd.DataFrame.from_dict(index, orient="index", columns=["pdb_id", "central_atom_idx"]).to_parquet(
            self.processed_paths[-1]
        )

    def process(self) -> None:
        """Prepare the dataset from raw files."""
        if self.preprocess_workers > 0:
            with Pool(processes=self.preprocess_workers) as pool:
                pool.map(
                    _process_file,
                    (
                        (
                            path,
                            os.path.join(
                                self.processed_dir,
                                self.extract_pdb_id_from_path(path),
                            ),
                            self.remove_failed,
                        )
                        for path in self.raw_paths
                    ),
                )
        else:
            for path in self.raw_paths:
                _process_file(
                    (
                        path,
                        os.path.join(
                            self.processed_dir,
                            self.extract_pdb_id_from_path(path),
                        ),
                        self.remove_failed,
                    )
                )
        self._calculate_index()

    @staticmethod
    def extract_pdb_id_from_path(path: str) -> str:
        """Extract the PDB ID from a path."""
        return path.split("/")[-1].split(".")[0]

    def generate_data_from_selection(self, structure: prody.AtomGroup, selection: str) -> Generator[Data, None, None]:
        """Generate data from a selection of atoms in a structure.

        Iterates over the target atoms in the selection.

        Args:
            structure (prody.AtomGroup): structure to sample from.
            selection (str): prody selection string.

        """
        atom_tensors = AtomTensors.from_atom_group(structure)
        selected_region: prody.Selection = structure.select(selection)
        if selected_region is None:
            raise ValueError(f"Selection {selection} returned no atoms.")
        selected_residues = set(
            zip(
                selected_region.getChindices(),
                selected_region.getResindices(),
                strict=False,
            )
        )
        mask = torch.zeros(atom_tensors.labels.shape[0], dtype=torch.bool)
        for selected_chain, selected_residue in selected_residues:
            mask |= (atom_tensors.labels[:, AtomPropertyName.resindex] == selected_residue) & (
                atom_tensors.labels[:, AtomPropertyName.chain] == selected_chain
            )
        mask &= self._get_target_atoms_mask(atom_tensors)
        target_indices = mask.nonzero().squeeze()
        for idx in target_indices:
            data = self._construct_data(atom_tensors, idx.item())

            if self.transform is not None:
                data = self.transform(data)

            yield data
