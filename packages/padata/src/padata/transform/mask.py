import numpy as np
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.filter import filter_peptide_backbone

from padata.transform.base import BaseTransform


class MaskSpans(BaseTransform):
    def __init__(
        self,
        num_spans: tuple[int, int] = (1, 6),
        span_length: tuple[int, int] = (1, 25),
        random_seed: int | None = None,
    ):
        """Initialize the MaskSpans transform.

        Args:
            num_spans: Range of number of spans to mask (both ends inclusive).
            span_length: Range of length of each span to mask (both ends inclusive).
            random_seed: Random seed for reproducibility.

        """
        super().__init__()
        self.num_spans = num_spans
        self.span_length = span_length
        self._rng = np.random.default_rng(random_seed)

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Transform the atom array or stack if the condition is met."""
        unique_chain_ids = np.unique(atoms.chain_id)
        current_num_spans = self._rng.integers(
            low=self.num_spans[0],
            high=self.num_spans[1] + 1,
            size=1,
            dtype=int,
        )
        chosen_chains = self._rng.choice(unique_chain_ids, size=current_num_spans, replace=True)
        mask = np.ones(len(atoms), dtype=bool)
        for chain_id in chosen_chains:
            res_ids = atoms.res_id[atoms.chain_id == chain_id]
            min_res_id = np.min(res_ids)
            max_res_id = np.max(res_ids)
            chosen_length = self._rng.integers(
                low=self.span_length[0],
                high=self.span_length[1] + 1,
                size=1,
                dtype=int,
            )
            start_res_id = self._rng.integers(
                low=min_res_id,
                high=max_res_id - chosen_length + 1,
                size=1,
                dtype=int,
            )
            end_res_id = start_res_id + chosen_length
            mask[(atoms.res_id >= start_res_id) & (atoms.res_id < end_res_id) & (atoms.chain_id == chain_id)] = False
        atoms.set_annotation("mask", mask)
        return atoms


class RemoveMaskedSideChains(BaseTransform):
    """Remove chains that are completely masked."""

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Remove chains that are completely masked."""
        mask = atoms.get_annotation("mask")

        return atoms[filter_peptide_backbone(atoms) | mask]  # Keep only peptide backbone atoms or those not masked
