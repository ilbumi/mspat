import warnings

import hydride
from biotite.structure import AtomArray, AtomArrayStack

from .base import BaseTransform


class ProtonateStructure(BaseTransform):
    def __init__(
        self, force_reprotonation: bool = False, max_iterations: int = 100, ignore_warnings: bool = False
    ) -> None:
        """Initialize the protonation transform.

        Args:
            force_reprotonation (bool, optional):
                if false the transform will not be applied to  structures with hydrogens.
                Defaults to False.
            max_iterations (int, optional):
                maximum number of iterations for the hydrogen relaxation.
                Defaults to 100.
            ignore_warnings (bool, optional):
                if true, warnings will be ignored during the transformation.

        """
        super().__init__()
        self.force_reprotonation = force_reprotonation
        self.max_iterations = max_iterations
        self.ignore_warnings = ignore_warnings

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Protonate the structure.

        Args:
            atoms (AtomArray | AtomArrayStack): atom array or stack

        Returns:
            AtomArray | AtomArrayStack: protonated atom array or stack

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore" if self.ignore_warnings else "default")
            if not (atoms.element == "H").any():
                atoms, _ = hydride.add_hydrogen(atoms)
                atoms.coord = hydride.relax_hydrogen(atoms, iterations=self.max_iterations)
            elif self.force_reprotonation:
                atoms = atoms[atoms.element != "H"]
                atoms, _ = hydride.add_hydrogen(atoms)
                atoms.coord = hydride.relax_hydrogen(atoms, iterations=self.max_iterations)
            return atoms


class DeProtonateStructure(BaseTransform):
    def __init__(self) -> None:
        """Initialize the deprotonation transform."""
        super().__init__()

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Deprotonate the structure.

        Args:
            atoms (AtomArray | AtomArrayStack): atom array or stack

        Returns:
            AtomArray | AtomArrayStack: deprotonated atom array or stack

        """
        return atoms[atoms.element != "H"]
