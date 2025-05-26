from biotite.structure import AtomArray, AtomArrayStack

from .base import BaseTransform


class ComposeTransform(BaseTransform):
    def __init__(self, transforms: list[BaseTransform]) -> None:
        """Initialize the compose transform.

        Args:
            transforms (list[BaseTransform]): list of transforms to apply.

        """
        super().__init__()
        self.transforms = transforms

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Apply a list of transforms to the atom array or stack."""
        for transform in self.transforms:
            atoms = transform(atoms)
        return atoms
