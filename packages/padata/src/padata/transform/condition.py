from collections.abc import Callable

from biotite.structure import AtomArray, AtomArrayStack

from .base import BaseTransform


class ConditionalTransform(BaseTransform):
    def __init__(self, transform: BaseTransform, condition: Callable[[AtomArray | AtomArrayStack], bool]) -> None:
        """Initialize the conditional transform.

        Args:
            transform (BaseTransform): transform to apply.
            condition (Callable[[AtomArray  |  AtomArrayStack], bool]):
                condition to check before applying the transform.

        """
        super().__init__()
        self._transform = transform
        self.condition = condition

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Transform the atom array or stack if the condition is met."""
        if self.condition(atoms):
            atoms = self._transform(atoms)
        return atoms
