from abc import ABCMeta, abstractmethod

from biotite.structure import AtomArray, AtomArrayStack


class BaseTransform(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Apply the transform."""
        raise NotImplementedError

    def __call__(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Apply the transform."""
        return self.transform(atoms)

    def __repr__(self) -> str:
        """Return the name of the class."""
        return self.__class__.__name__
