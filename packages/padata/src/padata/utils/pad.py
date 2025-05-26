from collections.abc import Sequence
from typing import Literal

import torch


def _get_pad(dims: Sequence[int], target_dims: Sequence[int], padding_side: Literal["right", "left"]) -> list[int]:
    """Get padding values based on the dimensions and padding side."""
    pad: list[int] = []
    for dim, target in zip(reversed(dims), reversed(target_dims), strict=False):
        if dim > target:
            msg = "All dimensions must be less than or equal to the target dimensions."
            raise ValueError(msg)
        if padding_side == "right":
            pad.extend((0, target - dim))
        else:
            pad.extend((target - dim, 0))
    return pad


def pad_matrix(
    matrix: torch.Tensor | list[torch.Tensor],
    padding_value: float = 0,
    padding_side: Literal["right", "left"] = "right",
) -> torch.Tensor:
    """Pad matrices to the same size with a specified value on the left or right side.

    Args:
        matrix (torch.Tensor | list[torch.Tensor]): The 3d or 4d tensors to pad.
        padding_value (float, optional): The value to use for padding. Defaults to 0.
        padding_side (Literal["right", "left"], optional): The side to pad. Can be "left" or "right".
            Defaults to "right".

    Returns:
        torch.Tensor: The padded matrix.

    """
    if not isinstance(matrix, list):
        return matrix

    target_size = [max(m.shape[1] for m in matrix), max(m.shape[2] for m in matrix)]
    if matrix[0].ndim > 3:  # noqa: PLR2004
        target_size.append(matrix[0].shape[3])
    return torch.concat(
        [
            torch.nn.functional.pad(
                m,
                _get_pad(m.shape[1:], target_size, padding_side),
                value=padding_value,
            )
            for m in matrix
        ]
    )
