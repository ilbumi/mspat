import torch


def pad_matrix(
    matrix: torch.Tensor | list[torch.Tensor], padding_value: float = 0, padding_side: str = "right"
) -> torch.Tensor:
    """Pad matrices to the same size with a specified value on the left or right side.

    Args:
        matrix (torch.Tensor | list[torch.Tensor]): The matrix to pad.
        padding_value (float, optional): The value to use for padding. Defaults to 0.
        padding_side (str, optional): The side to pad. Can be "left" or "right". Defaults to "right".

    Returns:
        torch.Tensor: The padded matrix.

    """
    if not isinstance(matrix, list):
        matrix = [matrix]

    max_size_1 = max(m.shape[1] for m in matrix)
    max_size_2 = max(m.shape[2] for m in matrix)
    if padding_side == "right":
        return torch.concat(
            [
                torch.nn.functional.pad(
                    m,
                    (
                        0,
                        0,
                        0,
                        max_size_1 - m.shape[1],
                        0,
                        max_size_2 - m.shape[2],
                    ),
                    value=padding_value,
                )
                for m in matrix
            ]
        )
    return torch.concat(
        [
            torch.nn.functional.pad(
                m,
                (
                    0,
                    0,
                    max_size_1 - m.shape[1],
                    0,
                    max_size_2 - m.shape[2],
                    0,
                ),
                value=padding_value,
            )
            for m in matrix
        ]
    )
