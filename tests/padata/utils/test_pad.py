import torch
from hypothesis import given
from hypothesis import strategies as st
from padata.utils.pad import pad_matrix


@st.composite
def random_3d_tensor(draw):
    """Generate a random tensor of shape n_dim 3 or 4."""
    dims = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3))
    return torch.randn(*dims)


@st.composite
def random_4d_tensor(draw):
    """Generate a random tensor of shape n_dim 4."""
    dims = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=3, max_size=3))
    return torch.randn(*dims, 4)


@given(st.lists(random_3d_tensor(), min_size=1, max_size=10) | st.lists(random_4d_tensor(), min_size=1, max_size=10))
def test_3d_tensors_pad(matrices: list[torch.Tensor]):
    """Test the pad_matrix function."""
    padded_matrix = pad_matrix(matrices, padding_value=0, padding_side="right")
    size_1 = max(matrix.shape[1] for matrix in matrices)
    size_2 = max(matrix.shape[2] for matrix in matrices)
    batch_size = sum(matrix.shape[0] for matrix in matrices)
    if matrices[0].ndim == 4:
        assert padded_matrix.shape == (batch_size, size_1, size_2, matrices[0].shape[3])
    else:
        assert padded_matrix.shape == (batch_size, size_1, size_2)
