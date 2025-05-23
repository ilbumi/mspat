import torch
from hypothesis import given
from hypothesis import strategies as st
from padata.utils.pad import pad_matrix


@st.composite
def random_tensor(draw):
    """Generate a random tensor of shape n_dim 3 or 4."""
    n_dim = draw(st.integers(min_value=3, max_value=4))
    dims = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=n_dim, max_size=n_dim))
    return torch.randn(*dims)


@given(st.lists(random_tensor(), min_size=1, max_size=10))
def test_pad(matrices: list[torch.Tensor]):
    """Test the pad_matrix function."""
    padded_matrix = pad_matrix(matrices, padding_value=0, padding_side="right")

    assert padded_matrix.shape == (6, 7, 8)
