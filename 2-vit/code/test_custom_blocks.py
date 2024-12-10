import torch
import pytest
from custom_blocks import PatchEmbed

def test_patch_embed():
    # Set up parameters
    kernel_size = (16, 16)
    stride = (16, 16)
    padding = (0, 0)
    in_chans = 3
    embed_dim = 768
    batch_size = 2
    img_size = (32, 32)  # Image size (H, W)

    # Create a PatchEmbed instance
    patch_embed = PatchEmbed(kernel_size, stride, padding, in_chans, embed_dim)

    # Create a dummy input tensor
    input_tensor = torch.randn(batch_size, in_chans, *img_size)

    # Forward pass
    output = patch_embed(input_tensor)

    # Check output shape
    expected_output_shape = (batch_size, img_size[0] // stride[0], img_size[1] // stride[1], embed_dim)
    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, but got {output.shape}"

    # Check if output is a tensor
    assert isinstance(output, torch.Tensor), "Output is not a tensor"

    # Check if the output is not NaN
    assert not torch.isnan(output).any(), "Output contains NaN values"

    # Check if the output is not infinite
    assert torch.isfinite(output).all(), "Output contains infinite values"

if __name__ == "__main__":
    pytest.main()