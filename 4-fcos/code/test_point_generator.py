import torch 

from libs.point_generator import PointGenerator

def test_point_generator():
    img_size = (448, 480) # can be different for different image
    img_max_size = max(img_size)
    fpn_strides = [8, 16, 32, 64, 128]
    regression_range = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 1000)]
    point_generator = PointGenerator(img_max_size, fpn_strides, regression_range)
    
    fpn_features = [
        torch.randn(4, 256, 56, 60),
        torch.randn(4, 256, 28, 30),
        torch.randn(4, 256, 14, 15),
        torch.randn(4, 256, 7, 8),
        torch.randn(4, 256, 4, 4),
    ]
    
    points, strides, reg_range = point_generator(fpn_features)
    assert len(points) == 5
    assert points[0].shape == torch.Size([56, 60, 2])
    assert points[1].shape == torch.Size([28, 30, 2])
    assert points[2].shape == torch.Size([14, 15, 2])
    assert points[3].shape == torch.Size([7, 8, 2])
    assert points[4].shape == torch.Size([4, 4, 2])
    
    assert len(strides) == 5
    assert torch.allclose(strides, torch.tensor(fpn_strides, dtype=torch.float))
    
    assert reg_range.shape == torch.Size([5, 2])
    assert torch.allclose(reg_range, torch.tensor(regression_range, dtype=torch.float))


# points:  [tensor([[[  4.,   4.],
#          [  4.,  12.],
#          [  4.,  20.],
#          ...,
#          [  4., 460.],
#          [  4., 468.],
#          [  4., 476.]],

#         [[ 12.,   4.],
#          ...
#         [444., 476.]]]), 
#         tensor([[[  8.,   8.],
#             [  8.,  24.],
#             [  8.,  40.],
#         ...
#         [[440, 472]]),
#         ...
#         [[448, 448]])]


if __name__ == "__main__":
    test_point_generator()
    print("PointGenerator test passed")

