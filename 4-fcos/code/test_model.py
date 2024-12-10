import torch 

from libs.model import  FCOSClassificationHead, FCOSRegressionHead, FCOS
from libs.point_generator import PointGenerator

def test_classification_head():
    in_channels = 256
    num_classes = 20
    cls_head = FCOSClassificationHead(in_channels, num_classes)
    
    fpn_features = [
        torch.randn(4, 256, 56, 60),
        torch.randn(4, 256, 28, 30),
        torch.randn(4, 256, 14, 15),
        torch.randn(4, 256, 7, 8),
        torch.randn(4, 256, 4, 4),
    ]
    
    cls_output = cls_head(fpn_features)
    assert len(cls_output) == 5
    assert cls_output[0].shape == (fpn_features[0].shape[0], ) + (num_classes,) + fpn_features[0].shape[2:]
    assert cls_output[1].shape == (fpn_features[1].shape[0], ) + (num_classes,) + fpn_features[1].shape[2:]
    assert cls_output[2].shape == (fpn_features[2].shape[0], ) + (num_classes,) + fpn_features[2].shape[2:]
    assert cls_output[3].shape == (fpn_features[3].shape[0], ) + (num_classes,) + fpn_features[3].shape[2:]
    assert cls_output[4].shape == (fpn_features[4].shape[0], ) + (num_classes,) + fpn_features[4].shape[2:]
    print("Classification head test passed")

def test_regression_head():
    in_channels = 256
    reg_head = FCOSRegressionHead(in_channels)
    
    fpn_features = [
        torch.randn(4, 256, 56, 60),
        torch.randn(4, 256, 28, 30),
        torch.randn(4, 256, 14, 15),
        torch.randn(4, 256, 7, 8),
        torch.randn(4, 256, 4, 4),
    ]
    
    reg_output, ctr_output = reg_head(fpn_features)
    assert len(reg_output) == 5
    assert reg_output[0].shape == (fpn_features[0].shape[0],) + (4,) + fpn_features[0].shape[2:]
    assert reg_output[1].shape == (fpn_features[1].shape[0],) + (4,) + fpn_features[1].shape[2:]
    assert reg_output[2].shape == (fpn_features[2].shape[0],) + (4,) + fpn_features[2].shape[2:]
    assert reg_output[3].shape == (fpn_features[3].shape[0],) + (4,) + fpn_features[3].shape[2:]
    assert reg_output[4].shape == (fpn_features[4].shape[0],) + (4,) + fpn_features[4].shape[2:]

    assert len(ctr_output) == 5
    assert ctr_output[0].shape == (fpn_features[0].shape[0],) + (1,) + fpn_features[0].shape[2:]
    print("Regression head test passed")

def test_reg_out_boxes():
    reg_outputs = [
        torch.tensor([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]]]),
        torch.tensor([[[[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]]]])
    ]  # Shape: [(1, 2, 2, 4), (1, 2, 2, 4)]
    points = [
        torch.tensor([[[4, 4], [12, 12]], [[20, 20], [28, 28]]]),
        torch.tensor([[[8, 8], [24, 24]], [[40, 40], [56, 56]]])
    ]  # Shape: [(2, 2, 2), (2, 2, 2)]

    boxes = FCOS.reg_out_boxes(reg_outputs, points)
    expected_boxes = [
        torch.tensor([[[[3.9, 3.8, 4.3, 4.4], [11.5, 11.4, 12.7, 12.8]], [[19.1, 19.0, 21.1, 21.2], [26.7, 26.6, 29.5, 29.6]]]]),
        torch.tensor([[[[7.8, 7.7, 8.4, 8.5], [23.4, 23.3, 24.8, 24.9]], [[39.0, 38.9, 41.2, 41.3], [54.6, 54.5, 57.6, 57.7]]]])   
    ]

    assert len(boxes) == len(expected_boxes)
    for b, eb in zip(boxes, expected_boxes):
        assert torch.isclose(b, eb, atol=1e-3).all()

    print("Regression out boxes test passed")

def test_filter_out_boxes():
    boxes = torch.tensor([
        [10, 10, 15, 50],  # h, w = 5, 40
        [20, 20, 40, 40],  # h, w = 20, 20
        [30, 30, 60, 80],  # h, w = 30, 50
        [5, 5, 15, 15],    # h, w = 10, 10
    ])
    labels = torch.tensor([1, 2, 3, 4])
    range_min = 10
    range_max = 40

    filtered_boxes, filtered_labels = FCOS.filter_out_boxes(range_min, range_max, boxes, labels)

    assert len(filtered_boxes) == 2
    assert len(filtered_labels) == 2
    assert torch.equal(filtered_boxes, torch.tensor([[10, 10, 15, 50], [20, 20, 40, 40]]))
    assert torch.equal(filtered_labels, torch.tensor([1, 2]))
    print("Filter out boxes test passed")

def test_centerness():
    point = torch.Tensor([8, 14])
    box = torch.Tensor([2, 6, 20, 29])
    stride = 2
    
    # c.s = sqrt (6/12 * 8/15)
    score = FCOS.calculate_centerness(point, box, stride)
    assert torch.isclose(score, torch.tensor(0.516), atol=1e-3)
    print("Centerness test passed")

def test_calculate_gt_for_single_pixel():
    point = torch.Tensor([8, 14])
    boxes = [
        torch.Tensor([2, 6, 20, 29]),  # point in center
        torch.Tensor([6, 10, 30, 23]), # point in box, but outside center
        torch.Tensor([4, 12, 24, 28])  # point at edge of center 
    ]
    labels = [1, 2, 3]
    stride = 4
    
    reg_targets = [
        torch.Tensor([6, 8, 12, 15]) / stride,
        torch.Tensor([2, 4, 22, 9]) / stride,
        torch.Tensor([4, 2, 16, 14]) / stride
    ]

    # Case 1: No ground truth boxes
    gt_label, gt_reg, gt_centerness = FCOS.calculate_gt_for_single_pixel(point, [], [], stride)
    assert gt_label == 0
    assert gt_reg == [0, 0, 0, 0]
    assert gt_centerness == 0

    # Case 2: One ground truth box
    gt_label, gt_reg, gt_centerness = FCOS.calculate_gt_for_single_pixel(point, [boxes[0]], [labels[0]], stride)
    assert gt_label == 1
    assert torch.isclose(gt_reg, reg_targets[0], atol=1e-3).all()
    assert torch.isclose(gt_centerness, torch.tensor(0.516), atol=1e-3)

    # Case 3: Multiple ground truth boxes
    gt_label, gt_reg, gt_centerness = FCOS.calculate_gt_for_single_pixel(point, boxes, labels, stride)
    assert gt_label == 3
    assert torch.isclose(gt_reg, reg_targets[2], atol=1e-3).all()
    assert torch.isclose(gt_centerness, torch.tensor(0.189), atol=1e-3)
    print("Calculate GT for single pixel test passed")


def test_compute_gt():
    img_size = (64, 64)
    img_max_size = max(img_size)
    fpn_strides = [8, 16]
    regression_range = [(0, 32), (32, 64)]
    point_generator = PointGenerator(img_max_size, fpn_strides, regression_range)
    
    fpn_features = [
        torch.randn(2, 256, 8, 8),  # (64 / 8, 64 / 8)
        torch.randn(2, 256, 4, 4),  # (64 / 16, 64 / 16)
    ]
    
    points, strides, reg_range = point_generator(fpn_features)
    
    # batch of 2 images
    targets = [
        {
            'boxes': torch.tensor(
                [[10, 10, 30, 30], [40, 40, 60, 60]], device='cuda:0'
            ),  # H, W = (20, 20), (20, 20). Both captured in 8x8 grid
            'labels': torch.tensor([1, 2], device='cuda:0')
        },
        {
            'boxes': torch.tensor(
                [[5, 5, 20, 20], [15, 15, 60, 60]], device='cuda:0'
            ), # H, W = (15, 15), (45, 45). First one in 8x8, second in 4x4 grid 
            'labels': torch.tensor([3, 4], device='cuda:0')
        }
    ]
    # 
    # centers = [
    #    [(20, 20), (50, 50)],
    #    [(12.5, 12.5), (37.5, 37.5)]
    #  ]
    # centerboxes stride 8 = [
    #    [(8, 8, 32, 32), (38, 38, 62, 62)],
    #    [(0.5, 0.5, 24.5, 24.5), (25.5, 25.5, 49.5, 49.5)]
    # ]
    # centerboxes stride 16 = [
    #    [(-4, -4, 44, 44), (26, 26, 74, 74)],
    #    [(-11.5, -11.5, 36.5, 36.5), (13.5, 13.5, 61.5, 61.5)]
    # ]
    
    # All are lists of len(pyramid_level). Each element has shape (N, H, W).
    cls_gt, reg_gt, ctr_gt = FCOS.compute_gt(points, strides, reg_range, targets)
    
    assert len(cls_gt) == 2
    assert len(reg_gt) == 2
    assert len(ctr_gt) == 2
    
    assert cls_gt[0].shape == (2, 8, 8)
    assert reg_gt[0].shape == (2, 8, 8, 4)
    assert ctr_gt[0].shape == (2, 8, 8)
    
    assert cls_gt[1].shape == (2, 4, 4)
    assert reg_gt[1].shape == (2, 4, 4, 4)
    assert ctr_gt[1].shape == (2, 4, 4)
    
    # Pyramid level 0
    expected_cls_gt_0 = torch.tensor(
        # (4, 4), (4, 12), ...
        # (12, 4), (12, 12), ...
        [
            [[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 2, 2, 2]],
            
            [[0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]],
        ], dtype=torch.float)
    
    assert cls_gt[0].shape == expected_cls_gt_0.shape
    assert torch.equal(cls_gt[0], expected_cls_gt_0)
    
    # Some negative samples  
    assert torch.equal(reg_gt[0][0, 1, 0], torch.tensor([0, 0, 0, 0]))
    assert torch.equal(reg_gt[0][0, 4, 4], torch.tensor([0, 0, 0, 0]))
    assert torch.equal(reg_gt[0][0, 7, 4], torch.tensor([0, 0, 0, 0]))
    assert torch.equal(reg_gt[0][1, 3, 1], torch.tensor([0, 0, 0, 0]))
    assert torch.equal(reg_gt[0][1, 7, 7], torch.tensor([0, 0, 0, 0]))
    
    # Some positive samples
    assert torch.equal(reg_gt[0][0, 2, 1], torch.tensor([10, 2, 10, 18]) / 8)
    assert torch.equal(reg_gt[0][0, 6, 6], torch.tensor([12, 12, 8, 8]) / 8)
    
    # Centerness
    expected_ctr_gt_0 = torch.tensor(
        [[[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0.1111, 0.3333, 0.1111, 0, 0, 0, 0],
         [0, 0.3333, 1.0000, 0.3333, 0, 0, 0, 0],
         [0, 0.1111, 0.3333, 0.1111, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0.2500, 0.4082, 0],
         [0, 0, 0, 0, 0, 0.4082, 0.6667, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0.8750, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]],
        dtype=torch.float
    )
    # print("ctr_gt[0] == expected_ctr_gt_0", ctr_gt[0] == expected_ctr_gt_0, sep="\n")
    assert torch.isclose(ctr_gt[0], expected_ctr_gt_0, atol=1e-4).all()
    
    # Pyramid level 1
    expected_cls_gt_1 = torch.tensor(
        # (8, 8), (8, 24), ...
        # (24, 8), (24, 24), ...
        [
            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]],
            
            [[0, 0, 0, 0],
            [0, 4, 4, 4],
            [0, 4, 4, 4],
            [0, 4, 4, 4]],
        ], dtype=torch.float)
    
    assert cls_gt[1].shape == expected_cls_gt_1.shape
    assert torch.equal(cls_gt[1], expected_cls_gt_1)
    
    # Some positive samples
    assert torch.equal(reg_gt[1][1, 3, 2], torch.tensor([41, 25, 4, 20]) / 16)
    
    expected_ctr_gt_1 = torch.tensor(
        [[[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[0, 0, 0, 0],
         [0, 0.25, 0.4472, 0.1562],
         [0, 0.4472, 0.8, 0.2794],
         [0, 0.1562, 0.2794, 0.0976]]], dtype=torch.float)
    
    assert torch.isclose(ctr_gt[1], expected_ctr_gt_1, atol=1e-4).all()
    
    
    print("Compute GT test passed")

def test_classification_loss():
    cls_head_output = [
        torch.tensor([[
            [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]], 
            [[0.5, 0.6, 0.2], [0.7, 0.8, 0.1]], 
            [[0.9, 1.0, 0.0], [1.1, 1.2, 0.1]]
        ]])
    ] # 1 level. Shape: [(1, 3, 2, 3)] # (B, C, H, W)
    
    cls_gt = [
        torch.tensor([[[1, 0, 2], [0, 1, 3]]]) 
    ] # 1 level. Shape: [(1, 2, 3)] # (B, H, W)
    
    loss = FCOS.classification_loss(cls_head_output, cls_gt, num_classes=3, alpha=-1)
    expected_loss = torch.tensor(7.005)
    assert torch.isclose(loss, expected_loss, atol=1e-3)
    
    loss = FCOS.classification_loss(cls_head_output, cls_gt, num_classes=3, alpha=1)
    expected_loss = torch.tensor(0.495)
    assert torch.isclose(loss, expected_loss, atol=1e-3)
    
    loss = FCOS.classification_loss(cls_head_output, cls_gt, num_classes=3, alpha=0.25)
    expected_loss = torch.tensor(5.006)
    assert torch.isclose(loss, expected_loss, atol=1e-3)
    print("Classification loss test passed")

def test_regression_loss():
    reg_head_output = [
        torch.tensor([[[[0.1, 0.2, 0.3, 0.35], [0.5, 0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]]]),
        torch.tensor([[[[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]]]])
    ] # Shape: [(2, 2, 2, 4), (2, 2, 2, 4)]
    reg_gt = [
        torch.tensor([[[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]]]),
        torch.tensor([[[[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]], [[1.0, 1.1, 1.2, 1.3], [1.4, 1.5, 1.6, 1.7]]]])
    ] # Shape: [(1, 2, 2, 4), (1, 2, 2, 4)]
    cls_gt = [
        torch.tensor([[[1, 0], [0, 1]]]),
        torch.tensor([[[0, 1], [1, 0]]])
    ] # Shape: [(1, 2, 2), (1, 2, 2)]
    
    loss = FCOS.regression_loss(reg_head_output, reg_gt, cls_gt)
    
    # GIoU loss for matching boxes should be 0
    # giou1: iou = 0.75, area_c = 0.04, miou = 0.75, loss = 0.25
    # giou2 = 0, giou3 = 0, giou4 = 0
    assert torch.isclose(loss, torch.tensor(0.25), atol=1e-3)
    print("Compute regression loss test passed")


def test_centerness_loss():
    center_head_output = [
        torch.tensor([0.1, 0.2]),
        torch.tensor([0.5, 0.6]),
    ]
    cls_gt = [
        torch.tensor([1, 0]),
        torch.tensor([0, 1]),
    ]
    ctr_gt = [
        torch.tensor([0.7, 0]),
        torch.tensor([0, 0.4]),
    ]
    loss = FCOS.centerness_loss(center_head_output, ctr_gt, cls_gt)
    
    # BCELoss(0.1, 0.7) = 1.643
    # BCELoss(0.6, 0.4) = 0.754
    # Loss = (1.643 + 0.754) = 2.398
    assert torch.isclose(loss, torch.tensor(2.398), atol=1e-3) 
    print("Centerness loss test passed")


def test_compute_loss():
    img_size = (64, 64)
    img_max_size = max(img_size)
    fpn_strides = [8, 16]
    regression_range = [(0, 32), (32, 64)]
    point_generator = PointGenerator(img_max_size, fpn_strides, regression_range)
    
    fpn_features = [
        torch.randn(2, 256, 8, 8),  # (64 / 8, 64 / 8)
        torch.randn(2, 256, 4, 4),  # (64 / 16, 64 / 16)
    ]
    
    points, strides, reg_range = point_generator(fpn_features)
    
    # batch of 2 images
    targets = [
        {
            'boxes': torch.tensor(
                [[10, 10, 30, 30], [40, 40, 60, 60]], device='cuda:0'
            ),  # H, W = (20, 20), (20, 20). Both captured in 8x8 grid
            'labels': torch.tensor([1, 2], device='cuda:0')
        },
        {
            'boxes': torch.tensor(
                [[5, 5, 20, 20], [15, 15, 60, 60]], device='cuda:0'
            ), # H, W = (15, 15), (45, 45). First one in 8x8, second in 4x4 grid 
            'labels': torch.tensor([3, 4], device='cuda:0')
        }
    ]
    
    cls_head = FCOSClassificationHead(in_channels=256, num_classes=4)
    reg_head = FCOSRegressionHead(256)
    
    cls_logits = cls_head(fpn_features)
    reg_outputs, ctr_logits = reg_head(fpn_features)
    
    model = FCOS(
        backbone='resnet18',
        backbone_freeze_bn=True,
        backbone_out_feats=['layer3', 'layer4'],
        backbone_out_feats_dims=[512, 1024],
        fpn_feats_dim=256,
        fpn_strides=fpn_strides,
        num_classes=4,
        regression_range=regression_range,
        img_min_size=[800],
        img_max_size=1333,
        img_mean=[0.485, 0.456, 0.406],
        img_std=[0.229, 0.224, 0.225],
        train_cfg={'center_sampling_radius': 1.5},
        test_cfg={'score_thresh': 0.05, 'nms_thresh': 0.5, 'detections_per_img': 100, 'topk_candidates': 1000}
    )
    
    losses = model.compute_loss(targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits)
    assert torch.isclose(losses['cls_loss'], torch.tensor(2.56), atol=1e-2)
    assert torch.isclose(losses['reg_loss'], torch.tensor(1.94), atol=1e-2)
    assert torch.isclose(losses['ctr_loss'], torch.tensor(0.7), atol=2e-1)
    assert torch.isclose(losses['final_loss'], torch.tensor(5.2), atol=1e-1)
    
    print("Compute loss test passed")

        
if __name__ == "__main__":
    test_classification_head()
    test_regression_head()
    test_reg_out_boxes()
    test_filter_out_boxes()
    test_centerness()
    test_calculate_gt_for_single_pixel()
    test_compute_gt()
    test_classification_loss()
    test_regression_loss()
    test_centerness_loss()
    test_compute_loss()
    