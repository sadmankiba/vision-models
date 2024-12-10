import torch

from libs.losses import sigmoid_focal_loss, giou_loss

def test_sigmoid_focal_loss():
    inputs = torch.tensor([0.5, 0.7, 0.2, 0.85])
    targets = torch.tensor([1, 0, 0, 1])
    alpha = 0.25
    gamma = 2.0

    # Test without reduction
    loss = sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="none")
    expected_loss_none = torch.tensor([0.0169, 0.3694, 0.1810, 0.0080])
    assert torch.allclose(loss, expected_loss_none, atol=1e-4), f"Expected {expected_loss_none}, but got {loss}"

    # Test with mean reduction
    loss_mean = sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="mean")
    expected_loss_mean = torch.tensor(0.1438)
    assert torch.allclose(loss_mean, expected_loss_mean, atol=1e-4), f"Expected {expected_loss_mean}, but got {loss_mean}"

    # Test with sum reduction
    loss_sum = sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="sum")
    expected_loss_sum = torch.tensor(0.5752)
    assert torch.allclose(loss_sum, expected_loss_sum, atol=1e-4), f"Expected {expected_loss_sum}, but got {loss_sum}"
    print("Sigmoid Focal Loss test passed")


def test_giou_loss():
    boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 2.0, 2.0]])
    boxes2 = torch.tensor([[0.5, 0.5, 1.5, 1.5], [1.0, 1.0, 2.0, 2.0]])
    
    # intsck = 0.25, unionk = 1.75, iouk = 0.1429, area_c = 2.25, miouk = -0.0794, loss = 1.0794
    expected_loss_none = torch.tensor([1.0794, 0.0])
    expected_loss_mean = torch.tensor(0.5397)
    expected_loss_sum = torch.tensor(1.0794)

    loss_none = giou_loss(boxes1, boxes2, reduction="none")
    loss_mean = giou_loss(boxes1, boxes2, reduction="mean")
    loss_sum = giou_loss(boxes1, boxes2, reduction="sum")

    assert torch.allclose(loss_none, expected_loss_none, atol=1e-4), f"Expected {expected_loss_none}, but got {loss_none}"
    assert torch.allclose(loss_mean, expected_loss_mean, atol=1e-4), f"Expected {expected_loss_mean}, but got {loss_mean}"
    assert torch.allclose(loss_sum, expected_loss_sum, atol=1e-4), f"Expected {expected_loss_sum}, but got {loss_sum}"

    print("GIOU Loss test passed")


if __name__ == "__main__":
    test_sigmoid_focal_loss()
    test_giou_loss()