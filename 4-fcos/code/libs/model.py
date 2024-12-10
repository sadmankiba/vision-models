import math
import torch
import torchvision
from .ground_truth_generator import pyramid_level_cls_ground_truth, pyramid_level_reg_ground_truth, pyramid_level_ctr_ground_truth, image_level_ground_truth

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn
import time

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


# Seeding
import numpy as np
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )  # keeps (H, W) the same
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        
        output = [self.cls_logits(self.conv(x[i])) for i in range(len(x))]        
        return output


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        
        x = [self.conv(x[i]) for i in range(len(x))]
        x_reg = [self.bbox_reg(x[i]) for i in range(len(x))]
        x_ctr = [self.bbox_ctrness(x[i]) for i in range(len(x))]
        return x_reg, x_ctr


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        

        start = time.time()
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)
        
       

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits, original_image_sizes
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictio ary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """
    
    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits, original_image_size, imgs = None, idx = None,
    ):
        """
        Args:
            targets (List[Dict[Tensor]]): ground-truth targets for each image
            points (List[Tensor]): 2D points on the image plane corresponding to each pyramid level pixels. 
                        Each tesnor is of shape (feature_height, feature_width, 2)
            strides (List[int]): feature strides on each pyramid level. 
            reg_range (List[Tuple[int, int]]): regression range on each pyramid level
            cls_logits (List[Tensor]): classification logits on each pyramid level. Each has shape (B, n_class, H, W)
            reg_outputs (List[Tensor]): regression outputs on each pyramid level. Each has shape (B, 4, H, W)
            ctr_logits (List[Tensor]): center-ness logits on each pyramid level. Each has shape (B, 1, H, W)
        """
        pyramid_levels = len(strides)
        safety_padding = 0
        max_height = max([size[0] for size in original_image_size]) + safety_padding
        max_width = max([size[1] for size in original_image_size]) + safety_padding

        
        # Head Output
        class_head_output = [cls_logit for cls_logit in cls_logits] # List(Tensor). Each has shape (B, n_class, H, W)
        center_head_output = [ctr_logit.squeeze(1) for ctr_logit in ctr_logits] # Sigmoind is not used as BCEWithLogitsLoss is used
        reg_head_output = [reg_out_lvl.permute(0, 2, 3, 1) for reg_out_lvl in reg_outputs] # List(Tensor). Each has shape (B, H, W, 4)
        reg_head_output_boxes = self.reg_out_boxes(reg_head_output, points, strides)
        
        
        # Image Lavel Ground Truth
        image_lvl_cls_gt, image_lvl_reg_gt, image_lvl_ctr_gt = image_level_ground_truth(targets, pyramid_levels, reg_range, (max_height, max_width), strides, self.center_sampling_radius)
        
        
        # Pyramid Level Ground Truth
        pyramid_lvl_cls_gt = [pyramid_level_cls_ground_truth(image_lvl_cls_gt[i], points[i]) for i in range(pyramid_levels)]
        pyramid_lvl_reg_gt = [pyramid_level_reg_ground_truth(image_lvl_reg_gt[i], points[i]) for i in range(pyramid_levels)]
        pyramid_lvl_ctr_gt = [pyramid_level_ctr_ground_truth(image_lvl_ctr_gt[i], points[i]) for i in range(pyramid_levels)]
        
        
        # Positive Pixel count
        positive_pixel_count = sum([(cls_gt_lvl > 0).sum() for cls_gt_lvl in pyramid_lvl_cls_gt])


        # Loss Compute
        cls_alpha = 0.25
        cls_loss = self.classification_loss(class_head_output, pyramid_lvl_cls_gt, self.num_classes, cls_alpha) / (positive_pixel_count + 1)
        reg_loss = self.regression_loss(reg_head_output_boxes, pyramid_lvl_reg_gt, pyramid_lvl_cls_gt) / (positive_pixel_count + 1)
        ctr_loss = self.centerness_loss(center_head_output, pyramid_lvl_ctr_gt, pyramid_lvl_cls_gt) / (positive_pixel_count + 1)
        
        
        # Compute the final loss
        final_loss = cls_loss + reg_loss + ctr_loss
        
        loss = {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss
        }
        
        return loss
        
        
    @staticmethod
    def reg_out_boxes(reg_head_output, points, strides):
        """
        Calculates the regression output boxes for each pixel in the feature map
        
        Args: 
            points (List[Tensor]): 2D points on the image plane corresponding to each pyramid level pixels. 
                        Each tesnor is of shape (H, W, 2)
            reg_head_output (List[Tensor]): regression outputs on each pyramid level. Each has shape (B, H, W, 4)
            
        Returns:
            reg_head_output_boxes (List[Tensor]): List of regression output boxes for each pyramid level.
        """
         
        reg_head_output_boxes = [ torch.zeros_like(reg_out_lvl) for reg_out_lvl in reg_head_output ]
        
        for point_lvl, reg_out_lvl, reg_out_boxes_lvl, stride in zip(points, reg_head_output, reg_head_output_boxes, strides):
            # x1 = x - l*s, y1 = y - t*s, x2 = x + r*s, y2 = y + b*s
            
            feature_height, feature_width = point_lvl.shape[0], point_lvl.shape[1]
            y, x = torch.meshgrid(torch.arange(0.5, feature_height+0.5), torch.arange(0.5, feature_width+0.5), indexing='ij')
            feat_point_lvl = torch.stack((y, x), dim=-1).float().to(reg_out_lvl.device)
            
            reg_out_boxes_lvl[:, :, :, 0] = -reg_out_lvl[:, :, :, 0] + feat_point_lvl[:, :, 1] 
            reg_out_boxes_lvl[:, :, :, 1] = -reg_out_lvl[:, :, :, 1] + feat_point_lvl[:, :, 0]
            reg_out_boxes_lvl[:, :, :, 2] = reg_out_lvl[:, :, :, 2] + feat_point_lvl[:, :, 1]
            reg_out_boxes_lvl[:, :, :, 3] = reg_out_lvl[:, :, :, 3]  + feat_point_lvl[:, :, 0]

        return reg_head_output_boxes
    
    @staticmethod
    def classification_loss(class_head_out, cls_gt, num_classes, alpha):
        """
        Computes the classification loss w.r.t. ground truth
        
        Args:
            class_head_out: List of classification head output for each pyramid level. Each element in the list is a tensor of shape (batch_size, num_classes, feature_height, feature_width).
            cls_gt: List of ground_truth classes for each pyramid level. Each element has shape (batch_size, feature_height, feature_width).
            num_classes: Number of classes
        """
        cls_gt_ohe = [(torch.nn.functional.one_hot(cls_gt_lvl.to(torch.int64), num_classes = num_classes + 1)[:, :, :, 1:]).permute(0, 3, 1, 2).to(class_head_out[0].device) for cls_gt_lvl in cls_gt]
        
        assert cls_gt_ohe[0].shape[1] == num_classes

        cls_loss = sum([sigmoid_focal_loss(inputs=cls_head_out_lvl, targets=cls_gt_lvl, alpha=alpha, reduction="sum") for cls_head_out_lvl, cls_gt_lvl in zip(class_head_out, cls_gt_ohe)])
        
        return cls_loss
    
    @staticmethod      
    def regression_loss(reg_head_output_boxes, reg_gt, cls_gt):
        """
        Computes the regression loss w.r.t. ground truth for the positive points
        
        Args:
            reg_head_output_boxes: List of regression head output boxes for each pyramid level. Each element in the list is a tensor of shape (B, H, W, 4).
            reg_gt: List of ground_truth regression for each pyramid level. Each element in the list is a tensor of shape (B, H, W, 4).
            cls_gt: List of ground_truth classes for each pyramid level. Each element has shape (B, H, W).
        """
        # Compute GIoU loss only for foreground points (label > 0)
        foreground_indices = [c_gt_lvl > 0 for c_gt_lvl in cls_gt]  # e.g. [tensor([ True, False]), tensor([False,  True])]
        reg_head_output_foreground = [ reg_head_boxes_lvl[fgnd_ind_lvl] for reg_head_boxes_lvl, fgnd_ind_lvl in zip(reg_head_output_boxes, foreground_indices)] # List(Tensor). Each has shape (n_pos, 4)
        reg_gt_foreground = [ reg_gt_lvl[fgnd_ind_lvl] for reg_gt_lvl, fgnd_ind_lvl in zip(reg_gt, foreground_indices) ] # List(Tensor). Each has shape (n_pos, 4)
        # Compute GIoU loss
        giou_loss_total = sum([giou_loss(boxes1=reg_head_lvl, boxes2=reg_gt_fgnd_lvl, reduction="sum") for reg_head_lvl, reg_gt_fgnd_lvl in zip(reg_head_output_foreground, reg_gt_foreground)])
        return giou_loss_total
    
    @staticmethod
    def centerness_loss(center_head_output, ctr_gt, cls_gt):
        """
        Computes the centerness loss for the positive points
        
        Args: 
            center_head_output: List of center head output for each pyramid level. Each element in the list is a tensor of shape (B, H, W).
            ctr_gt: List of ground_truth centerness for each pyramid level. Each element in the list is a tensor of shape (B, H, W).
            cls_gt: List of ground_truth classes for each pyramid level. Each element in the list is a tensor of shape (B, H, W).
        """
        # Compute Binary Cross Entropy loss only for foreground points (label > 0)
        foreground_indices = [c_gt > 0 for c_gt in cls_gt]  # e.g. [tensor([ True, False]), tensor([False,  True])]
        center_head_output_foreground = [ ctr_head[fgnd_ind] for ctr_head, fgnd_ind in zip(center_head_output, foreground_indices)]
        ctr_gt_foreground = [ ct_gt[fgnd_ind] for ct_gt, fgnd_ind in zip(ctr_gt, foreground_indices) ]
        
        # Compute Binary Cross Entropy loss
        loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        bce_loss = sum([ loss_func(ctr_head_fgnd, ct_gt_fgnd) for ctr_head_fgnd, ct_gt_fgnd in zip(center_head_output_foreground, ctr_gt_foreground) ]) 
       

        return bce_loss
      

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        

        detections = []
        batch_size = cls_logits[0].shape[0]  # Number of images in the batch

        # Loop over each image in the batch
        for batch_idx in range(batch_size):
            image_detections = []

            # Looping pyramid
            for pyramid_idx in range(len(cls_logits)):
                # (1) Compute the object scores
                cls_logits_lvl = cls_logits[pyramid_idx][batch_idx].sigmoid()  # (C x H x W)
                ctr_logits_lvl = ctr_logits[pyramid_idx][batch_idx].sigmoid()  # (1 x H x W)
                reg_outputs_lvl = reg_outputs[pyramid_idx][batch_idx]  # (4 x H x W)
                points_lvl = points[pyramid_idx]
                stride_lvl = strides[pyramid_idx]

                object_scores = torch.sqrt(cls_logits_lvl * ctr_logits_lvl)  # Combined object score

                # (2) Filter out boxes with low object scores
                object_scores_flat = object_scores.flatten(1)  # (C, H * W)
                max_scores, labels = object_scores_flat.max(dim=0)  # (H * W,)
                keep = max_scores > self.score_thresh
                if keep.sum() == 0:
                    continue

                object_scores = max_scores[keep]
                labels = labels[keep]
                reg_outputs_lvl = reg_outputs_lvl.permute(1, 2, 0).view(-1, 4)[keep]
                points_lvl = points_lvl.view(-1, 2)[keep]

                # (3) Select the top K boxes
                num_candidates = min(self.topk_candidates, object_scores.size(0))
                topk_indices = object_scores.topk(num_candidates).indices
                object_scores = object_scores[topk_indices]
                labels = labels[topk_indices]
                reg_outputs_lvl = reg_outputs_lvl[topk_indices]
                points_lvl = points_lvl[topk_indices]

                # (4) Decode the boxes
                boxes = torch.cat([
                    points_lvl[:, 1:2] - reg_outputs_lvl[:, 0:1] * stride_lvl,
                    points_lvl[:, 0:1] - reg_outputs_lvl[:, 1:2] * stride_lvl,
                    points_lvl[:, 1:2] + reg_outputs_lvl[:, 2:3] * stride_lvl,
                    points_lvl[:, 0:1] + reg_outputs_lvl[:, 3:4] * stride_lvl,
                   
                ], dim=1)

                # (5) Clip boxes outside the image boundaries
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, image_shapes[batch_idx][1])
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, image_shapes[batch_idx][0])

                # Add detections from this pyramid level
                image_detections.append((boxes, object_scores, labels))

            # (b) Collect all candidate boxes across all pyramid levels
            if len(image_detections) == 0:
                detections.append({"boxes": torch.empty((0, 4)), "scores": torch.empty((0,)), "labels": torch.empty((0,))})
                continue

            boxes, scores, labels = map(torch.cat, zip(*image_detections))

            # NMS Claswise  
            keep_boxes, keep_scores, keep_labels = [], [], []
            for class_idx in labels.unique():  # Process each class separately
                class_keep = labels == class_idx
                class_boxes = boxes[class_keep]
                class_scores = scores[class_keep]
                keep = batched_nms(class_boxes, class_scores, labels[class_keep], self.nms_thresh)
                keep_boxes.append(class_boxes[keep])
                keep_scores.append(class_scores[keep])
                keep_labels.append(labels[class_keep][keep])

            # Concatenate all kept boxes, scores, and labels across classes
            boxes = torch.cat(keep_boxes, dim=0)
            scores = torch.cat(keep_scores, dim=0)
            labels = torch.cat(keep_labels, dim=0)

            # remove extra boxes
            keep = torch.argsort(scores, descending=True)[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            detections.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels + 1  # Offset labels by +1
            })
            
        return detections

