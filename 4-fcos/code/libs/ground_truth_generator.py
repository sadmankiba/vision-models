import torch
import numpy as np
import time



def pyramid_level_cls_ground_truth(image_level_gt, points):
    
    ''' Creates a pyramid level classification ground truth from the image level ground truth'''

    points = points.long()
    batchsize, image_height, image_width  = image_level_gt.shape

    feature_shape = points.shape
    points = points.view(-1, 2)
    
    # Clamp Points
    points[:, 0] = torch.clamp(points[:, 0], min=0, max=image_height-1)
    points[:, 1] = torch.clamp(points[:, 1], min=0, max=image_width-1)
    
    cls_gt = image_level_gt[:, points[:, 0], points[:, 1]].view(batchsize, feature_shape[0], feature_shape[1])
    
    return cls_gt
    
def pyramid_level_reg_ground_truth(image_level_gt, points):
    
    ''' Creates a pyramid level regression ground truth from the image level ground truth'''
    
    points = points.long()
    batchsize, image_height, image_width, _  = image_level_gt.shape
    feature_shape = points.shape
    points = points.view(-1, 2)
    
    # Clamp Points
    points[:, 0] = torch.clamp(points[:, 0], min=0, max=image_height-1)
    points[:, 1] = torch.clamp(points[:, 1], min=0, max=image_width-1)
    
    reg_gt = image_level_gt[:, points[:, 0], points[:, 1], :].view(batchsize, feature_shape[0], feature_shape[1], 4)
    
    return reg_gt
    
def pyramid_level_ctr_ground_truth(image_level_gt, points):
    
    ''' Creates a pyramid level centerness ground truth from the image level ground truth'''
    
    points = points.long()
    batchsize, image_height, image_width  = image_level_gt.shape
    feature_shape = points.shape
    points = points.view(-1, 2)
    
    # Clamp Points
    points[:, 0] = torch.clamp(points[:, 0], min=0, max=image_height-1)
    points[:, 1] = torch.clamp(points[:, 1], min=0, max=image_width-1)
    
    ctr_gt = image_level_gt[:, points[:, 0], points[:, 1]].view(batchsize, feature_shape[0], feature_shape[1])
    
    return ctr_gt
    

def image_level_ground_truth(targets, pyramid_levels, regression_ranges, image_size, strides, radius_factor):
    
    ''' Creates the image level ground truth from the targets'''
    
    image_level_cls_gt = []
    image_level_reg_gt = []
    image_level_ctr_gt = []
    batchsize = len(targets)
    max_image_height = image_size[0]
    max_image_width = image_size[1]
    
    
    for pyramid_level in range(pyramid_levels):
        
        regression_range = regression_ranges[pyramid_level]
        min_range = regression_range[0]
        max_range = regression_range[1]
        stride = strides[pyramid_level]
        
        # Create Placeholder Variables
        cls_gt = torch.zeros((batchsize, max_image_height, max_image_width), device = strides[0].device)
        reg_gt = torch.zeros((batchsize, max_image_height, max_image_width, 4), device = strides[0].device)
        ct_gt = torch.zeros((batchsize, max_image_height, max_image_width), device=strides[0].device)
        
        
        for batch_index, target in enumerate(targets):
    
            # Get the boxes and labels for the image
            boxes, labels = target['boxes'], target['labels']
            
            # Filter out the boxes and labels that are in the regression range
            boxes, labels = filter_out_boxes(min_range, max_range, boxes, labels)
            
            # Sort Boxes
            boxes, labels = sort_boxes(boxes, labels)
            
            # Create the cls gt label
            cls_gt[batch_index, :, :] = create_image_level_cls_gt(boxes, labels, pyramid_level, (max_image_height, max_image_width), stride, radius_factor)

            
            # Create reg gt label
            reg_gt[batch_index, :, :, :], ct_gt[batch_index, :, :] = create_image_level_reg_and_ctr_gt(boxes, labels, (max_image_height, max_image_width), stride, radius_factor)

        # Normalize the regression gt
        reg_gt = reg_gt / stride
            
        # Append into the list
        image_level_cls_gt.append(cls_gt)
        image_level_reg_gt.append(reg_gt)
        image_level_ctr_gt.append(ct_gt)
    
    return image_level_cls_gt, image_level_reg_gt, image_level_ctr_gt

 
def create_image_level_cls_gt(clsboxes, labels, pyramid_level, image_size, stride, radius_factor):
    
    '''
    Creates the classification groung truth using bounding boxes and labels
    ''' 
    
    output = torch.zeros((image_size[0], image_size[1]))
    
    for box, label in zip(clsboxes, labels):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2, radius_factor = int(x1), int(y1), int(x2), int(y2), float(radius_factor)
        
        x1, y1, x2, y2 = create_internal_box(x1=x1, y1=y1, x2=x2, y2=y2, radius=stride*radius_factor)
        
        # Clamp the Bounding box between image size
        x1 = int(max(0, min(x1, image_size[1] - 1)))
        y1 = int(max(0, min(y1, image_size[0] - 1)))
        x2 = int(max(0, min(x2, image_size[1] - 1)))
        y2 = int(max(0, min(y2, image_size[0] - 1)))
        
        
        output[y1:y2, x1:x2] = label + 1
        
    return output

def create_image_level_reg_and_ctr_gt(boxes, labels, image_size, stride, radius_factor):
        
        '''
        Creates the regression groung truth using bounding boxes and labels.

        ''' 
        
        reg_output = torch.zeros((image_size[0], image_size[1], 4), dtype=torch.float32, device=stride.device)
        ctr_output = torch.zeros((image_size[0], image_size[1]), dtype=torch.float32, device=stride.device)
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2, radius_factor = int(x1), int(y1), int(x2), int(y2), float(radius_factor)
            

        
            # Clamp the Bounding box between image size
            x1 = int(max(0, min(x1, image_size[1] - 1)))
            y1 = int(max(0, min(y1, image_size[0] - 1)))
            x2 = int(max(0, min(x2, image_size[1] - 1)))
            y2 = int(max(0, min(y2, image_size[0] - 1))) 
            
            # Regression GT Calculation
            reg_output[y1:y2, x1:x2, :] = torch.tensor([x1, y1, x2, y2])
            
            x_range = torch.arange(x1, x2)
            y_range = torch.arange(y1, y2)
            yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
            yy = yy.to(stride.device)
            xx = xx.to(stride.device)
            l = xx - x1
            t = yy - y1
            r = x2 - xx
            b = y2 - yy

            # Calculate the centerness
            centerness = torch.sqrt((torch.min(l, r) / torch.max(l, r)) * (torch.min(t, b) / torch.max(t, b)))
            ctr_output[yy, xx] = centerness
            
           

        return reg_output, ctr_output


def create_internal_box(x1, y1, x2, y2, radius):
    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    x1_new = center_x - radius
    y1_new = center_y - radius
    x2_new = center_x + radius
    y2_new = center_y + radius
    
    # Clamp the new box inside the outer box
    x1_new = max(x1, x1_new)
    y1_new = max(y1, y1_new)
    x2_new = min(x2, x2_new)
    y2_new = min(y2, y2_new)
    
    return x1_new, y1_new, x2_new, y2_new
    
    
def sort_boxes(boxes, labels):
    
    """
    Sorts the boxes and labels based on the area of the boxes in descending order.
    
    Args:
        boxes (Tensor): Coordinates of ground truth boxes. Shape (N, 4).
        labels (Tensor): labels. Shape (N,).
    
    Returns:
        sorted_boxes (Tensor): List of sorted boxes. Shape (N, 4)
        sorted_labels (Tensor): List of sorted labels. Shape (N,)
    """
    
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_indices = torch.argsort(areas, descending=True)
    sorted_boxes = boxes[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    return sorted_boxes, sorted_labels

def filter_out_boxes(range_min, range_max, boxes, labels):
    """
    Filters out the boxes and labels that are beyond the regression range.
    
    Args:
        range_min (int): Minimum range for the boxes
        range_max (int): Maximum range for the boxes
        boxes (Tensor): Coordinates of ground truth boxes. Shape (N, 4).
        labels (Tensor): labels. Shape (N,).
    
    Returns:
        output_boxes (Tensor): List of boxes that are in the range. Shape (N, 4)
        output_labels (Tensor): List of labels that are in the range. Shape (N,)
    """
    
    # Iterate over boxes and check if that box is in the range
    # Approaches:
    # 1. Ensure both height and width are in the range (can miss boxes with aspect ratio > 2)
    # 2. Take max(height, width) and check if it is in the range (Current)
    x1, y1, x2, y2 = boxes.unbind(1)
    height = y2 - y1
    width = x2 - x1
    
    
    option = 2
    if option == 1:
        mask = (height > range_min) & (height <= range_max) & (width > range_min) & (width <= range_max)
    elif option == 2:
        max_dim = torch.max(torch.stack((height, width)), dim=0).values
        assert max_dim.shape == height.shape
        mask = (max_dim > range_min) & (max_dim <= range_max)

    return boxes[mask], labels[mask]
        

        
    