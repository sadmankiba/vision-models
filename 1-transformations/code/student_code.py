import math
import random
import glob
import os
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import torch
from torch.utils import data

from utils import resize_image, load_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            return img
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    This class will resize an input image based on its shortest side.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
            Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        
        # scale the image
        if isinstance(self.size, int):
            #################################################################################
            # Fill in the code here
            #################################################################################
            if (img.shape[0] <= img.shape[1]):
                new_shape = (self.size * img.shape[1] // img.shape[0], self.size)
            else:
                new_shape = (self.size, self.size * img.shape[0] // img.shape[1])
            
            img = resize_image(img, new_shape, interpolation)
            return img
        else:
            #################################################################################
            # Fill in the code here
            #################################################################################
            img = resize_image(img, self.size, interpolation)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])


class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    This class will crop a random region with in an input image. The target area
    / aspect ratio (width/height) of the region is first sampled from a uniform
    distribution. A region satisfying the area / aspect ratio is then sampled
    and cropped. This crop is finally resized to a fixed given size. This is
    widely used as data augmentation for training image classification models.

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
            Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of areas to sample from
        ratio_range (list of int): range of aspect ratios to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

            #################################################################################
            # Fill in the code here
            #################################################################################
            # compute the width and height
            # crop the image and resize to output size
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))
            if target_w <= img.shape[1] and target_h <= img.shape[0]:
                x1 = random.randint(0, img.shape[1] - target_w)
                y1 = random.randint(0, img.shape[0] - target_h)
                img = img[y1:(y1 + target_h), x1:(x1 + target_w)]
                size = (self.size, self.size) if type(self.size) is int else self.size
                im_scale = Scale(size, interpolations=self.interpolations)
                img = im_scale(img)
                return img

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale((self.size, self.size), interpolations=self.interpolations)
            img = im_scale(img)
            #################################################################################
            # Fill in the code here
            #################################################################################
            # with a square sized output, the default is to crop the patch in the center
            # (after all trials fail)
            
            return img
        else:
            # with a pre-specified output size, the default crop is the image itself
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


class RandomColor(object):
    """Perturb color channels of a given image.

    This class will apply random color perturbation to an input image. An alpha
    value is first sampled uniformly from the range of (-r, r). 1 + alpha is
    further multiply to a color channel. The sampling is done independently for
    each channel. An efficient implementation can be achieved using a LuT.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range

    def __call__(self, img):
        #################################################################################
        # Fill in the code here
        #################################################################################
        im_f = img.astype(np.float32)
        alpha = random.uniform(-self.color_range, self.color_range)
        for i in range(3):
            img[:, :, i] = np.clip(im_f[:, :, i] * (1 + alpha), 0, 255).astype(np.uint8)

        return img

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )

@dataclass
class Point:
    x: float
    y: float

class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    This class will randomly rotate an image and further crop a local region with
    maximum area. A rotation angle is first sampled and then applied to the input.
    A region with maximum area and without any empty pixel is further determined
    and cropped.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        #################################################################################
        # Fill in the code here
        #################################################################################
        # get the rectangular with max area in the rotated image
        w_orig = img.shape[1]
        h_orig = img.shape[0]
        rot_mat = cv2.getRotationMatrix2D((w_orig//2, h_orig//2), degree, 1)
        img = cv2.warpAffine(img, rot_mat, (w_orig, h_orig), flags=cv2.INTER_LINEAR)
        
        # Intuition: The inscribed rectangle with maximum area in a rotated rectangle has 
        # same center as the rectangle bordering the rotated rectangle from outside and the 
        # sides are proportional. 
        # https://stackoverflow.com/a/7513445
        
        # Find the corner points of the rotated image
        # Assume, the center of the image is (0, 0)
        # The corner points are (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
        #  r4      r3
        # 
        #  r1      r2
        degree = degree % 360
        r1 = self._rotate_point(Point(x=-w_orig//2, y=-h_orig//2), degree / 180 * math.pi)
        r2 = self._rotate_point(Point(x=w_orig//2, y=-h_orig//2), degree / 180 * math.pi)
        r3 = self._rotate_point(Point(x=w_orig//2, y=h_orig//2), degree / 180 * math.pi)
        r4 = self._rotate_point(Point(x=-w_orig//2, y=h_orig//2), degree / 180 * math.pi)
        if degree > 270: 
            r1, r2, r3, r4 = r2, r3, r4, r1
        elif degree > 180:
            r1, r2, r3, r4 = r3, r4, r1, r2
        elif degree > 90:
            r1, r2, r3, r4 = r4, r1, r2, r3
        
        # Find the outer bounding box of the rotated image
        #   b4      b3
        # 
        #   b1      b2
        b1, b3 = self._bounding_box([r1, r2, r3, r4])
        b2 = Point(x=b3.x, y=b1.y)
        b4 = Point(x=b1.x, y=b3.y)
        
        # Find the inner rectangle with maximum area
        # The inner rectangle will have corner points on the rotated image
        # These corner points will be on the diagonal of the outer rectangle
        # The corner points with shorter distance will form the inner rectangle with max area 
        c1 = self._find_intersection_lines([r1, r4], [b1, b3])
        c2 = self._find_intersection_lines([r2, r3], [b1, b3])
        d1 = self._distance(c1, c2)
        
        c3 = self._find_intersection_lines([r1, r2], [b2, b4])
        c4 = self._find_intersection_lines([r3, r4], [b2, b4])
        d2 = self._distance(c3, c4)
        
        if d1 > d2:
            c1, c2 = c3, c4
        
        # c4      c3
        # 
        # c1      c2
        c1, c3 = self._bounding_box([c1, c2])
        c2 = Point(x=c3.x, y=c1.y)
        c4 = Point(x=c1.x, y=c3.y)
        # Transform to original image coordinates
        c2 = Point(x=c2.x + w_orig//2, y=-c2.y + h_orig//2)
        c4 = Point(x=c4.x + w_orig//2, y=-c4.y + h_orig//2)
        
        region = img[int(c4.y):int(c2.y), int(c4.x):int(c2.x)]
        return region

# An example run 
# Image shape (361, 410, 3)
# degree -25.6
# 
# Outer bounding box
# r1 Point(x=106.9, y=-251.7) r2 Point(x=262.6, y=74) r3 Point(x=-107.3, y=250.8) r4 Point(x=-263.0, y=-74.9)
# b1 Point(x=-263.0, y=-251.7) b2 Point(x=262.6, y=-251.7) b3 Point(x=262.6, y=250.8) b4 Point(x=-263.0, y=250.8)
# 
# Diagonal intersects with rotated image edges
# c1 Point(x=-139.7, y=-133.8) c2 Point(x=139.3, y=132.9) d1 386
# c3 Point(x=155.7, y=-149.5) c4 Point(x=-156.2, y=148.6) d2 431.5
# 
# Rectangle with max area
# c1 Point(x=-139.7, y=-133.8) c2 Point(x=139.3, y=-133.8) c3 Point(x=139.3, y=132.9) c4 Point(x=-139.7, y=132.9)
# c2 Point(x=344.3, y=313.8) c4 Point(x=65.3, y=47.1)
# region (266, 279, 3)

    def _find_intersection_lines(self, line1: Sequence[Point], line2: Sequence[Point]):
        def det(a: Tuple, b: Tuple):
            return a[0] * b[1] - a[1] * b[0]

        det1 = det((line1[0].x, line1[1].x), (line1[0].y, line1[1].y))
        det2 = det((line2[0].x, line2[1].x), (line2[0].y, line2[1].y))
        detb = det((line1[0].x - line1[1].x, line2[0].x - line2[1].x), (line1[0].y - line1[1].y, line2[0].y - line2[1].y))
        
        x = det((det1, line1[0].x - line1[1].x), (det2, line2[0].x - line2[1].x)) / detb
        y = det((line1[1].y - line1[0].y, det1), (line2[1].y - line2[0].y, det2)) / detb
        return Point(x=x, y=y)
    
    def _rotate_point(self, p: Point, angle: float):
        """
        Rotate a point by a given angle in radians.

        Args:
            p: A tuple of two floats (x, y) representing the point.
            angle: A float representing the angle in radians.
            
        Returns: 
            A tuple of two floats (x, y) representing the rotated point.
        """
        x = p.x * math.cos(angle) - p.y * math.sin(angle)
        y = p.x * math.sin(angle) + p.y * math.cos(angle)
        return Point(x=x, y=y)
    
    def _bounding_box(self, points):
        """
        Find the bounding box of a set of points.

        Args:
            points: A list of Point objects.
            
        Returns: 
            A tuple of two Points representing the top-left and bottom-right corners of the bounding box.
        """
        min_x = min(points, key=lambda p: p.x).x
        max_x = max(points, key=lambda p: p.x).x
        min_y = min(points, key=lambda p: p.y).y
        max_y = max(points, key=lambda p: p.y).y
        return Point(x=min_x, y=min_y), Point(x=max_x, y=max_y)
    
    def _distance(self, p1: Point, p2: Point):
        """
        Find the Euclidean distance between two points.

        Args:
            p1: A Point object.
            p2: A Point object.
            
        Returns: 
            A float representing the distance between the two points.
        """
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions. No need to modify.
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image to tensor.
    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        if img.ndim == 3:
            tensor_img = torch.from_numpy(
                np.ascontiguousarray(img.transpose((2, 0, 1)))
            )
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img


class SimpleDataset(data.Dataset):
    """
    A simple dataset using PyTorch dataloader
    """

    def __init__(self, root_folder, file_ext, transforms=None):
        # root folder, split
        self.root_folder = root_folder
        self.transforms = transforms
        self.file_ext = file_ext

        # load all labels
        file_list = glob.glob(os.path.join(root_folder, "*.{:s}".format(file_ext)))
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # load image and its label (from file name)
        filename = self.file_list[index]
        img = load_image(filename)
        label = os.path.basename(filename)
        label = label.rstrip(".{:s}".format(self.file_ext))
        # apply data augmentations
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
