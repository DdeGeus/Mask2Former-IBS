# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Implement many useful :class:`Augmentation`.
"""
import sys
import torch
import numpy as np
from typing import Any, List, Optional, Tuple, Union

from PIL import Image

from fvcore.transforms.transform import (
    CropTransform,
    Transform,
    TransformList,
    NoOpTransform
)

from detectron2.data.transforms.augmentation import Augmentation, AugmentationList
from detectron2.data.transforms.augmentation_impl import RandomCrop
from detectron2.data.transforms.transform import ResizeTransform

__all__ = [
    "RandomCropWithInstance",
    "AugInput",
    "ResizeShortestEdgeAdaptive"
]


class RandomCropWithInstance(RandomCrop):
  """
  Make sure the cropping region contains the center of a random instance from annotations.
  """

  def get_transform(self, image, boxes=None):
    # boxes: list of boxes with mode BoxMode.XYXY_ABS
    h, w = image.shape[:2]
    croph, cropw = self.get_crop_size((h, w))

    assert h >= croph and w >= cropw, "Shape computation in {} has bugs.".format(self)
    offset_range_h = max(h - croph, 0)
    offset_range_w = max(w - cropw, 0)
    # Make sure there is always at least one instance in the image
    assert boxes is not None, "Can not get annotations infos."
    if len(boxes) == 0:
      h0 = np.random.randint(h - croph + 1)
      w0 = np.random.randint(w - cropw + 1)
    else:
      rand_idx = np.random.randint(0, high=len(boxes))
      bbox = torch.tensor(boxes[rand_idx])
      center_xy = (bbox[:2] + bbox[2:]) / 2.0
      offset_range_h_min = max(center_xy[1] - croph, 0)
      offset_range_w_min = max(center_xy[0] - cropw, 0)
      offset_range_h_max = max(min(offset_range_h, center_xy[1] - 1), offset_range_h_min)
      offset_range_w_max = max(min(offset_range_w, center_xy[0] - 1), offset_range_w_min)

      h0 = np.random.randint(offset_range_h_min, offset_range_h_max + 1)
      w0 = np.random.randint(offset_range_w_min, offset_range_w_max + 1)
    return CropTransform(w0, h0, cropw, croph)


def _check_img_dtype(img):
  assert isinstance(img, np.ndarray), "[Augmentation] Needs an numpy array, but got a {}!".format(
    type(img)
  )
  assert not isinstance(img.dtype, np.integer) or (
      img.dtype == np.uint8
  ), "[Augmentation] Got image of type {}, use uint8 or floating points instead!".format(
    img.dtype
  )
  assert img.ndim in [2, 3], img.ndim


class ResizeShortestEdgeAdaptive(Augmentation):
  """
  Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
  If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
  """

  def __init__(
      self,
      short_edge_length,
      max_size=sys.maxsize,
      sample_style="range",
      interp=Image.BILINEAR,
      new_sampling_shortest_side_min=1024,
      new_sampling_shortest_side_max=6192,
      new_sampling_overall_max_size=9288,
  ):
    """
    Args:
        short_edge_length (list[int]): If ``sample_style=="range"``,
            a [min, max] interval from which to sample the shortest edge length.
            If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
        max_size (int): maximum allowed longest edge length.
        sample_style (str): either "range" or "choice".
    """
    super().__init__()
    assert sample_style in ["range", "choice"], sample_style

    self.is_range = sample_style == "range"
    if isinstance(short_edge_length, int):
      short_edge_length = (short_edge_length, short_edge_length)
    if self.is_range:
      assert len(short_edge_length) == 2, (
        "short_edge_length must be two values using 'range' sample style."
        f" Got {short_edge_length}!"
      )
    self._init(locals())

  def get_transform(self, image, ratio=None):
    h, w = image.shape[:2]

    if ratio is None:
      if self.is_range:
        size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
      else:
        size = np.random.choice(self.short_edge_length)
      if size == 0:
        return NoOpTransform()

      scale = size * 1.0 / min(h, w)
      if h < w:
        newh, neww = size, scale * w
      else:
        newh, neww = scale * h, size
      if max(newh, neww) > self.max_size:
        scale = self.max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale

    else:
      scale = ratio
      min_scale = self.new_sampling_shortest_side_min / min(h, w)
      max_scale = self.new_sampling_shortest_side_max / min(h, w)
      if scale < min_scale:
        scale = min_scale
      if scale > max_scale:
        scale = max_scale
      max_size = self.new_sampling_overall_max_size

      if h < w:
        newh, neww = scale * h, scale * w
      else:
        newh, neww = scale * h, scale * w
      if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale

    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return ResizeTransform(h, w, newh, neww, self.interp)


class AugInput:
  """
  Input that can be used with :meth:`Augmentation.__call__`.
  This is a standard implementation for the majority of use cases.
  This class provides the standard attributes **"image", "boxes", "sem_seg"**
  defined in :meth:`__init__` and they may be needed by different augmentations.
  Most augmentation policies do not need attributes beyond these three.

  After applying augmentations to these attributes (using :meth:`AugInput.transform`),
  the returned transforms can then be used to transform other data structures that users have.

  Examples:
  ::
      input = AugInput(image, boxes=boxes)
      tfms = augmentation(input)
      transformed_image = input.image
      transformed_boxes = input.boxes
      transformed_other_data = tfms.apply_other(other_data)

  An extended project that works with new data types may implement augmentation policies
  that need other inputs. An algorithm may need to transform inputs in a way different
  from the standard approach defined in this class. In those rare situations, users can
  implement a class similar to this class, that satify the following condition:

  * The input must provide access to these data in the form of attribute access
    (``getattr``).  For example, if an :class:`Augmentation` to be applied needs "image"
    and "sem_seg" arguments, its input must have the attribute "image" and "sem_seg".
  * The input must have a ``transform(tfm: Transform) -> None`` method which
    in-place transforms all its attributes.
  """

  def __init__(
      self,
      image: np.ndarray,
      *,
      boxes: Optional[np.ndarray] = None,
      sem_seg: Optional[np.ndarray] = None,
      ratio = None
  ):
    """
    Args:
        image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
            floating point in range [0, 1] or [0, 255]. The meaning of C is up
            to users.
        boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
        sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
            is an integer label of pixel.
    """
    _check_img_dtype(image)
    self.image = image
    self.boxes = boxes
    self.sem_seg = sem_seg
    self.ratio = ratio

  def transform(self, tfm: Transform) -> None:
    """
    In-place transform all attributes of this class.

    By "in-place", it means after calling this method, accessing an attribute such
    as ``self.image`` will return transformed data.
    """
    self.image = tfm.apply_image(self.image)
    if self.boxes is not None:
      self.boxes = tfm.apply_box(self.boxes)
    if self.sem_seg is not None:
      self.sem_seg = tfm.apply_segmentation(self.sem_seg)

  def apply_augmentations(
      self, augmentations: List[Union[Augmentation, Transform]]
  ) -> TransformList:
    """
    Equivalent of ``AugmentationList(augmentations)(self)``
    """
    return AugmentationList(augmentations)(self)
