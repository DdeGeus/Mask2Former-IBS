# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os
import json
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances, BoxMode

from .augmentations import RandomCropWithInstance, ResizeShortestEdgeAdaptive

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

from .augmentations import AugInput

__all__ = ["MaskFormerPanopticDatasetMapperCropSampling"]


class MaskFormerPanopticDatasetMapperCropSampling:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        cfg,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.cfg = cfg

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

        dataset_names = self.cfg.DATASETS.TRAIN
        self.meta = MetadataCatalog.get(dataset_names[0])

        with open(self.meta.images_per_cat_json, 'r') as fp:
            self.images_per_class_dict = json.load(fp)
        self.num_images_per_class = dict()
        for cat_id in self.images_per_class_dict['cat_ids']:
            self.num_images_per_class[cat_id] = len(self.images_per_class_dict['imgs_per_cat'][str(cat_id)])
        self.scale_options = cfg.INPUT.SCALE_OPTIONS

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if cfg.INPUT.NEW_SAMPLING:
            augs = [
                ResizeShortestEdgeAdaptive(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                    new_sampling_shortest_side_min=cfg.INPUT.NEW_SAMPLING_SHORTEST_SIDE_MIN,
                    new_sampling_shortest_side_max=cfg.INPUT.NEW_SAMPLING_SHORTEST_SIDE_MAX,
                    new_sampling_overall_max_size=cfg.INPUT.NEW_SAMPLING_OVERALL_MAX_SIZE,
                )
            ]
        else:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
        if cfg.INPUT.CROP.ENABLED:
            if not cfg.INPUT.CROP.WITH_INSTANCE:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            else:
                augs.append(
                    RandomCropWithInstance(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                    )
                )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        augs.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "cfg": cfg,
        }
        return ret

    @staticmethod
    def _convert_category_id(cat_id, meta):
        if cat_id in meta.thing_dataset_id_to_contiguous_id:
            cat_id_converted = meta.thing_dataset_id_to_contiguous_id[cat_id]
        else:
            cat_id_converted = meta.stuff_dataset_id_to_contiguous_id[cat_id]
        return cat_id_converted

    @staticmethod
    def is_thing(cat_id, meta):
        return cat_id in meta.thing_dataset_id_to_contiguous_id.values()

    def __call__(self, input_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerPanopticDatasetMapper should only be used for training!"

        output_dict = dict()
        output_dict['height'] = self.cfg.INPUT.CROP.SIZE[0]
        output_dict['width'] = self.cfg.INPUT.CROP.SIZE[1]

        scale_rand_id = np.random.randint(5)
        scale_selected = self.scale_options[scale_rand_id]

        input_dict = copy.deepcopy(input_dict)  # it will be modified by code below

        for im_i in range(2):
            # Retrieve the randomly sampled category id
            cat_id = copy.deepcopy(input_dict['cat_id'])

            # Get image_id for this category_id
            num_images_per_cat = copy.deepcopy(self.num_images_per_class[cat_id])
            image_rand_id = np.random.randint(num_images_per_cat)

            # Retrieve dataset_dict for this image_id
            image_id = self.images_per_class_dict['imgs_per_cat'][str(cat_id)][image_rand_id]
            sample_json = os.path.join(self.meta.json_per_img_dir, image_id + '.json')
            with open(sample_json, 'r') as fp:
                dataset_dict = json.load(fp)

            # convert category_ids to contiguous ids
            for segm_info in dataset_dict['segments_info']:
                segm_info['category_id'] = self._convert_category_id(segm_info['category_id'], self.meta)
            cat_id = self._convert_category_id(copy.deepcopy(cat_id), self.meta)

            # Sample a random instance or segment from the cat_id
            segments_per_cat = list()
            for segm_info in dataset_dict['segments_info']:
                if segm_info['category_id'] == cat_id:
                    segments_per_cat.append(segm_info)
            num_segments_per_cat = len(segments_per_cat)
            segm_rand_id = np.random.randint(num_segments_per_cat)
            segm_selected = segments_per_cat[segm_rand_id]

            # Get the bounding box from that particular instance
            # bbox is in format XYWH_ABS
            segm_bbox = segm_selected['bbox']
            segm_bbox_area = np.sqrt(segm_bbox[2] * segm_bbox[3])

            # Calculate rescale ratio
            if self.is_thing(cat_id, self.meta):
                im_rescale_ratio = scale_selected / segm_bbox_area
                im_rescale_ratio = np.array(im_rescale_ratio).astype(np.float32)
            else:
                im_rescale_ratio = None

            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)

            # semantic segmentation
            if "sem_seg_file_name" in dataset_dict:
                # PyTorch transformation not implemented for uint16, so converting it to double first
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            else:
                sem_seg_gt = None

            # panoptic segmentation
            if "pan_seg_file_name" in dataset_dict:
                pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
                segments_info = dataset_dict["segments_info"]
            else:
                pan_seg_gt = None
                segments_info = None

            if pan_seg_gt is None:
                raise ValueError(
                    "Cannot find 'pan_seg_file_name' for panoptic segmentation dataset {}.".format(
                        dataset_dict["file_name"]
                    )
                )

            boxes = [segm_bbox]
            boxes = np.array(boxes)
            if not len(boxes) == 0:
                boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

            aug_input = AugInput(image, sem_seg=sem_seg_gt, boxes=boxes, ratio=im_rescale_ratio)
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
            image = aug_input.image
            if sem_seg_gt is not None:
                sem_seg_gt = aug_input.sem_seg

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
            if sem_seg_gt is not None:
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            pan_seg_gt = torch.as_tensor(pan_seg_gt.astype("long"))

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                image = F.pad(image, padding_size, value=128).contiguous()
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
                pan_seg_gt = F.pad(
                    pan_seg_gt, padding_size, value=0
                ).contiguous()  # 0 is the VOID panoptic label

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = image
            if sem_seg_gt is not None:
                dataset_dict["sem_seg"] = sem_seg_gt.long()

            if "annotations" in dataset_dict:
                raise ValueError("Pemantic segmentation dataset should not have 'annotations'.")

            # Prepare per-category binary masks
            pan_seg_gt = pan_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                mask = pan_seg_gt == segment_info["id"]
                if np.sum(mask) > 1:
                    class_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        classes.append(class_id)
                        masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

            output_dict[im_i] = dataset_dict

        return output_dict
