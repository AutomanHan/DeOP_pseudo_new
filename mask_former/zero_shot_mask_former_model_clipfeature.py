# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from turtle import ScrolledCanvas
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from .modeling.clip_adapter import (
    ClipAdapter,
    MaskFormerClipAdapter,
    ClipFeatureAdapter,
    MaskFormerClipFeatureAdapter,
    build_prompt_learner,
)
from .mask_former_model import MaskFormer


@META_ARCH_REGISTRY.register()
class ZeroShotMaskFormerClipFeature(MaskFormer):
    """
    Main class for zero shot mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        clipAsBackbone: bool,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        clip_adapter: nn.Module,
        region_clip_adapter: nn.Module = None,
        criterion: nn.Module,
        num_queries: int,
        panoptic_on: bool,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        clip_ensemble: bool,
        clip_ensemble_weight: float,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_kd_loss: bool,
        maskformer_hiddendim: int,
        clipKdProj: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            clip_adapter: adapter for clip-based mask classification
            num_queries: int, number of queries
            panoptic_on: bool, whether to output panoptic segmentation prediction
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__(
            clipAsBackbone=clipAsBackbone,
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            panoptic_on=panoptic_on,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        self.clip_adapter: ClipAdapter = clip_adapter
        self._region_clip_adapter = region_clip_adapter

        self.clip_ensemble: bool = clip_ensemble
        self.clip_ensemble_weight: float = clip_ensemble_weight
        self.clip_kd_loss: bool = clip_kd_loss
        self.clipKdProj: bool = clipKdProj
        if self.clipKdProj:
            self.kd_proj = nn.Linear(maskformer_hiddendim,maskformer_hiddendim)

    @classmethod
    def from_config(cls, cfg):
        init_kwargs = MaskFormer.from_config(cfg)
        prompt_learner = build_prompt_learner(cfg.MODEL.CLIP_ADAPTER)
        region_clip_adapter = None
        if cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER:
            log_first_n(
                logging.WARNING,
                "Using different head for region classification and query classification",
            )
            cls_prompt_learner = build_prompt_learner(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER
            )
            region_clip_adapter = MaskFormerClipFeatureAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
            )

        clip_adapter = MaskFormerClipFeatureAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
        )
        init_kwargs["clip_adapter"] = clip_adapter
        init_kwargs["region_clip_adapter"] = region_clip_adapter
        init_kwargs["clip_ensemble"] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE
        init_kwargs[
            "clip_ensemble_weight"
        ] = cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT

        init_kwargs["clip_kd_loss"] = cfg.MODEL.MASK_FORMER.CLIP_KD_LOSS
        init_kwargs["clipAsBackbone"] = cfg.MODEL.BACKBONE_CLIP
        init_kwargs["clipKdProj"] = cfg.MODEL.MASK_FORMER.CLIP_KD_PROJ
        init_kwargs["maskformer_hiddendim"] = cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM
        return init_kwargs

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]
        

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        '''
        # clip image encoder 作为backbone
        # import pdb; pdb.set_trace()
        if self.clipAsBackbone:
            features, det_features = self.backbone.visual(images.tensor, return_cls = False)
            # outputs = self.sem_seg_head([{"res5",features}, det_features])
            # import pdb; pdb.set_trace()
            outputs = self.sem_seg_head([features, det_features])
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)
        # features = self.backbone(images.tensor)
        '''
        
        class_names = self.get_class_name_list(dataset_name)
        text_features = self.clip_adapter.get_text_features(class_names)
        '''
        # import pdb; pdb.set_trace()
        if self.clip_kd_loss:
            semseg_pred_mask = outputs["pred_masks"]
            semseg_pred_logits = outputs["pred_logits"]
            
        # import pdb; pdb.set_trace()
        # print("forward finish")

        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )
        '''
        
        if "sem_seg" in batched_inputs[0]:
            semseg_gts = [x["sem_seg"].to(self.device) for x in batched_inputs]
        else:
            semseg_gts = None

        processed_results = []
        for input_per_image, image_size, semseg_gt in zip(
            batched_inputs, images.image_sizes, semseg_gts
        ):
            height = image_size[0]
            width = image_size[1]
            
            image = input_per_image["image"].to(self.device)
            # semantic segmentation inference
            r, semseg_masks = self.semantic_inference_gt(image, class_names, dataset_name, semseg_gt, text_features)
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = sem_seg_postprocess(r, image_size, height, width)
            processed_results.append({"sem_seg": r, "pred_masks":semseg_masks})

            return processed_results
    
    # image 经过 clip image encode
    def kd_region_feature(self, mask_pred_results,batched_inputs, images):
        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        regionfeature_results = []
        regionfeature_valid = []
        for mask_pred_result, input_per_image, image_size in zip(
             mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = image_size[0]
            width = image_size[1]
            mask_pred_result = sem_seg_postprocess(
                mask_pred_result, image_size, height, width
            )
            image = input_per_image["image"].to(self.device)
            mask_pred_result = mask_pred_result.sigmoid()
            region_features, valid_flag = self.region_clip_adapter.get_region_features(
                image, mask_pred_result, normalize=True
            )
            regionfeature_results.append(region_features)
            regionfeature_valid.append(valid_flag)
        return torch.stack(regionfeature_results), torch.stack(regionfeature_valid)

    def semantic_inference_gt(self, image, class_names, dataset_name, semseg_gt, text_features):
        semseg_masks, classes_gts = self._semseg2semmask(semseg_gt)
        # import pdb; pdb.set_trace()
        # self.semseg_masks = semseg_masks
        # clip_cls_gt, valid_flag_gt = self.region_clip_adapter(
        #         image, class_names, semseg_masks, normalize=True
        #     )

        image = (image-self.pixel_mean) / self.pixel_std
        
        images = image.unsqueeze(0)
        classes_clip=self.clip_adapter(images, class_names, semseg_masks)
        # clip_feature = self.clip_adapter.clip_model.visual(image.unsqueeze(0), return_cls = False)
        
        #########################################################
        '''
        clip_feature = clip_feature.permute(0,3,1,2)
        semseg_feature = []
        clip_feature_tmp = clip_feature.squeeze(0)
        for semseg_mask in semseg_masks_downsample.squeeze(0):
            # import pdb; pdb.set_trace()
            clip_feature_mask = clip_feature_tmp * semseg_mask
            mask_sum = (clip_feature_tmp > 0.5).sum()
            clip_feature_sum = clip_feature_mask.sum(dim=[1,2]) / mask_sum
            semseg_feature.append(clip_feature_sum)
        clip_maskfeature = torch.stack(semseg_feature)
        '''
        #########################################################
        
        if(len(classes_clip.shape)==1):
            classes_clip= classes_clip.unsqueeze(0)
        # import pdb; pdb.set_trace()
        classes_clip = F.softmax(classes_clip, dim = -1)
        semseg_gt_out = torch.einsum("qc,qhw->chw", classes_clip, semseg_masks)
        
        
        # clip_cls_gt = F.softmax(clip_cls_gt[:, :-1], dim=-1)


        #########################################################
        # # gtmask + gt cls
        # classes_gts_onehot = F.one_hot(classes_gts, num_classes=171)
        # semseg_gt_out = torch.einsum("qc,qhw->chw", classes_gts_onehot.to(torch.float32), semseg_masks)
        #########################################################
        # semseg_gt_out = torch.einsum("qc,qhw->chw", clip_cls_gt, semseg_masks)
        return semseg_gt_out, semseg_masks

    def semantic_inference(self, mask_cls, mask_pred, image, class_names, dataset_name):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        # get the classification result from clip model
        if self.clip_ensemble:
            clip_cls, valid_flag = self.region_clip_adapter(
                image, class_names, mask_pred, normalize=True
            )
            if clip_cls is None:
                clip_cls = torch.empty(0, mask_cls.shape[-1] + 1, device=self.device)
            # import pdb; pdb.set_trace()
            # softmax before index or after?
            clip_cls = F.softmax(clip_cls[:, :-1], dim=-1)
            if self.clip_ensemble_weight > 0:
                map_back_clip_cls = mask_cls.new_ones(mask_cls.shape)
                map_back_clip_cls[valid_flag] = clip_cls
                if hasattr(MetadataCatalog.get(dataset_name), "trainable_flag"):
                    trained_mask = torch.Tensor(
                        MetadataCatalog.get(dataset_name).trainable_flag
                    ).to(mask_cls.device)[None, :]
                else:
                    trained_mask = mask_cls.new_zeros(mask_cls.shape)
                mask_cls = trained_mask * torch.pow(
                    mask_cls, self.clip_ensemble_weight
                ) * torch.pow(map_back_clip_cls, 1 - self.clip_ensemble_weight) + (
                    1 - trained_mask
                ) * torch.pow(
                    mask_cls, 1 - self.clip_ensemble_weight
                ) * torch.pow(
                    map_back_clip_cls, self.clip_ensemble_weight
                )
            else:
                # only clip model predictions are used
                mask_cls = clip_cls
                mask_pred = mask_pred[valid_flag]
                mask_pred[mask_pred > 0.5] = 1.0
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).stuff_classes
        ]
        return class_names

    @property
    def region_clip_adapter(self):
        if self._region_clip_adapter is None:
            return self.clip_adapter
        return self._region_clip_adapter

    def kd_loss_cal(self, output, indices):
        semseg_pred_logits =output["pred_region_logits"]
        target_regionfeature_results=output["clip_region_logits"]
        # import pdb; pdb.set_trace()
        if self.clipKdProj:
            semseg_pred_logits = self.kd_proj(semseg_pred_logits)
        target_regionfeature_valid = output["clip_region_valid"]
        # import pdb; pdb.set_trace()
        # semseg_pred_logits = semseg_pred_logits[target_regionfeature_valid]
        # for idx in range(semseg_pred_logits.shape[0]):
        #     semseg_pred_logits[idx] = semseg_pred_logits[idx][target_regionfeature_valid[idx]]
        #     target_regionfeature_results[idx] = target_regionfeature_results[idx][target_regionfeature_valid[idx]]
        # import pdb; pdb.set_trace()
        # target_regionfeature_results = target_regionfeature_results[target_regionfeature_valid]
        # if()
        src_idx = self._get_src_permutation_idx(indices)
        loss_kd = F.l1_loss(semseg_pred_logits[src_idx], target_regionfeature_results[src_idx])
        # if target_regionfeature_valid.sum() > 0:
        #     loss_kd = F.l1_loss(semseg_pred_logits[target_regionfeature_valid],target_regionfeature_results[target_regionfeature_valid])
        # else:
        #     loss_kd = F.l1_loss(semseg_pred_logits,target_regionfeature_results)
        return {"loss_kd": loss_kd}
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    
    def _semseg2semmask(self,semseg_gt):
        sem_seg_gt = semseg_gt.cpu().numpy()
        
        classes = np.unique(sem_seg_gt)
        classes = classes[classes != 255]
        masks = []
        classes_gt = []
        for class_id in classes:
            mask = np.zeros_like(semseg_gt.cpu())
            mask[sem_seg_gt == class_id] = 1.0
            masks.append(mask)
            classes_gt.append(class_id)
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            masks= torch.zeros(
                (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
            )
        else:
            masks=torch.stack(
                [
                    torch.from_numpy(np.ascontiguousarray(x.copy()))
                    for x in masks
                ]
            )        
        return masks.to(torch.float32).to(semseg_gt.device), torch.tensor(classes_gt).to(torch.int64).to(semseg_gt.device)
    def get_sim_logits(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        temperature: float = 100,
    ):
        return temperature * image_features @ text_features.T