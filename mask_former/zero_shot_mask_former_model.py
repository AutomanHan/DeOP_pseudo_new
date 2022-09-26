# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from turtle import ScrolledCanvas
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

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
    build_prompt_learner,
)
from .mask_former_model import MaskFormer


@META_ARCH_REGISTRY.register()
class ZeroShotMaskFormer(MaskFormer):
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
            region_clip_adapter = MaskFormerClipAdapter(
                cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME,
                cls_prompt_learner,
                mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
                mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
                mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
                mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
                region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
            )

        clip_adapter = MaskFormerClipAdapter(
            cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME,
            prompt_learner,
            mask_fill=cfg.MODEL.CLIP_ADAPTER.MASK_FILL,
            mask_expand_ratio=cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO,
            mask_thr=cfg.MODEL.CLIP_ADAPTER.MASK_THR,
            mask_matting=cfg.MODEL.CLIP_ADAPTER.MASK_MATTING,
            region_resized=cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED,
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
        
        class_names = self.get_class_name_list(dataset_name)
        text_features = self.clip_adapter.get_text_features(class_names)

        # import pdb; pdb.set_trace()
        if self.clip_kd_loss:
            semseg_pred_mask = outputs["pred_masks"]
            semseg_pred_logits = outputs["pred_logits"]
            
        # import pdb; pdb.set_trace()
        # print("forward finish")

        outputs["pred_logits"] = self.clip_adapter.get_sim_logits(
            text_features, self.clip_adapter.normalize_feature(outputs["pred_logits"])
        )
        if self.training:
            

            if "aux_outputs" in outputs.keys():
                for i in range(len(outputs["aux_outputs"])):
                    outputs["aux_outputs"][i][
                        "pred_logits"
                    ] = self.clip_adapter.get_sim_logits(
                        text_features,
                        self.clip_adapter.normalize_feature(
                            outputs["aux_outputs"][i]["pred_logits"]
                        ),
                    )
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses, indicesSrc = self.criterion(outputs, targets)
            if self.clip_kd_loss:
                outputs["pred_region_logits"] = semseg_pred_logits
                # outputs["pred_masks_semseg"]
                target_regionfeature_results, target_regionfeature_valid= self.kd_region_feature(semseg_pred_mask,batched_inputs, images)
                outputs["clip_region_logits"] = target_regionfeature_results
                outputs["clip_region_valid"] = target_regionfeature_valid
                losses.update(self.kd_loss_cal(output=outputs, indices=indicesSrc))
            # import pdb; pdb.set_trace()
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            # print(losses)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = image_size[0]
                width = image_size[1]
                mask_pred_result = sem_seg_postprocess(
                    mask_pred_result, image_size, height, width
                )
                image = input_per_image["image"].to(self.device)
                # semantic segmentation inference
                r = self.semantic_inference(
                    mask_cls_result, mask_pred_result, image, class_names, dataset_name
                )
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r, "pred_masks":mask_pred_result})

                # evluate ap:
                # processed_results.append({})

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = self.panoptic_inference(
                        mask_cls_result, mask_pred_result
                    )
                    processed_results[-1]["panoptic_seg"] = panoptic_r

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