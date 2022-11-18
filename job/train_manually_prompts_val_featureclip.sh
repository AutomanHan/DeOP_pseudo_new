export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
echo ${DETECTRON2_DATASETS}
python_version==`python --version`
echo ${python_version}
clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/ViT-B-16.pt"
clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/RN50.pt"
# clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/RN101.pt"
datamap="mask_former_binary_resize_semantic"
#datamap="mask_former_binary_noaug_semantic"
imgsize=960
# python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_single_prompt_bs32_60k.yaml --num-gpus 4 --eval-only --resume 
CUDA_VISIBLE_DEVICES=2 python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_proposal_classification_bs32_10k_featureclip.yaml --num-gpus 1 --eval-only --resume MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME ${clipmodel} SOLVER.TEST_IMS_PER_BATCH 4 INPUT.DATASET_MAPPER_NAME ${datamap} INPUT.MIN_SIZE_TEST ${imgsize}
