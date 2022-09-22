# Training prompts
export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
#CUDA_VISIBLE_DEVICES=1,2 python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/voc-11k-15/zero_shot_proposal_classification_learn_prompt_bs16_10k.yaml --num-gpus 4
python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/voc-11k-15/zero_shot_proposal_classification_learn_prompt_bs16_10k.yaml --num-gpus 1
# Training seg model
#python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS}
