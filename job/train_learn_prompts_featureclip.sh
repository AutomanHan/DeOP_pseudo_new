# Training prompts
export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
outputDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_featureclip_bs32_10k"
# CUDA_VISIBLE_DEVICES=1 python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_bs32_10k_featureclip.yaml --num-gpus 1 OUTPUT_DIR outputDir
python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_proposal_classification_learn_prompt_bs32_10k_featureclip.yaml --num-gpus 1 --eval-only --resume OUTPUT_DIR $outputDir
# Training seg model
#python train_net.py --config-file configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS}
