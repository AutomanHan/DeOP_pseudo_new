export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k/model_final.pth"
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k_2/model_final.pth"
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k_3/model_final.pth"
TRAINED_PROMPTS="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/output/promptproposal/learnprompt_bs32_10k_4/model_final.pth"
echo ${TRAINED_PROMPTS}
# OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_5"
OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-clipbackbone"
python3 /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py --config-file /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_clipbackbone_R101c_bs32_60k.yaml --num-gpus 8 MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT ${TRAINED_PROMPTS} OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE False

