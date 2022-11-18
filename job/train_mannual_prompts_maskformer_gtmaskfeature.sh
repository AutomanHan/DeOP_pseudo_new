export DETECTRON2_DATASETS=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/datasets
clipmodel="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/weights/clip/RN50.pt"
OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_learned_prompt_bs32_60k_newdata_4-verify"
OutPutDir="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/prompt/output0819/R101c_mannual_prompt_bs32_60k_clipfeature_gtmask"
trainpy=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/train_net.py
configfile=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/hancong/code/pretrain/ov-seg/zsseg.baseline/configs/coco-stuff-164k-156/zero_shot_maskformer_R101c_bs32_60k_clipfeature_gtmask_hand.yaml
CUDA_VISIBLE_DEVICES=2 python3 ${trainpy}  --config-file ${configfile}  --num-gpus 1 --eval-only --resume OUTPUT_DIR ${OutPutDir} MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME ${clipmodel}
