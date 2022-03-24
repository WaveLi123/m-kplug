#!/bin/bash

set -u
set -e

user_tag=liweikang

gpu_ids=${1-1}
cate=${2-home_appliances}
task_type=${3-mkplug}
task_tag=${4-test}
root_dir=${5-/data/${user_tag}/jdsum/}
model_name=${6-checkpoint_best}
patch_num=${7-49}
patch_embed_size=${8-2048}

export CUDA_VISIBLE_DEVICES=${gpu_ids}

CATEGORY=${cate}
TASK_TYPE=${task_type}
ROOT_DATA_DIR=${root_dir}

DATA_DIR=${ROOT_DATA_DIR}/${CATEGORY}/bin_demo
MODEL_DIR=${ROOT_DATA_DIR}/${CATEGORY}/checkpoints_${CATEGORY}_${TASK_TYPE}_${task_tag}/
INFER_MODEL=${MODEL_DIR}/${model_name}.pt

nohup fairseq-generate ${DATA_DIR} \
    --path ${INFER_MODEL} \
    --user-dir model \
    --task matchgo \
    --bertdict \
    --sku2vec-path ${DATA_DIR}/../raw_demo/ \
    --img2ids-path ${DATA_DIR}/../raw_demo/ \
    --img2vec-path ${DATA_DIR}/../raw_demo/image_patch_vectors_test.npy \
    --max-source-positions 512 --max-target-positions 512 \
    --batch-size 64 \
    --beam 5 \
    --min-len 50 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    --task_type ${TASK_TYPE} \
    --patch_num ${patch_num} \
    --patch_embed_size ${patch_embed_size} \
    > ${ROOT_DATA_DIR}/${CATEGORY}/${CATEGORY}_${TASK_TYPE}_${task_tag}_${model_name}.output.res 2>&1 &

tail -f ${ROOT_DATA_DIR}/${CATEGORY}/${CATEGORY}_${TASK_TYPE}_${task_tag}_${model_name}.output.res