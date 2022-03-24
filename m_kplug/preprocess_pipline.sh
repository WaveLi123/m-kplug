#!/bin/bash

set -u
set -b

user_tag=liweikang

VOCAB_FILE=${1-/data/${user_tag}/jdsum/vocab/vocab.jd.txt}
RAW_DIR=${2-/data/${user_tag}/jdsum/home_appliances/raw_demo/}
BPE_DIR=${3-/data/${user_tag}/jdsum/home_appliances/bpe_demo/}
BIN_DIR=${4-/data/${user_tag}/jdsum/home_appliances/bin_demo/}
SPLIT=${5-valid}

doMakeDir() {
  dir_path=$1
  if [ ! -d $dir_path ]; then
    mkdir -p $dir_path
  fi
}

doMakeDir ${BPE_DIR}

# Tokenize
begin=$(date +%s)
echo "[Tokenize Dataset Starts]"
split=${SPLIT}
for lang in src tgt; do
  python data_process/multiprocessing_bpe_encoder.py \
    --vocab-bpe ${VOCAB_FILE} \
    --inputs ${RAW_DIR}/${split}.${lang} \
    --outputs ${BPE_DIR}/${split}.${lang} \
    --workers 60
  cp -r ${RAW_DIR}/${split}.sku ${BPE_DIR}/${split}.sku
done
end=$(date +%s)
spend=$(expr $end - $begin)
echo "[Tokenize Dataset Ends] Consumes $spend S"

# Binarized data
begin=$(date +%s)
echo "[Binarized Dataset Starts]"

destdir=${BIN_DIR}

if [ ${split} == "train" ]; then

  fairseq-preprocess \
    --user-dir model \
    --task translation \
    --bertdict \
    --source-lang src \
    --target-lang tgt \
    --srcdict ${VOCAB_FILE} \
    --tgtdict ${VOCAB_FILE} \
    --trainpref ${BPE_DIR}/${split} \
    --destdir ${destdir} \
    --workers 20

elif [ ${split} == "test" ]; then
  fairseq-preprocess \
    --user-dir model \
    --task translation \
    --bertdict \
    --source-lang src \
    --target-lang tgt \
    --srcdict ${VOCAB_FILE} \
    --tgtdict ${VOCAB_FILE} \
    --testpref ${BPE_DIR}/${split} \
    --destdir ${destdir} \
    --workers 20

else

  fairseq-preprocess \
    --user-dir model \
    --task translation \
    --bertdict \
    --source-lang src \
    --target-lang tgt \
    --srcdict ${VOCAB_FILE} \
    --tgtdict ${VOCAB_FILE} \
    --validpref ${BPE_DIR}/${split} \
    --destdir ${destdir} \
    --workers 20

fi

echo "[Binarized Dataset Ends] Consumes $spend S"
