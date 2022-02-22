#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
BATCH_SIZE=128
CROP_SIZE=48
EPOCHS=100
UPDATE_ITERS=10
DIVNORM_FSIZE=5
L_SZ="15 21"
BASE_DIR="runs"
MODEL_NAME="div_norm"
OPTIMIZER="sgd"
LEARNING_RATE=0.1
WEIGHT_DECAY=5e-4
DATA_DIR="/mnt/cube/projects/bsds500/HED-BSDS/"

for idx in 1 2 3
do
	EXPT_NAME="bsds_testing_denoising_${MODEL_NAME}_${idx}"
	python denoise_train.py \
		--batch_size ${BATCH_SIZE} \
		--crop_size ${CROP_SIZE} \
		--epochs $EPOCHS \
		--update_iters $UPDATE_ITERS \
		--divnorm_fsize $DIVNORM_FSIZE \
		--l_sz $L_SZ \
		--base_dir $BASE_DIR \
		--expt_name $EXPT_NAME \
		--learning_rate $LEARNING_RATE \
		--weight_decay $WEIGHT_DECAY \
		--data_dir $DATA_DIR \
		--model_name ${MODEL_NAME}
done
