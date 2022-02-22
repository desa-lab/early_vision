#!/bin/bash
BATCH_SIZE=256
CROP_SIZE=48
DOWNSAMPLE=4
EPOCHS=100
UPDATE_ITERS=10
DIVNORM_FSIZE=5
L_SZ="15 21"
BASE_DIR="runs_superres_sony"
MODEL_NAME="div_norm"
OPTIMIZER="sgd"
LEARNING_RATE=0.1
WEIGHT_DECAY=5e-4
DATA_DIR="/mnt/cube/projects/bsds500/HED-BSDS/"
TRANSFORM="superres"

# for idx in 1 2 3
for idx in 1
do
	EXPT_NAME="bsds_denoising_downsample_${DOWNSAMPLE}_${MODEL_NAME}_${idx}_transform_${TRANSFORM}_$1"
	python denoise_train.py \
		--batch_size ${BATCH_SIZE} \
		--crop_size ${CROP_SIZE} \
		--downsample ${DOWNSAMPLE} \
		--epochs $EPOCHS \
		--update_iters $UPDATE_ITERS \
		--divnorm_fsize $DIVNORM_FSIZE \
		--l_sz $L_SZ \
		--transform ${TRANSFORM} \
		--base_dir $BASE_DIR \
		--expt_name $EXPT_NAME \
		--learning_rate $LEARNING_RATE \
		--weight_decay $WEIGHT_DECAY \
		--data_dir $DATA_DIR \
		--model_name ${MODEL_NAME}
done
