#!/bin/bash
NUM_STEPS=25
L_SIZE=15,21
LEARNING_RATE=1e-4
PADDING_MODE='zeros'
IMAGE_SIZE=128
GRPS=1
OBJECTIVE='wainwright'

python maximize_independence.py \
	--num_steps=${NUM_STEPS} \
       	--l_sz=${L_SIZE} \
       	--learning_rate=${LEARNING_RATE} \
       	--padding_mode=${PADDING_MODE} \
       	--image_size=${IMAGE_SIZE} \
       	--groups=${GRPS} \
	--objective=${OBJECTIVE}
