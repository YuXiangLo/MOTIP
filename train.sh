#!/bin/bash

accelerate launch \
	--num_processes=4 train.py \
	--config-path ./configs/r50_deformable_detr_motip_oinktrack.yaml
