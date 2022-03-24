#!/bin/bash

set -e
set -u

#name=$1
name=home_appliances

# step1: img format: RGBA -> RGB
nohup python aaai_step1.change_rgba_to_rgb.py /data/liweikang/jdsum/${name}/img > /data/liweikang/jdsum/${name}/img_format_mender.log 2>&1 &

# step2: img patch feature extract
#nohup python aaai_step2_img_to_vec.py /data/liweikang/jdsum/${name}/img_rgb/ > /data/liweikang/jdsum/${name}/img_patch_feature.log 2>&1 &