#/bin/bash
python submit.py --image_folder $1 --model_def config/tinytiger.cfg --weights_path checkpoints/km_adam_yolov3_tinytiger_ckpt_9.pth --result $2 --conf_thres 0.5
