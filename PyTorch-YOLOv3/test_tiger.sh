#/bin/bash
python submit.py --image_folder $1 --model_def config/tigeryolov3.cfg --weights_path checkpoints/n_yolov3_tiger_13.pth --result $2 
