#/bin/bash
echo $1
if [ $1 = "tiny" ]; then
	echo training tiny yolov3
	python train.py --pretrained_weights weights/yolov3-tiny.weights
else
	echo training yolo3
	python train.py --epochs 14 --model_def config/tigeryolov3.cfg --pretrained_weights weights/yolov3.weights
fi
