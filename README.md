# cvcw2019_det
Tiger detection using  yolov3 and yolov3-tiny intel open_vino inference or pytorch

Openvino usage

For yolov3 tiny
python3 openvino_tiny-yolov3_tiger_submit.py --model ../openvino_converted_model/km_adam_yolov3_tinytiger_ckpt_9.xml --test_dir <dir-to-test> --result <result-file>
For yolov3
python3 openvino_yolov3_tiger_submit.py --model ../openvino_converted_model/n_yolov3_tiger_13.xml --test_dir <dir-to-test> --result <result-file>


pytorch usage

./test_tiger.sh <test-dir> <result-filename>
./test_tiger_tiny.sh <test-dir> <result-filename>

