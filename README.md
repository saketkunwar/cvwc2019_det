# cvcw2019_det
Tiger detection using  yolov3 and yolov3-tiny on intel open_vino inference or pytorch

Inference Openvino usage
Install openvino (https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)

and setup environment paths for openvino

cd OpenVINO-YoloV3/

For yolov3 tiny
python3 openvino_tiny-yolov3_tiger_submit.py --model ../openvino_converted_model/km_adam_yolov3_tinytiger_ckpt_9.xml --test_dir <dir-to-test> --result <result-file>

For yolov3
python3 openvino_yolov3_tiger_submit.py --model ../openvino_converted_model/n_yolov3_tiger_13.xml --test_dir <dir-to-test> --result <result-file>

------------------------------------------------------------
Inference pytorch usage

./test_tiger.sh <test-dir> <result-filename>
./test_tiger_tiny.sh <test-dir> <result-filename>

-----------------------------------------------------------
Train 
Pytorch only(https://github.com/eriklindernoren/PyTorch-YOLOv3)
cd PyTorch-YOLOv3
cd weights
./download_weights.sh
Required for imagenet pretrained weights(i.e yolov3.weights and yolov3-tiny.weights)


To train  yolov3
./train_tiger

To train yolov3 tiny
./train_tiger tiny
---------------------------------------------------------
Conversion of model from pytorch to tensorflow
please visit https://github.com/PINTO0309/OpenVINO-YoloV3  and follow the guidelines ,specifically convert_weights_pb.py.
Converion of model from tensorflow  to openvino
Follow (https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)





