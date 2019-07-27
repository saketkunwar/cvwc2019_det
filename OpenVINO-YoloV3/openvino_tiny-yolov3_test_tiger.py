import sys, os, cv2, time
import numpy as np, math
from argparse import ArgumentParser
from PIL import Image

try:
    from armv7l.openvino.inference_engine import IENetwork, IEPlugin
except:
    from openvino.inference_engine import IENetwork, IEPlugin

m_input_size = 416

yolo_scale_13 = 13
yolo_scale_26 = 26
yolo_scale_52 = 52

classes = 1
coords = 4
num = 3
anchors = [82,86,  163,134,  277,182,  313,338,  552,378,  985,640]

LABELS = ("tiger")

label_text_color = (255, 255, 255)
label_background_color = (125, 175, 75)
box_color = (255, 128, 0)
box_thickness = 1

def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", help="Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                Sample will look for a suitable plugin for device specified (CPU by default)", default="CPU", type=str)
    return parser


def EntryIndex(side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)


class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


def IntersectionOverUnion(box_1, box_2):
    width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
    height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
    area_of_overlap = 0.0
    if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
        area_of_overlap = 0.0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1.ymax - box_1.ymin)  * (box_1.xmax - box_1.xmin)
    box_2_area = (box_2.ymax - box_2.ymin)  * (box_2.xmax - box_2.xmin)
    area_of_union = box_1_area + box_2_area - area_of_overlap
    retval = 0.0
    if area_of_union <= 0.0:
        retval = 0.0
    else:
        retval = (area_of_overlap / area_of_union)
    return retval


def ParseYOLOV3Output(blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects):

    out_blob_h = blob.shape[2]
    out_blob_w = blob.shape[3]

    side = out_blob_h
    anchor_offset = 0

    if len(anchors) == 18:   ## YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    elif len(anchors) == 12: ## tiny-YoloV3
        if side == yolo_scale_13:
            anchor_offset = 2 * 3
        elif side == yolo_scale_26:
            anchor_offset = 2 * 1

    else:                    ## ???
        if side == yolo_scale_13:
            anchor_offset = 2 * 6
        elif side == yolo_scale_26:
            anchor_offset = 2 * 3
        elif side == yolo_scale_52:
            anchor_offset = 2 * 0

    side_square = side * side
    output_blob = blob.flatten()

    for i in range(side_square):
        row = int(i / side)
        col = int(i % side)
        for n in range(num):
            obj_index = EntryIndex(side, coords, classes, n * side * side + i, coords)
            box_index = EntryIndex(side, coords, classes, n * side * side + i, 0)
            scale = output_blob[obj_index]
            if (scale < threshold):
                continue
            x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
            y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
            height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
            width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
            for j in range(classes):
                class_index = EntryIndex(side, coords, classes, n * side_square + i, coords + 1 + j)
                prob = scale * output_blob[class_index]
                if prob < threshold:
                    continue
                obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h), (original_im_w / resized_im_w))
                objects.append(obj)
    return objects


def main_IE_infer():
    fps = ""
    framepos = 0
    frame_count = 0
    vidfps = 0
    skip_frame = 0
    elapsedTime = 0
    args = build_argparser().parse_args()
    #model_xml = "lrmodels/tiny-YoloV3/FP32/frozen_tiny_yolo_v3.xml" #<--- CPU
    model_xml = "/home/saket/openvino_models/tiger/km_adam_yolov3_tinytiger_ckpt_9.xml" #<--- MYRIAD
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    '''
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    '''
    #cap = cv2.VideoCapture("data/input/testvideo.mp4")
    #camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #vidfps = int(cap.get(cv2.CAP_PROP_FPS))
    #print("videosFrameCount =", str(frame_count))
    #print("videosFPS =", str(vidfps))

    time.sleep(1)

    plugin = IEPlugin(device=args.device)
    if "CPU" in args.device:
        plugin.add_cpu_extension("lib/libcpu_extension.so")
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = plugin.load(network=net)
    import glob
    images = glob.glob('/media/saket/014178da-fdf2-462c-b901-d5f4dbce2e275/nn/PyTorch-YOLOv3/data/tigersamples/*.jpg')
    print ('Num images',len(images))
    counter = 0
    while cv2.waitKey(0):
        if cv2.waitKey(1)&0xFF == ord('q') or  counter>len(images)-1:
            break
        t1 = time.time()
        image =  np.asarray(Image.open(images[counter]).convert('RGB'))
        master = np.zeros((1920, 1920, 3),dtype='uint8')
        master[420:1500,:]=image
        image = master
        camera_height, camera_width = image.shape[0:2]
        new_w,new_h = 416,416
        resized_image = cv2.resize(image, (new_w, new_h), interpolation = cv2.INTER_CUBIC)
        canvas = np.full((m_input_size, m_input_size, 3), 0)
        canvas[(m_input_size-new_h)//2:(m_input_size-new_h)//2 + new_h,(m_input_size-new_w)//2:(m_input_size-new_w)//2 + new_w,  :] = resized_image
        prepimg = canvas
        prepimg = prepimg[np.newaxis, :, :, :]     # Batch size axis add
        prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
        #print (prepimg.shape)
        outputs = exec_net.infer(inputs={input_blob: prepimg})
        objects = []

        #output_name = detector/yolo-v3-tiny/Conv_12/BiasAdd/YoloRegion
        #output_name = detector/yolo-v3-tiny/Conv_9/BiasAdd/YoloRegion

        objects = []
        for output in outputs.values():
            objects = ParseYOLOV3Output(output,new_h, new_w, camera_height, camera_width, 0.5, objects)

        # Filtering overlapping boxes
        objlen = len(objects)
        for i in range(objlen):
            if (objects[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (IntersectionOverUnion(objects[i], objects[j]) >= 0.5):
                    if objects[i].confidence < objects[j].confidence:
                        objects[i], objects[j] = objects[j], objects[i]
                    objects[j].confidence = 0.0
        
        # Drawing boxes
        image = image[420:1500,:]
        print (image.shape)
        for obj in objects:
            if obj.confidence < 0.5:
                continue
            label = obj.class_id
            confidence = obj.confidence
            if confidence > 0.5:
                label_text = LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
                cv2.rectangle(image, (obj.xmin, obj.ymin-420), (obj.xmax, obj.ymax-420), box_color, box_thickness)
                cv2.putText(image, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_text_color, 1)

        cv2.putText(image, fps, (camera_width - 170, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Result", image)
        
        elapsedTime = time.time() - t1
        fps = "(Playback) {:.1f} FPS".format(1/elapsedTime)
        ## frame skip, video file only
        #skip_frame = int((vidfps - int(1/elapsedTime)) / int(1/elapsedTime))
        #framepos += skip_frame
        counter= counter+1

    cv2.destroyAllWindows()
    del net
    del exec_net
    del plugin


if __name__ == '__main__':
    sys.exit(main_IE_infer() or 0)

