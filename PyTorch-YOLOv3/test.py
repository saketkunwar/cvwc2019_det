from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import json

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = True

try:
    import sys
    sys.path.insert(0,'/media/saket/014178da-fdf2-462c-b901-d5f4dbce2e275/nn/apex')
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
with_apex=False


def get_results(outputs,imname,count):
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            return None, count

        output = outputs[sample_i]
        pred_boxes = output[:, :4].cpu().numpy()
        pred_boxes = rescale_boxes(pred_boxes, 416, (1080,1920)).tolist()
        pred_scores = output[:, 4].cpu().numpy().tolist()
        pred_labels = output[:, -1].cpu().numpy().tolist()
        for i,item in enumerate(pred_boxes):
            bbox=[item[0],item[1],item[2]-item[0],item[3]-item[1]]
            area = bbox[2]*bbox[3]
            anns = {
                        
                        'image_id': imname[0].split('/')[-1],
                        'category_id': 1,
                        'segmentation': [],
                        'area': area,
                        'bbox': bbox,
                        'score': pred_scores[i],
                        'iscrowd': 0
                    }
            count = count +1
        return anns, count

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    itime=[]
    results=[]
    count = len(results)
    for batch_i, (imname, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        prev_time = time.time()
        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
        current_time = time.time()
        inference_time = current_time - prev_time
        itime.append(inference_time) 
        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        anns,count = get_results(outputs,imname,count)
        if anns is not None:
            results.append(anns)
    print ('Mean inference time',np.mean(itime))
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    json.dump(results, open('output/results.json','w'))
    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/tinytiger.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/tiger.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/km_adam_yolov3_tinytiger_ckpt_9.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/tiger/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    if with_apex:
        optimizer = torch.optim.Adam(model.parameters())
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    print("Compute mAP...")
	
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )
    
    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    print(f"mAP: {AP.mean()}")
