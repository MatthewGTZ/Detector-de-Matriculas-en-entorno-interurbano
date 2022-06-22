
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()

#NO FUNCIONA QUE DIGAMOS Y NO USA PRECISAMENTE OPENCV MUY CLARAMENTE OSEA TE MUESTRA LAS COSAS DE COSAS COMO OPENCV PERO NO USA LAS MISMAS FUCNOONES
    
def run():
    source =str(1)# transforma el numero en string 
   
    weights=ROOT / 'GoogleColabBest1.pt'
    data=ROOT / 'data/Custom_Dataset.yaml'
    device='' # cuda device, i.e. 0 or 0,1,2,3 or cpu
    webcam = source.isnumeric() #parece ser una funcion para dar un bool por lo tanto si es numero pues es verdadero 
   
    #parametros que no tengo mucho en cuenta
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference

    imgsz=(640, 480)  # inference size (height, width)
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt= model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    augment=False  # augmented inference
    visualize=False
    
    
    model.model.float()

    # Dataloader
    view_img = check_imshow() # esto comprueba si es posible usar cv2.imshow ya que en google colab esta restringido
    cudnn.benchmark = True  # set True to speed up constant image size inference

    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
    bs = len(dataset)  # batch_size
        
    
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
    dt= [0.0, 0.0, 0.0]
    for im, im0s, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

         # NMS
        pred = model(im, augment=augment, visualize=visualize)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            
            
            im0= im0s[i].copy()
            s += f'{i}: '
                  
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                   

                    if view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow('LEctor', im0)
                cv2.waitKey(1)  # 1 millisecond

          
run()




