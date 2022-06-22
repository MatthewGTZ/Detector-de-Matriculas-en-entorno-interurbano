import CustomOCRDataset
import torch
import torch.nn.functional as F

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms

import os

def collate_batch(batch):
    x,y_prev=map(list,zip(*batch))
    max_len=len(max(y_prev,key=len))
    x=torch.stack(x,dim=0)
    y=torch.stack([F.pad(torch.Tensor(each),(0,0,0,max_len-len(each)),value=-1)for each in y_prev],dim=0)
    return x,y

ocr_model = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path='bestCustomServerExp13.pt',force_reload=True,source='local')

images_dir = os.path.join('D:/Escritorio/Prueba/Datasets/lp_uc3m_ocr/images/val' ) #el profe  lo tiene como (ROOT_DIR,'DATASET,IMAGES,VAL)
labels_dir = os.path.join('D:/Escritorio\Prueba\Datasets/lp_uc3m_ocr/labels/val')

img_width =2400
img_height =2400
img_transforms = transforms.Compose([transforms.Resize((img_width,img_height))]) #escalo imagenes a un tamaño grande para conseguir buena resollucion despues 

yolo_width = 640
yolo_height = 640
yolo_resize_transform = transforms.Resize((yolo_width,yolo_height))

val_dataset = CustomOCRDataset.YOLODataset(images_dir,labels_dir,transform=img_transforms, preload_images=False,pad_to_aspect_ratio=1.0 )

batch_size =8
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=8,collate_fn=collate_batch)





device = torch.device('cuda')

ocr_conf_threshold=0.5#esto es un umbral para el porcentaje de seguridad del modelo
ocr_iou_threshold=0.5# esto es para evitar los repetidos 
val_iter=iter(val_dataloader)
images,lp_labels= next(val_iter)
images=(images/255.0).to(device)

images.shape

plt.imshow(images[0].permute(1,2,0))
