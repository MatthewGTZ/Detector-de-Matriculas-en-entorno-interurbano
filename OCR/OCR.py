from cv2 import threshold
import torch
import numpy as np

#este es mi modelo OCR 


import math

import cv2

import torch.nn as nn
import torch.optim as optin
import torch.nn.functional as F
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,transforms

import YoloOCRDataset

batch_size = 2

torch.cuda.empty_cache()# libera la memoria en cache de la gpu para ser usada

def collate_batch(batch):
    x,y_prev=map(list,zip(*batch))
    max_len=len(max(y_prev,key=len))
    x=torch.stack(x,dim=0)
    y=torch.stack([F.pad(torch.Tensor(each),(0,0,0,max_len-len(each)),value=-1)for each in y_prev],dim=0)
    return x,y

images_dir = 'D:/Escritorio/tfgdataset/PRUEBADATASETS/imagesval' # mis carpetas donde se encuentran las imagenes para testear
labels_dir = 'D:/Escritorio/tfgdataset/PRUEBADATASETS/labelsval'

img_width =2400
img_height =2400
img_transforms = transforms.Compose([transforms.Resize((img_width,img_height))]) #escalo imagenes a un tamaño grande para conseguir buena resollucion despues 
yolo_width = 640
yolo_height = 640
yolo_resize_transform = transforms.Resize((yolo_width,yolo_height))

threshold_ocr = 0.6

listaCaracteres= []

val_dataset = YoloOCRDataset.YOLODataset(images_dir,labels_dir,transform=img_transforms, preload_images=False,pad_to_aspect_ratio=1.0 )

device = 'cuda' if torch.cuda.is_available() else 'cpu'




def OrdenarCaracteres(ArrayCaracter):
   
   
    operacion = ArrayCaracter[1]+4*ArrayCaracter[2]
    print(ArrayCaracter)
    print(operacion)
    
    return operacion


def MostrarDataset():

    val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=1,collate_fn=collate_batch) #
    batch = next(iter(val_dataloader))
    images,labels = batch
    images=yolo_resize_transform(images) # resize la imagen 

    #for Cantidad in batch_size:


    images = images[1].permute(1,2,0).numpy() #aqui esta  la imagen y depende el numero segun el batch size

    model = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path='BestEntrenamiento.pt',force_reload=True,source='local')
    classes = model.names
    
    resultados = model(images)
    
    labels, cord = resultados.xyxyn[0][:,-1],resultados.xyxyn[0][:,:-1]
    n = len(labels)
    x_shape,y_shape= images.shape[0],images.shape[1]

    images = np.ascontiguousarray(images, dtype=np.uint8)
    images = cv2.cvtColor(images, cv2.COLOR_RGBA2BGR)
    
    for i in range(n):
        row = cord[i]
        if row[4]>=threshold_ocr:
            x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
            bgr = (0,255,0)
            
            cv2.rectangle(images,(x1,y1),(x2,y2),bgr,2)
                
            cv2.putText(images,classes[int(labels[i])],(x1,y1),cv2.FONT_HERSHEY_PLAIN,0.6,(0,255,0),1,)
            listaCaracteres.append([classes[int(labels[i])],x1,y1])

    listaCaracteres.sort(key= OrdenarCaracteres,reverse=False ) # ordeno la segunda columna 
    ArrayList =  np.array(listaCaracteres)

    if(len(ArrayList)!=0):
        Matricula = "".join(map(str,ArrayList[:,0]))
        print(Matricula)        
        posText_x,posText_y=ArrayList[0,1],ArrayList[0,2]
        cv2.putText(images,Matricula,(int(posText_x),int(posText_y)-20),cv2.FONT_HERSHEY_PLAIN,4,(0,0,250),2) #para imprimir en pantalla lo que viene siendo la matricula 

    listaCaracteres.clear()
    cv2.imshow("Matricula Detectada",images)

  

    while(True):
        if cv2.waitKey(5)&0xFF==27:
            break
   



if __name__ == '__main__':
    MostrarDataset()