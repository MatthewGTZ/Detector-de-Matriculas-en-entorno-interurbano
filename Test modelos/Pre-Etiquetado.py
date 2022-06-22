from importlib.resources import path
import linecache
from threading import local
from tkinter.font import BOLD
import cv2
from cv2 import putText
import torch
import numpy as np
from time import time
import random

import os
import re
#cambiar esto para etiquetar matriculas o ocr

#NO DETECTA MUY BIEN CUANDO SE TRATA DE ETIQUETAR DE UNA IMAGEN ENTERA, PERO SI CUANDO LA IMAGEN ESTA RECORTADA
#ASIQUE PODRIA POR EJEMPLO RECORTAR LA IMAGEN Y EXPORTAR LA IMAGEN RECORTADA Y EN ESTA APLICAR EL OCR 
#LO PODRIA HACER PERO ME VOY A CENTRAR EN EL TFG 


model_name='bestCustomServerExp13.pt'
#model_name='best_lp.pt' 

class Detector:
    def __init__(self) -> None:
        self
        self.model = self.load_model(model_name)
        self.classes = self.model.names # especificar clases
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # mirar si es posible usar gpu cuda si esta disponible o cpu
        #self.device='cpu'
        print("Using Device: ",self.device)

        self.lineaCaracter= []
    def load_model(self, model_name):
        """Loads yolov5 model from pytorch hub
        retorna trained model  """
        if model_name:
            model = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path=model_name,force_reload=True,source='local') 
            #model = torch.hub.load('ultralytics/yolov5','custom',path=model_name,force_reload=True)#custom model esto se introducira desde la terminal como argumento 
        else:
            model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)

        return model

    def score_frame(self,image):
        self.model.to(self.device)

        #print('haber que es esto 1 : ',frame)
        image = [image] 
        #print('haber que es esto 2: ',frame)
        results = self.model(image)
        labels, cord = results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
        return labels, cord

    def class_to_label(self,x):
        return self.classes[int(x)]

    def plot_boxes(self, results,image):
        labels,cord =results

        #print(results)
        print("labels: ",labels)
        print("cordenadas: ",cord)


        n = len(labels)
        x_shape,y_shape= image.shape[1],image.shape[0] 
        for i in range(n):
            row = cord[i]
            x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
            #print(row," es la letra: ",self.class_to_label(labels[i]))
            #bgr = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            bgr =(0,255,0)
            cv2.rectangle(image,(x1,y1),(x2,y2),bgr,2)
                
            cv2.putText(image,self.class_to_label(labels[i]),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            print("Label: ",labels[i], "Corresponde al caracter: ",self.class_to_label(labels[i]))
            self.lineaCaracter.append([int(labels[i]),round(float(row[0]),6),round(float(row[1]),6),round(float(row[2]),6),round(float(row[3]),6)])
            #print(self.lineaCaracter)
        return image

    def __call__(self):
        #cap = self.get_VideoCapture()
        #assert cap.isOpened()
  
        input_images_path = "D:/Escritorio/tfg/capturas/input/MatriculasRectificadas/nu" #D:/Escritorio/tfg/capturas/lp_uc3m_ocr/images/train
        files_names = os.listdir(input_images_path)
        print(files_names)
        #output_images_path = "D:/Escritorio/tfg/capturas/output"#carpeta del OCR
        
        output_images_path = "D:/Escritorio/tfg/capturas/output/PruebaMatriculas/Resultados/1"
        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
            print("Directorio creado: ", output_images_path)
        
        count = 0
        for file_name in files_names:
       
             
            image_path = input_images_path + "/" + file_name
            print(image_path)
            image = cv2.imread(image_path)

            #image =cv2.imread("D:/Escritorio/tfg/capturas/input/00002.jpg")
            #print("entro")
            if image is None:
                continue
                
            #AQUI HACER PARA SACAR TANTO UN TXT CON LA IMAGEN ETIQUETADA EN UN FORMATO CONCRETO SUPONGO NO?

            results=self.score_frame(image)
            image = self.plot_boxes(results, image)

            ArrayList =  np.array(self.lineaCaracter)
            #file = open("D:/Escritorio/tfg/capturas/output/filename" + str(count)+".txt", "w")
            #file.write("Prueba" + os.linesep)
            with open(output_images_path+"/00" + str(count)+".txt", 'w') as file:
                for item in self.lineaCaracter:
                    #print(item)
                    item = str(item)
                    item = re.sub("\[|\]","",item) # con el import re tiene dentro funciones utiles y con esta puedo elimimar caracteres especificos del este re.sub(pattern, repl, string, count) |\
                    file.write("%s\n" % item)
            file.close()
            
            #image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(output_images_path + "/00" + str(count) + ".jpg", image)
            count += 1
            self.lineaCaracter.clear()
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

Etiquetado = Detector()
Etiquetado() ## esto ejecuta la clase como funcion eso es lo que hace __call__