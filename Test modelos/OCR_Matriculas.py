##AQUI COMIENZA EL CODIGO PARA MI OCR
"""
-quiero que sea ordenado
-quiero que tenga clases 
-quiero que tenga un main en condiciones 
-quiero que sea versatil
-aun asi esto lo hare no tan de golpe
-primer oestuidare como hacer clases atributos y metodos en condiciones


"""
from importlib.resources import path

from tkinter.font import BOLD
import cv2

import torch
import numpy as np
from time import time




class OCR:
    def __init__(self,capture_index,model_name):
        self.capture_index = capture_index 
        self.model = self.load_model(model_name)
        self.classes = self.model.names # especificar clases
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        print("Using Device: ",self.device)


        self.listaCaracteres= []
    def get_VideoCapture(self):
       
        return cv2.VideoCapture('D:\Escritorio\Cosas\Edicion\Grabaciones OBS\matricul3.mp4')
        
    def get_video(self):

        
        pass


    def load_model(self, model_name):
        
        if model_name:
            model = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path=model_name,force_reload=True,source='local') 
       
        else:
            model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True) # en caso de no obtener ningun argumento por la terminal pues se ejecutara los weight de yolo por defecto supongo
        
      
       
        return model
    def score_frame(self,frame):
        self.model.to(self.device)

        
        frame = [frame] 
        
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
        return labels, cord

    def class_to_label(self,x):
        return self.classes[int(x)]

    def OCR():
        pass
    def plot_boxes(self, results,frame):
        labels,cord =results
        n = len(labels)
        x_shape,y_shape= frame.shape[1],frame.shape[0] 
        for i in range(n):
            row = cord[i] 
            if row[4]>=0.5: 
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                
                cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)

                self.listaCaracteres.append([self.class_to_label(labels[i]),x1,y1])
        
        return frame


    def __call__(self):
        cap = self.get_VideoCapture()
        assert cap.isOpened()

        while True:
            ret,frame = cap.read()
            assert ret
            frame = cv2.resize(frame,(416,416))
            start_time = time()
            results=self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
           
            
            self.listaCaracteres.sort(key=lambda x:x[1],reverse=False) 
            ArrayList =  np.array(self.listaCaracteres)
            if(len(ArrayList)!=0):
                Matricula = "".join(map(str,ArrayList[:,0]))
                posText_x,posText_y=ArrayList[0,1],ArrayList[0,2]
                cv2.putText(frame,Matricula,(int(posText_x),int(posText_y)-20),cv2.FONT_HERSHEY_COMPLEX,1,(100,20,250),2) #para imprimir en pantalla lo que viene siendo la matricula 
            
            self.listaCaracteres.clear()
            end_time = time()
            
            fps = 1/np.round(end_time-start_time,2)
            cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
            cv2.imshow('YOLOv5 Detector',frame)
            if cv2.waitKey(5)&0xFF==27:
                break
           
        cap.release()


detector = OCR(capture_index=1,model_name='bestCustomServerExp13.pt')#GoogleColabBest1.pt
detector()



