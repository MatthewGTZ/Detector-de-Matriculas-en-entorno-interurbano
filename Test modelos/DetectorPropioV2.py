from importlib.resources import path
from threading import local
from tkinter.font import BOLD
import cv2
import torch
import numpy as np
from time import time

class DetectorOCR:
    #
    def __init__(self,capture_index,model_name,model_name2):
        self.capture_index = capture_index 
        self.model = self.load_model(model_name)
        self.model2 = self.load_model2(model_name2)
        self.classes = self.model.names # especificar clases
        self.classes2 = self.model2.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # mirar si es posible usar gpu cuda si esta disponible o cpu
        #self.device='cpu'
        print("Using Device: ",self.device)

    def get_VideoCapture(self):
        """ 
        Crea un nuevo video streaming para extraer frame por frame para hacer una prediccion
        retorna un opencv2 videocapture objeto con calidad baja 

        """
        return cv2.VideoCapture(self.capture_index)# le pasa el parametro que elijamos podra por lo tanto ser o la camara o un URL de youtube 
    
    def load_model(self, model_name):
        """Loads yolov5 model from pytorch hub
        retorna trained model  """
        if model_name:
            model = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path=model_name,force_reload=True,source='local') 
            
            #model = torch.hub.load('D:\Escritorio\Prueba\yolov5-master',model='custom',path=model_name,force_reload=True)#custom model esto se introducira desde la terminal como argumento 
        else:
            model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True) # en caso de no obtener ningun argumento por la terminal pues se ejecutara los weight de yolo por defecto supongo
        return model
        
    def score_frame(self,frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:,-1] , results.xyxyn[0][:,:-1]
        return labels, cord

    def class_to_label(self,x):
        return self.classes[int(x)]

    def plot_boxes(self, results,frameRecorte,frame):
        labels,cord =results
        
        n = len(labels)
        x_shape,y_shape= frameRecorte.shape[1],frameRecorte.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4]>=0.3:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                
                cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                #print(i,' ',self.class_to_label(labels[i]))
                ## como se escribe en oorden la matricula osea escribir en un texto aparte la matricula detectada porque solo hace una deteccion 
        return frame
    def load_model2(self,model_name2):
        modelMatriculas = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path=model_name2,force_reload=True,source='local')
       # model = torch.hub.load('ultralytics/yolov5:master','custom',path= model_name,force_reload=True)
        #model = torch.hub.load('.',path= 'GoogleColabBest1.pt',force_reload=True, source='local')
        return modelMatriculas

    def score_frame2(self,frame2):
        self.model2.to(self.device)
        frame2 = [frame2]
        results = self.model2(frame2)
        labels, cord = results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
        return labels, cord

    def class_to_label2(self,x):
        return self.classes2[int(x)]

    def plot_boxes2(self, results,frame):
        labels,cord =results
        n = len(labels)
        x_shape,y_shape= frame.shape[1],frame.shape[0]
        
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        for i in range(n):
            row = cord[i]
            if row[4]>=0.3:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                #frame = frame[y1:y2,x1:x2] # primero van las filas y luego las columnas
                cv2.putText(frame,self.class_to_label2(labels[i]),(x1,y1-3),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,0,0),1)
                
                cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)# creo una mascara rectangulos 

                #cv2.rectangle(mask[i],(x1,y1),(x2,y2),255,-1)# creo una mascara rectangulos

                
                #crear una mascara aqui 
                # con eso analizo ahi dentro lo que encuentro y lo imprimo en el frame
                


                #print(i,' ',self.class_to_label(labels[i]))
                ## como se escribe en oorden la matricula osea escribir en un texto aparte la matricula detectada porque solo hace una deteccion 
        #return frame
            
            cv2.imshow('Rectangular Mask {i}', mask[i])
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        
        
        return masked,frame


    def __call__(self):
        cap = self.get_VideoCapture()
        assert cap.isOpened()

        while True:
            ret,frame = cap.read()
            assert ret
            frame = cv2.resize(frame,(416,416))
            frameRecorte= frameOriginal = frame
            start_time = time()
            
            
            results2=self.score_frame2(frameRecorte) # detecta la matricula 
            frameRecorte,frame = self.plot_boxes2(results2,frameRecorte) 
            

            results=self.score_frame(frameRecorte)
            frame= self.plot_boxes(results, frameRecorte,frame)
            
            ##aqui me he quedado ahora lo que pasa es que cuando detecta matricula toda la imagen se hace mas pequeña dont know why
            # osea tengo que pensar esto mas detenidamente VIEJO

            # vale pues ya he conseguido separar las matriculas 
            #ahora hay algunos errores cuando detecta las estas ya qeu deteta constantemente 2 y trata de poner recuadros en donde no hay ni region ni nada de la matricula, no se muy bien porque
            # ahora bien otra cosa mas es como pongo en un puttext la matricula para que no tenga que hacer los cuadrados y tal
            # se me ocurre quizas que de alguna forma separar cada uno de las mascaras en un vector expandible quizas para que vaya por cada vectoor leyendo 
            # ESTO ULTIMO ME PARECE UNA GRANDIOSA IDEA
            # dentro de cada mascara pues hago un for para el rango del vector y ahi uno las letras en cada mascara independientemente 

            
            end_time = time()
            fps = 1/np.round(end_time-start_time,2)
            cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)

            cv2.imshow('YOLOv5 Detector',frame)
            cv2.imshow("Prueba", frameRecorte)
            if cv2.waitKey(5)&0xFF==27:
                break
           
        cap.release()


#Crear un nuevo objeto y ejecutar, aqui se ponen los weight que queremos ejecutar 
detectorOCR = DetectorOCR(capture_index=1,model_name='GoogleColabBest1.pt',model_name2='best_lp.pt')

detectorOCR()
