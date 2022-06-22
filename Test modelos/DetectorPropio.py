from importlib.resources import path
from threading import local
from tkinter.font import BOLD
import cv2
from cv2 import putText
import torch
import numpy as np
from time import time

class DetectorOCR:
    #
    def __init__(self,capture_index,model_name):
        self.capture_index = capture_index 
        self.model = self.load_model(model_name)
        self.classes = self.model.names # especificar clases
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # mirar si es posible usar gpu cuda si esta disponible o cpu
        #self.device='cpu'
        print("Using Device: ",self.device)


        self.listaCaracteres= [] # esto inicializo los atributos , aunque la cosa es si podria funcionar independiente de cada mascara 
   
    def get_VideoCapture(self):
        """ 
        Crea un nuevo video streaming para extraer frame por frame para hacer una prediccion
        retorna un opencv2 videocapture objeto con calidad baja 

        """
        return cv2.VideoCapture('D:\Escritorio\Cosas\Edicion\Grabaciones OBS\matricul3.mp4')
        #return cv2.VideoCapture(self.capture_index)# le pasa el parametro que elijamos podra por lo tanto ser o la camara o un URL de youtube 
    
    def load_model(self, model_name):
        """Loads yolov5 model from pytorch hub
        retorna trained model  """
        if model_name:
            model = torch.hub.load('D:/Escritorio/Prueba/yolov5-master', 'custom', path=model_name,force_reload=True,source='local') 
            #model = torch.hub.load('D:\Escritorio\Prueba\yolov5-master',model='custom',path=model_name,force_reload=True)#custom model esto se introducira desde la terminal como argumento 
        else:
            model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True) # en caso de no obtener ningun argumento por la terminal pues se ejecutara los weight de yolo por defecto supongo
        
      
       # model = torch.hub.load('ultralytics/yolov5:master','custom',path= model_name,force_reload=True)
        #model = torch.hub.load('.',path= 'GoogleColabBest1.pt',force_reload=True, source='local')
        return model
    def score_frame(self,frame):
        self.model.to(self.device)

        #print('haber que es esto 1 : ',frame)
        frame = [frame] 
        #print('haber que es esto 2: ',frame)
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:,-1],results.xyxyn[0][:,:-1]
        return labels, cord

    def class_to_label(self,x):
        return self.classes[int(x)]

    def OCR():
        pass
    def plot_boxes(self, results,frame):
        labels,cord =results
        #print('cord: ',cord,' creo que es la matriz entera con todos los datos ')
        #print('labels: ',labels,' es la posicion en la que se encuentra en el Custom_Dataset.yaml' )
        #Haber si queda claro labels es la lista y dentro estan las coordenadas y algo mas osea del 0 al 3 x1,y1,x2,y2 
        n = len(labels)
        #print('n: ',n ) #longitud del label nos sirve para recorrer las columnas con el for 
        x_shape,y_shape= frame.shape[1],frame.shape[0] 
        #print(x_shape,y_shape,' esto es como la relacion para escalar los valores al size del frame  ')

        
        """para hacer el OCR a partir de aqui"""
        #listaCaracteres = []


        for i in range(n):
            row = cord[i] #aqui lo que estoy pasando es algo
            #print('n:',row,' y el numero ',i )
            if row[4]>=0.5: #threshold o umbral por ejemplo e podido ver que en algunos llega el 5 parametro de tensor a 0.41 
#despues de este filtro, bastante importante ya que sino quedan fuera las letras que al parecer si no esta muy seguro lo considera dos 
#creo que haria falta alomejor entrenar con imagenes donde no se encuentre una matricula para evitar los falsos positivos y asi tener un threshold mas bajo 
# es creo que el porcentaje de prediccion por lo tnato queremos que sea cuanto mas alto mejor
# este valor puede mejorar consiguiendo un mejor entrenamiento supongo que cambiando los hiperparametros o mejorando el dataset

                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                #print( 'x1: ',x1,' y1 : ',y1,' x2: ',x2,' y2: ',y2,) # creo qeu es x1 y y1  esquina arriba izquierda y x2 y y2 esquina inferior derecha para completar el rectangulo 
                bgr = (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                
                cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)

                #print('Caracter matricula: ',self.class_to_label(labels[i]),) # con esto consigo el valor de la matricula 
                # AHORA BIEN 
                #como ordeno los caracteres paara ponerlos en un string teniendo en cuenta ssolo los valores que esta mostrando
                # por lo tanto tiene que hacerse denttro de esta parte del codigo 
                
                #espacio para programar 
                #IDEAS: 
                # Podrias intentar leer de izquierda a derecha opencv 

                #esto tendria que usar self.class_to_label(labels[i])

                # la i es la posicion en la que se encuentra la primera vez que detecta osea es la columna quizas a partir de las coordenadas pueda ir viendo si la anterior es mayor o menor 
                # vale pues ya entiendo la mayoria de parametros
                # tengo que ver la relacion de todos 
                # ahora a tratar de hacer el OCR 

                """PODRIA USAR X1 para hacer de menor a mayor 
                
                PERO como lo hago, como ordeno las cosas 
                dentro de aqui de este if que es un filtro importante hago un for que mire los valores que introduzco a un vector?

                haber ahora lo que se me a courrido quizas, seria 
                meter todos los caracteres que me interesan en una lista sin ordenar, luego desde otro metodo salen ordenadors 
                """
                
                #self.IntroducirALista(self.class_to_label(labels[i]),x1)
                self.listaCaracteres.append([self.class_to_label(labels[i]),x1,y1])
                #print(self.listaCaracteres)



                #print(i,' ',self.class_to_label(labels[i]))
                ## como se escribe en oorden la matricula osea escribir en un texto aparte la matricula detectada porque solo hace una deteccion 
        return frame

    def IntroducirALista(self,caracter,pos):
        self.listaCaracteres.append()
        pass

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
            #print (self.listaCaracteres)
            
            self.listaCaracteres.sort(key=lambda x:x[1],reverse=False) #esto funciona ordeno los caracteres 
            
            #sorted(self.listaCaracteres, key=lambda x:x[1],reverse=True)

            #print (self.listaCaracteres)

            #print(" ".join(map(str,self.listaCaracteres))) # esto es para convertir en string 
            #print(" ".join(map(str,self.listaCaracteres[:,0]))) # esto seria genia ls i funcionase 
            ArrayList =  np.array(self.listaCaracteres)
            #print(len(ArrayList))
            if(len(ArrayList)!=0):
            #print(ArrayList[:,0]) # lo convierto a numpy porque no se me ocurria otra manera de hacer para conseguir la maldita primera columna de 
            #print("".join(map(str,ArrayList[:,0]))) # haber si esto funciona XD
                Matricula = "".join(map(str,ArrayList[:,0]))
                posText_x,posText_y=ArrayList[0,1],ArrayList[0,2]
            #print(posText_x,posText_y,' POSICIONES')
                cv2.putText(frame,Matricula,(int(posText_x),int(posText_y)-20),cv2.FONT_HERSHEY_COMPLEX,1,(100,20,250),2) #para imprimir en pantalla lo que viene siendo la matricula 
            # quiero imprimirla encima del primer numero por ejemplo un poquito mas encima 
            
            self.listaCaracteres.clear()
            end_time = time()
            #print(end_time-start_time)
            fps = 1/np.round(end_time-start_time,2)
            cv2.putText(frame,f'FPS:{int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)
            cv2.imshow('YOLOv5 Detector',frame)
            if cv2.waitKey(5)&0xFF==27:
                break
           
        cap.release()

#Crear un nuevo objeto y ejecutar, aqui se ponen los weight que queremos ejecutar 
detector = DetectorOCR(capture_index=1,model_name='bestCustomServerExp13.pt')#GoogleColabBest1.pt
detector()


#ahora el problema que esta pasando es que cuando detecto de forma individual los caracteres estos se ordenan de forma impredecible 

"""
class OCR:
    # voy a meter un par de atributos que contrendra mi OCR 
    def __init__(self,caracter,posicion):
        self.caracter = caracter
        self.posicion = posicion
        pass
    """

    #cuando tenga mas o menos esto hare un programa mas ordenado pensando en las clases y tal ...