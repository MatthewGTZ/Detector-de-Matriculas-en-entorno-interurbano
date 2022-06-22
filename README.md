# Detector-de-Matriculas-en-entorno-interurbano
Trabajo de Fin de Grado de la universidad Carlos 3

Este repositorio esta creado para compartir el codigo desarrollado para el TFG, con la finalidad de poder continuar avanzando en el codigo para mejorarlo y para poder usarlo en futuros trabajos.

Para usar el codigo en un ordenador es necesario tener instalados todas las bibliotecas de pytorch

![image2](https://user-images.githubusercontent.com/98813691/174993164-09349da1-cb08-49ff-8d38-be9bc0c3cca0.jpg)

En este trabajo se han ido creando distintos tipos de códigos para finalidades distintas, dentro de la carpeta de Test modelos

1º Usando OpenCv inicialmente para comprobar que el modelo entrenado funciona correctamente, donde se podía activar en tiempo real la cámara de un ordenador o introducir videos y imagenes, obteniendo por pantalla el resultado de la matrícula detectada.

2º También hay un código sin terminar para el auto etiquetado y crear labels de las imagenes, faltaria poder comparar las imagenes etiquetadas por el modelo con los XML de las matrículas.

3º También hay distintas pruebas de los distintos modelos entrenados

Para su correcto funcionamiento es necesario tener instalado YOLO v5.

Aun así este trabajo se ha centrado mas en el desarrollo de un OCR, con un dataset creado desde cero, y poder unir una red encargada de detectar la matricula y otra red que rectifica la matrícula detectada, una vez rectificada la matrícula, entraria como lista de tensores en el programa OCR

1º Programa OCR: Necesita el archivo YoloOCRDataset para funcionar, este se encarga de hacer las transformaciones necesarias al dataset que sea crea con las matriculas detectadas y rectificadas. Luego el Modelo OCR detecta utilizando el modelo entrenado "BestEntrenamiento.pt" los caracteres, yendo por cada matrícula de forma independiente y se encarga de ordenar los caracteres para la correcta interpretación de la matricula. Finalmente para la representacion de la matricula detectada al ser un parte de un proyecto de investigacion mas extenso, unicamente se muestra por pantalla, aun asi el resultado obtenido se podria usar facilmente para otra finalidad

   ![image3](https://user-images.githubusercontent.com/98813691/174993367-f19224b1-5ee4-4087-8d1a-642d95bceb46.jpg)

El código esta comentado, para poder ser interpretado de forma fácil, también indica donde hay que hacer cambios para que funcione. "Medianamente comentado :|"
