# Detector-de-Matriculas-en-entorno-interurbano
Trabajo de Fin de Grado de la universidad Carlos 3

Este repositorio esta creado para compartir el codigo desarrollado para el TFG, con la finalidad de poder continuar avanzando en el codigo para mejorarlo y para poder usarlo en futuros trabajos.

Para usar el codigo en un ordenador es necesario tener instalados todas las bibliotecas de pytorch

En este trabajo se han ido creando disitintos tipos de codigos para finalidades disitintas
1º Usando OpenCv inicialmente para comporbar que el modelo entrenado funciona correctamente, donde se podia activar en tiempo real la camara de un ordenador o introducir videos y imagenes, obteniendo por pantalla el resultado de la matricula detectada

Aun asi este trabajo se ha centrado mas en el desarrollo de un OCR, con un dataset creado desde cero, y poder unir una red encargada de detectar la matricula y otra red que rectifica la matricula detectada, na vez rectificada la matricula, entraria como lista de tensores en el programa OCR

2º Programa OCR: Necesita el archivo YoloOCRDataset para funcionar, este se encarga de hacer las trransformaciones necesarias al dataset que sea crea con las matriculas detectadas y rectificadas. Luego el Modelo OCR detecta utilizando el modelo entrenado "BestEntrenamiento.pt" los caracteres, yendo por cada matricula de forma independiente y se encarga de ordenar los caracteres para la correcta interpretacion de la matricula. Finalmente para la representacion de la matricula detectada al ser un parte de un proyecto de investigacion mas extenso, unicamente se muestra por pantalla, aun asi el resultado obtenido se podria usar facilmente para otra finalidad
