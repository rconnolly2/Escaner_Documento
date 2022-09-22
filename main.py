from audioop import reverse
from pickletools import uint8
import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

def Preprocesado(fotograma):
    lower_thresshold = 50
    upper_thresshold = 150
    kernel = np.ones((1, 1), np.uint8)

    #Tenemos que preprocesar el fotograma para posteriormente poder encontrar los contornos y => x, y de los cuatro puntos de la hoja
    fotograma_gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY) # Primero convertimos fotograma en blanco y negro
    fotograma_gris_blur = cv2.GaussianBlur(fotograma_gris, (3, 3), 0) # Aplicamos gausian blur para hacerlo borroso porque va moejor para canny edge 
    fotograma_gris_blur_CannyEdge = cv2.Canny(fotograma_gris_blur, lower_thresshold, upper_thresshold)
    fotograma_gris_blur_CannyEdge_disminuido = cv2.erode(fotograma_gris_blur_CannyEdge, kernel, iterations=2) # Reducimos el ruido con erosion y expansion
    fotograma_gris_blur_CannyEdge_disminuido_expandido = cv2.dilate(fotograma_gris_blur_CannyEdge_disminuido, kernel, iterations=1)
    return fotograma_gris_blur_CannyEdge_disminuido_expandido # Devolvemos la imagen preprocesada




def EcontrarContornos(fotograma_preprocesado):
    contornos, _ = cv2.findContours(fotograma_preprocesado, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    iterador_contornos_validos = 0
    contornos_encontrados = []
    for contorno in contornos:
    #Si el area es grande asumimos que es la hoja
        area = cv2.contourArea(contorno)
        if area > 6000:
            epilson = 0.02 * cv2.arcLength(contorno, True)
            aproximacion_contorno = cv2.approxPolyDP(contorno, epilson, True)
            contornos_encontrados.append(contorno)
            cv2.drawContours(fotograma, [aproximacion_contorno], -1, (255, 0, 0), thickness=3)
            if len(aproximacion_contorno) == 4:
                return aproximacion_contorno
  

while(webcam.isOpened()):

    _, fotograma = webcam.read()
    fotograma_procesado = Preprocesado(fotograma)
    print(EcontrarContornos(fotograma_procesado))
    cv2.imshow("Webcam", fotograma)
    waitkey = cv2.waitKey(20)

    #Si presionamos la tecla "q" el programa se cierra
    if waitkey == ord("q"):
        break
        

cv2.destroyAllWindows()