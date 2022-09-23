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

def ReOrdenar(contornos_mas_grandes):
    if not np.all(contornos_mas_grandes) == 0:
        if contornos_mas_grandes.size < 10:
            Contornos = contornos_mas_grandes.reshape((4, 2))
            Sumatorio_Contornos = np.sum(Contornos, axis=1)
            NuevosContornos = np.zeros((4, 1, 2), np.int32)
            # Primer contorno:
            NuevosContornos[0] = Contornos[np.argmin(Sumatorio_Contornos)]
            # Ultimo contorno:
            NuevosContornos[3] = Contornos[np.argmax(Sumatorio_Contornos)]
            # # Buscamos el dato mayor los 2 arrays restantes y lo restamos a los menores para obtener la DIFERENCIA
            # # La diferencia mas pequeña es el segundo dato para Nuevoscontornos y la mas grande el tercero
            diferencias = []
            for contornos in Contornos:
                diferencia = (contornos[0]) - (contornos[1])
                diferencias.append(diferencia)

            # Segundo contorno:
            NuevosContornos[1] = Contornos[diferencias.index(max(diferencias))]
            # Tercer contorno:
            NuevosContornos[2] = Contornos[diferencias.index(min(diferencias))]
            # Retornamos nueva array ordenada correctamente:
            return NuevosContornos
    else:
        return contornos_mas_grandes



def WarpPerspective(imagen, contornos_mas_grandes):
    anchura = imagen.shape[0]
    altura = imagen.shape[1]

    contornosmasgrandes_ordenados = ReOrdenar(contornos_mas_grandes)
    pt1 = np.float32(contornosmasgrandes_ordenados)
    pt2 = np.float32([[0, 0], [anchura, 0], [0, altura], [anchura, altura]])
    matriz = cv2.getPerspectiveTransform(pt1, pt2)
    Imagen_Warp = cv2.warpPerspective(imagen, matriz, (anchura, altura))
    return Imagen_Warp



def EcontrarContornos(fotograma_preprocesado):
    aproximacion_contorno = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
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
    return aproximacion_contorno
  

while(webcam.isOpened()):

    _, fotograma = webcam.read()

    fotograma_procesado = Preprocesado(fotograma)
    Contornos_mas_grandes = EcontrarContornos(fotograma_procesado)
    Imagen_warp = WarpPerspective(fotograma, Contornos_mas_grandes)

    cv2.imshow("Webcam", fotograma)
    cv2.imshow("Hoja detectada", Imagen_warp)
    waitkey = cv2.waitKey(20)

    #Si presionamos la tecla "q" el programa se cierra
    if waitkey == ord("q"):
        break
        

cv2.destroyAllWindows()