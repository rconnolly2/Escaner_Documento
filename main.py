import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

while(webcam.isOpened()):

    _, fotograma = webcam.read()
    cv2.imshow("Webcam", fotograma)
    waitkey = cv2.waitKey(20)

    #Si presionamos la tecla "q" el programa se cierra
    if waitkey == ord("q"):
        break


cv2.destroyAllWindows()