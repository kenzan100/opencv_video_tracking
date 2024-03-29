import sys
import cv2
import os
from numpy import *

cap = cv2.VideoCapture(os.environ['MOV_FILE_PATH'])

while(cap.isOpened()):
  ret, frame = cap.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  cv2.imshow('frame',gray)
  if  cv2.waitKey(15) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
