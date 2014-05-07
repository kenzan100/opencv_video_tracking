import numpy as np
import cv2
import os

cap = cv2.VideoCapture(os.environ['MOV_FILE_PATH'])

ret,old_frame = cap.read()

r,height,c,width = 240,30,390,50
track_window = (c,r,width,height)

roi = old_frame[r:r+height, c:c+width]
hsv_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0.,60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

mask = np.zeros_like(old_frame)

while(1):
  ret,frame = cap.read()

  if ret == True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

    ret,track_window = cv2.meanShift(dst, track_window, term_crit)

    x,y,w,h = track_window
    img2 = cv2.rectangle(frame,(x,y),(x+w,y+h),255,2)
    line = cv2.line(mask,(0,0),(100,100),255,2)
    # cv2.add(frame,line)
    cv2.imshow('img2',frame)

    k = cv2.waitKey(50) & 0xff
  else:
    break

cv2.destroyAllWindows()
cap.release()
