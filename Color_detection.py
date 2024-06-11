from PIL import Image
import cv2
from getting_details import give_Color
cap=cv2.VideoCapture(0)
green=[0,255,0] #bgr
while True:
    ret,frame=cap.read()
    hsv_image=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_limit,upper_limit=give_Color(color=green)
    mask=cv2.inRange(hsv_image,lower_limit,upper_limit)
    #convert the image into  numpy array
    mask_=Image.fromarray(mask)
    bounding_box=mask_.getbbox()
    if bounding_box!=None:
        x1,y1,x2,y2=bounding_box
        cv2.rectangle(frame,(x1-20,y1-20),(x2+20,y2+20),(0,255,0),2,cv2.FONT_HERSHEY_COMPLEX)
    cv2.imshow("Real_image",frame)
    cv2.imshow("mask",mask)
    if cv2.waitKey(40)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
