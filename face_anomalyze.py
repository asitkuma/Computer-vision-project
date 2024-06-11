import cv2
import mediapipe as mp
import os
import argparse
output_dir='./output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
img_read=os.path.join('.','images','fam.jpg')
img_read=cv2.imread(img_read)
img=cv2.resize(img_read,(650,700))
height,width,_=img.shape
mp_face_detection=mp.solutions.face_detection
#here model_selection=0 means it can detect the face which are 2m than camera if beyound 5m u want to detect the face then u must need to set 1.

def process_image(img,face_detection):
    height,width,_=img.shape
    converted_image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    all_details=face_detection.process(converted_image)
    if all_details.detections!=None:
        for detection in all_details.detections:
            location_data=detection.location_data.relative_bounding_box
            x,y,w,h=location_data.xmin,location_data.ymin,location_data.width,location_data.height
            x=int(x*width) #these values are unwrapped so, first we will convert those values.
            y=int(y*height)
            w=int(w*width)
            h=int(h*height)
            img[y:y+h,x:x+w,:]=cv2.blur(img[y:y+h,x:x+w,:],(143,143)) 
    return img


args=argparse.ArgumentParser()

# for image
# args.add_argument("--mode",default='image')
# args.add_argument("--filePath",default='./images/fam.jpg')



# for video
# args.add_argument("--mode",default='video')
# args.add_argument("--filePath",default='./images/vid_video.mp4')

#for webcam
# args.add_argument("--mode",default='webcam')

args=args.parse_args()

with mp_face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5) as face_detection:
    if args.mode in ['image']:
        img=cv2.imread(args.filePath)
        img_process=process_image(img,face_detection)
        cv2.imwrite(os.path.join(output_dir,'blurred_image.jpg'),img_process)
    elif args.mode=="video":
        cap=cv2.VideoCapture('./images/vid_video.mp4')
        ret,frame=cap.read() # read first frame..
        # cv2.VideoWriter_fourcc(*'MP44') this is the video decoder...
        video_object=cv2.VideoWriter(('./output/output.mp4'),cv2.VideoWriter_fourcc(*'MP44'),25,(frame.shape[1],frame.shape[0]))
        while ret:
            frame=process_image(frame,face_detection)  #yaha pie blur hua..then hum usse write karenge
            video_object.write(frame)
            ret,frame=cap.read()
    elif args.mode=='webcam':
        cap=cv2.VideoCapture(0)
        while True:
            ret,frame=cap.read()
            frame=process_image(frame,face_detection)
            cv2.imshow("Window",frame)
            cv2.waitKey(25)
            ret,frame=cap.read()
        cap.release()
cv2.destroyAllWindows()



# import cv2
# import mediapipe as mp
# import os
# import argparse

# output_dir = './output'

# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# def process_image(img, face_detection):
#     height, width, _ = img.shape  # Calculate inside the function
#     converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     all_details = face_detection.process(converted_image)
#     if all_details.detections:
#         for detection in all_details.detections:
#             location_data = detection.location_data.relative_bounding_box
#             x, y, w, h = location_data.xmin, location_data.ymin, location_data.width, location_data.height
#             x = int(x * width)  # Convert values
#             y = int(y * height)
#             w = int(w * width)
#             h = int(h * height)
#             img[y:y + h, x:x + w, :] = cv2.blur(img[y:y + h, x:x + w, :], (93, 93)) 
#     return img

# def main(args):
#     mp_face_detection = mp.solutions.face_detection

#     with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#         if args.mode == 'image':
#             img = cv2.imread(args.filePath)
#             img_process = process_image(img, face_detection)
#             cv2.imwrite(os.path.join(output_dir, 'blurred_image.jpg'), img_process)
#         # Additional modes (e.g., video) can be added here if needed

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", default='image', help="Mode to run the script in, either 'image' or 'video'")
#     parser.add_argument("--filePath", default='./images/fam.jpg', help="Path to the image file")
#     args = parser.parse_args()
#     main(args)
#     cv2.destroyAllWindows()





