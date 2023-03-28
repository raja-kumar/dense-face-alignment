import os
import cv2
import mediapipe as mp
import numpy as np

# this function uses mediapipe api to detect face and saves the images after cropping only the bounding box
def mediapipe_face_detection(data_path, output_path):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    #IMAGE_FILES = ["../Dataset/H3DS_sample/sample2/original/0/img_0009.jpg"]
    IMAGE_FILES = []

    files = os.listdir(data_path)

    for file in files:
        curr_image = os.path.join(data_path, file)
        if (file.split('.')[-1] in ['jpg', 'jpeg', 'png']):
            IMAGE_FILES.append(curr_image)

    #print(len(IMAGE_FILES))
    bboxes = {}

    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        for idx, file in enumerate(IMAGE_FILES):
            #print("sancjnajcdd")
            image = cv2.imread(file)
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if not results.detections:
                continue
            annotated_image = image.copy()

            for detection in results.detections:
                """print('Nose tip:')
                print(mp_face_detection.get_key_point(
                    detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))"""
                mp_drawing.draw_detection(annotated_image, detection)
            curr_bbox = results.detections[0].location_data.relative_bounding_box

            if (curr_bbox.xmin < 0 or curr_bbox.ymin < 0):
                continue
            results.detections[0].location_data.relative_bounding_box
            output_image = os.path.join(output_path, os.path.basename(file))
            filename = os.path.basename(file)
            # get the bounding box parameters which is returned by media pipe api
            bboxes[filename] = results.detections[0].location_data.relative_bounding_box
            #print(output_image)
            #print(bboxes[filename])
            xmin = int(bboxes[filename].xmin*image.shape[1]) ## finds the xmin of bounding box
            ymin = int(bboxes[filename].ymin*image.shape[0]) ## fidn the ymin of bounding box
            w,h = int(bboxes[filename].width*image.shape[1]), int(bboxes[filename].height*image.shape[0]) # find the width and height of bounding box
            detected_portion = image[ymin:ymin+h, xmin:xmin+w] # find the detected region in the image
            #print(detected_portion)
            #print(output_image)
            cv2.imwrite(output_image, detected_portion) #save the detected portion
        
        return bboxes