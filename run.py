## it needs to take an image folder, number of landmarks point/custom array as input
##  and need to generate detected face, landmark drawn on face and numpy file

import argparse
import os
from detector import mediapipe_face_detection
from landmarks import find_68_lmks, find_custom_lmks, find_dense_lmks

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="path to the images folder", required=True)
    parser.add_argument("--num_points", type=int, help="number of landmarks. can be 68/191/445", default=68)
    parser.add_argument("--custom_index", type=str, help="use custom list of points")
    parser.add_argument("--output_path", type=str, help="output path to save landmark file and visualization", default="./output")

    args = parser.parse_args()

    input_path = args.data_path
    num_points = args.num_points
    custom_index = args.custom_index
    output_path = args.output_path

    if (not num_points in [68, 191, 445] and custom_index == None):
        raise ValueError("Invalid input. num_points must be among [68, 191, 445] or pass custom_index file")
    
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)

    ## run detector to find the face
    detected_face = os.path.join(output_path, "detected_faces")
    if (not os.path.exists(detected_face)):
        os.makedirs(detected_face)

    bounding_boxes = mediapipe_face_detection(input_path,detected_face)

    ## run the landmark detector
    if (custom_index != None):
        output_path = os.path.join(output_path, "custom")
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        find_custom_lmks(input_path,detected_face, output_path, bounding_boxes,custom_index)
        return
    
    if (num_points == 68):
        output_path = os.path.join(output_path, str(num_points))
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        find_68_lmks(input_path,detected_face, output_path, bounding_boxes)
    else:
        output_path = os.path.join(output_path, str(num_points))
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        find_dense_lmks(input_path,detected_face, output_path, bounding_boxes,num_points)

if __name__ == "__main__":
    main()