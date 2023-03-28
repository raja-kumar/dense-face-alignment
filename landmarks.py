from functools import partial
from collections import namedtuple
import os
from pytorch_toolbelt.utils import read_rgb_image
from predictor import FaceMeshPredictor
import cv2
import mediapipe as mp
from model_training.utils import load_indices_from_npy
from utils import get_relative_path
import numpy as np

## this functions take an image as input and returns it's 68 landmark coordinates using dadnet network. 
def run_dad_net(image_path):
    image = read_rgb_image(image_path)
    predictor = FaceMeshPredictor.dad_3dnet()
    predictions = predictor(image)
    coordinates = predictions['points']
    x_pred,y_pred = coordinates[:,0], coordinates[:,1]
    return x_pred, y_pred

# this function takes as original image with a face in it and corresponding detected face images (generated using above mediapipe function).
# it runs the DadNet on detected face and find the coordiantes on original image.

def find_68_lmks(original_images, data_path, output_path, bboxes):
    #data_path is the path to the detected faces images (not the full image)
    IMAGE_FILES = []

    files = os.listdir(data_path)

    for file in files:
        curr_image = os.path.join(data_path, file)
        if (file.split('.')[-1] in ['jpg', 'jpeg', 'png']):
            IMAGE_FILES.append(curr_image)

    for idx, file in enumerate(IMAGE_FILES):
        pred_x,pred_y = run_dad_net(file)

        filename = os.path.basename(file)
        original_image = os.path.join(original_images, filename)
        image = cv2.imread(original_image)
        xmin = int(bboxes[filename].xmin*image.shape[1]) # find the xmin of detected face wrt to original image
        ymin = int(bboxes[filename].ymin*image.shape[0]) # find the ymin of the detected face wrt to original image
        pred_x_new, pred_y_new = xmin + pred_x, ymin + pred_y #the coordiantes of landmarks on the original image

        output_file = os.path.join(output_path, filename) 
        lmk_file = os.path.join(output_path,os.path.basename(filename).split(".")[0] + ".npy")
        lmks = []

        for idx,point in enumerate(zip(pred_x_new, pred_y_new)):
            lmks.append(point)
            image = cv2.circle(image, point, radius=6, color=(0, 0, 255), thickness=-1) #save the original image with landmark on it
        
        lmks = np.array(lmks)
        np.save(lmk_file, lmks)
        cv2.imwrite(output_file, image)

def run_dad_net_more_points(image_path, num_points):

    # keypoint dir contain the index of each keypoints that we want to consider for our landmarks. 
    # Only these points are extracted from the projected list of indices

    if (num_points == 445):
        keypoint_dir = "model_training/model/static/face_keypoints/keypoints_445/"
    elif(num_points == 191):
        keypoint_dir = "model_training/model/static/face_keypoints/keypoints_191/"
    else:
        ValueError("Invalid keypoints subset provided.\n"
                   "Available options are: 191, 445")
    
    image = read_rgb_image(image_path)
    predictor = FaceMeshPredictor.dad_3dnet()
    predictions = predictor(image)

    subset_dir = []
    for file in os.listdir(keypoint_dir):
        subset_dir.append(os.path.join(keypoint_dir, file))
    projected_vertices = predictions["projected_vertices"].squeeze().numpy().astype(int) # projected vertices (total of 5023)
    points = []
    for subs_file in subset_dir:
        points.extend(np.take(projected_vertices, load_indices_from_npy(subs_file), axis=0))
    
    pred_x,pred_y = [], []
    for point in points:
        pred_x.append(point[0])
        pred_y.append(point[1])
    
    return np.array(pred_x), np.array(pred_y)

def find_dense_lmks(original_images, data_path, output_path, bboxes, n_points):
    #data_path is the path to the detected faces images (not the full image)
    IMAGE_FILES = []

    files = os.listdir(data_path)

    for file in files:
        curr_image = os.path.join(data_path, file)
        if (file.split('.')[-1] in ['jpg', 'jpeg', 'png']):
            IMAGE_FILES.append(curr_image)
        #IMAGE_FILES.append(curr_image)

    for idx, file in enumerate(IMAGE_FILES):
        pred_x,pred_y = run_dad_net_more_points(file,n_points)

        filename = os.path.basename(file)
        original_image = os.path.join(original_images, filename)
        image = cv2.imread(original_image)
        xmin = int(bboxes[filename].xmin*image.shape[1])
        ymin = int(bboxes[filename].ymin*image.shape[0])
        pred_x_new, pred_y_new = xmin + pred_x, ymin + pred_y

        output_file = os.path.join(output_path, filename)
        lmk_file = os.path.join(output_path,os.path.basename(filename).split(".")[0] + ".npy")
        lmks = []

        for idx,point in enumerate(zip(pred_x_new, pred_y_new)):
            lmks.append(point)
            image = cv2.circle(image, point, radius=6, color=(0, 0, 255), thickness=-1)

        lmks = np.array(lmks)
        np.save(lmk_file, lmks)
        cv2.imwrite(output_file, image)

def run_dad_net_custom_points(image_path, indices_file):

    # pass a numpy indices to get those points 
    
    image = read_rgb_image(image_path)
    predictor = FaceMeshPredictor.dad_3dnet()
    predictions = predictor(image)
    projected_vertices = predictions["projected_vertices"].squeeze().numpy().astype(int) # projected vertices (total of 5023)
    indices = np.load(indices_file)
    points = []
    points.extend(np.take(projected_vertices, indices, axis=0))
    
    pred_x,pred_y = [], []
    for point in points:
        pred_x.append(point[0])
        pred_y.append(point[1])
    
    return np.array(pred_x), np.array(pred_y)

def find_custom_lmks(original_images, data_path, output_path, bboxes, indices_file):
    #data_path is the path to the detected faces images (not the full image)
    IMAGE_FILES = []

    files = os.listdir(data_path)

    for file in files:
        curr_image = os.path.join(data_path, file)
        if (file.split('.')[-1] in ['jpg', 'jpeg', 'png']):
            IMAGE_FILES.append(curr_image)
        #IMAGE_FILES.append(curr_image)

    for idx, file in enumerate(IMAGE_FILES):
        pred_x,pred_y = run_dad_net_custom_points(file,indices_file)

        filename = os.path.basename(file)
        original_image = os.path.join(original_images, filename)
        image = cv2.imread(original_image)
        xmin = int(bboxes[filename].xmin*image.shape[1])
        ymin = int(bboxes[filename].ymin*image.shape[0])
        pred_x_new, pred_y_new = xmin + pred_x, ymin + pred_y

        output_file = os.path.join(output_path, filename)
        lmk_file = os.path.join(output_path,os.path.basename(filename).split(".")[0] + ".npy")
        lmks = []
        for idx,point in enumerate(zip(pred_x_new, pred_y_new)):
            lmks.append(point)
            image = cv2.circle(image, point, radius=3, color=(0, 0, 255), thickness=-1)
        
        lmks = np.array(lmks)
        np.save(lmk_file, lmks)
        cv2.imwrite(output_file, image)