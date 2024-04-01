import torch
import torchvision
from torchvision import transforms
from resnet import BasicBlock, ResNet

import cv2
import numpy as np
from PIL import Image
import os
import random
from predict_numbers import create_bb, slope, y_intercept, calc_intersection, homography_board, showImage, saveImage, show_predictions

def save_hex_images(image, save_dir, dst_points, side_length, num_offset, ind):
    """ Get all the images for a NN to identify hex types
    dst_points: perimiter points of the board
    side_length: side length of bounding box for each number
    num_offset: the number of pixels above the number point
    """
    subimages = []
    R = 526
    # This is the actual center point
    x1, y1, x2, y2 = create_bb(image,
                               (R, R - num_offset),
                               side_length)
    print(type(x1), type(y1), type(x2), type(y2))
    subimage_center = image[y1:y2, x1:x2]
    subimages.append(subimage_center)
    # This gets all the numbers that fall on the line between the center point of the board and the perimiter points
    for pt in dst_points:
        delta_x = R - pt[0]
        delta_y = R - pt[1]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (7/24)*delta_x)), 
                                    int((R + (7/24)*delta_y) - num_offset)), 
                                   side_length)
        subimage1 = image[y1:y2, x1:x2]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (3/5)*delta_x)), 
                                    int((R + (3/5)*delta_y) - num_offset)), 
                                    side_length)
        subimage2 = image[y1:y2, x1:x2]

        subimages.append(subimage1)
        subimages.append(subimage2)

    # This gets all the numbers not found by the center lines to the perimiter points using every other perimiter point to find the numbers
    for i in range(len(dst_points)):
        # Draw lines between every other corner
        next_point_ind = i+2
        if i+2 >= len(dst_points):
            next_point_ind = i+2-len(dst_points)

        pt1 = [dst_points[i][0], dst_points[i][1]]
        pt2 = [dst_points[next_point_ind][0], dst_points[next_point_ind][1]]

        # Draw circles at varying distances
        delta_x = pt2[0] - pt1[0]
        delta_y = pt2[1] - pt1[1]

        ## Tests
        cir_pt = (int((pt1[0] + (1/3)*delta_x)), int((pt1[1] + (1/3)*delta_y)))

        ### Shift the circle toward the center a bit
        xfc = R - cir_pt[0] # x distance from center
        yfc = R - cir_pt[1] # y distance from center

        shift_factor = 0.10 # Shifts the center point closer to the center by a factor if this much

        shifted_x = int(cir_pt[0] + shift_factor * xfc)
        shifted_y = int(cir_pt[1] + shift_factor * yfc)

        x1, y1, x2, y2 = create_bb(image, (shifted_x, shifted_y - num_offset), sl=side_length)
        subimage = image[y1:y2, x1:x2]
        subimages.append(subimage)

    for img in subimages:
        if ind < 10:
            saveImage(f"00{ind}.jpg", img, save_dir)
        elif ind < 100:
            saveImage(f"0{ind}.jpg", img, save_dir)
        else:
            saveImage(f"{ind}.jpg", img, save_dir)
        ind += 1

    return ind

def get_hex_images(image, dst_points, side_length, num_offset):
    """ Get all the images for a NN to identify hex types
    dst_points: perimiter points of the board
    side_length: side length of bounding box for each number
    num_offset: the number of pixels above the number point
    """
    subimages = []
    R = 526
    # This is the actual center point
    x1, y1, x2, y2 = create_bb(image,
                               (R, R - num_offset),
                               side_length)
    print(type(x1), type(y1), type(x2), type(y2))
    subimage_center = image[y1:y2, x1:x2]
    subimages.append(subimage_center)
    # This gets all the numbers that fall on the line between the center point of the board and the perimiter points
    for pt in dst_points:
        delta_x = R - pt[0]
        delta_y = R - pt[1]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (7/24)*delta_x)), 
                                    int((R + (7/24)*delta_y) - num_offset)), 
                                   side_length)
        subimage1 = image[y1:y2, x1:x2]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (3/5)*delta_x)), 
                                    int((R + (3/5)*delta_y) - num_offset)), 
                                    side_length)
        subimage2 = image[y1:y2, x1:x2]

        subimages.append(subimage1)
        subimages.append(subimage2)

    # This gets all the numbers not found by the center lines to the perimiter points using every other perimiter point to find the numbers
    for i in range(len(dst_points)):
        # Draw lines between every other corner
        next_point_ind = i+2
        if i+2 >= len(dst_points):
            next_point_ind = i+2-len(dst_points)

        pt1 = [dst_points[i][0], dst_points[i][1]]
        pt2 = [dst_points[next_point_ind][0], dst_points[next_point_ind][1]]

        # Draw circles at varying distances
        delta_x = pt2[0] - pt1[0]
        delta_y = pt2[1] - pt1[1]

        ## Tests
        cir_pt = (int((pt1[0] + (1/3)*delta_x)), int((pt1[1] + (1/3)*delta_y)))

        ### Shift the circle toward the center a bit
        xfc = R - cir_pt[0] # x distance from center
        yfc = R - cir_pt[1] # y distance from center

        shift_factor = 0.10 # Shifts the center point closer to the center by a factor if this much

        shifted_x = int(cir_pt[0] + shift_factor * xfc)
        shifted_y = int(cir_pt[1] + shift_factor * yfc)

        x1, y1, x2, y2 = create_bb(image, (shifted_x, shifted_y - num_offset), sl=side_length)
        subimage = image[y1:y2, x1:x2]
        subimages.append(subimage)

    return subimages

def predict_hexes_on_resnet(img_dir):
    """
    images - list of subimages of the numbers of a Catan board

    returns a list of labels in the order the images are passed in
    """
    # Load in the image directories
    file_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]) # Must be sorted for 
    # Set up and load the model
    CLASS_NAMES = ['brick', 'desert', 'ore', 'sheep', 'wheat', 'wood']
    CLASS_CNT = len(CLASS_NAMES) # eleven classes to be predicted
    MODEL_SAVE_PATH = "./HEXIST/models/hexistV1.pth"
    LABELS = []

    # Device agnostic code
    device = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
            )

    resnet_model = ResNet(input_shape=3, 
                        block=BasicBlock,
                        layers=[2, 2, 2],
                        class_cnt=CLASS_CNT).to(device)

    resnet_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    resnet_model.to(device)

    # Define the image transform
    input_transform = transforms.Compose([
        transforms.Resize(size=(40, 40))
    ])
    # Put the model into eval mode
    resnet_model.eval()
    for file in file_paths:
        image = torchvision.io.read_image(str(file)).type(torch.float32)
        image /= 255
        transformed_img = input_transform(image[:3, :, :])
        with torch.inference_mode():
            img_pred = resnet_model((transformed_img.unsqueeze(0)).to(device))
        # Logits -> Predictions probabilites -> Prediction labels -> class name
        img_pred_probs = torch.softmax(img_pred, dim=1)
        img_pred_label = torch.argmax(img_pred_probs, dim=1)
        img_label = CLASS_NAMES[img_pred_label]
        LABELS.append(img_label)

    return LABELS


def main():
    eval_img_path = "./images/eval/eval00.jpg"
    og_img = cv2.imread(eval_img_path)
    showImage(og_img)
    eval_save_path = "./images/eval hexes"
    hom_img, dst_points = homography_board(eval_img_path, vis=True)
    showImage(hom_img)
    SIDE_LENGTH = 40
    PX_OFFSET = 55
    hex_imgs = get_hex_images(hom_img, dst_points, SIDE_LENGTH, PX_OFFSET)
    labels = predict_hexes_on_resnet(eval_save_path)
    show_predictions(hex_imgs, labels)
    

if __name__ == "__main__":
    main()