import torch
import torchvision
from torchvision import transforms
from resnet import BasicBlock, ResNet

import cv2
import numpy as np
from PIL import Image
import os
import random

from predict_hexes import save_hex_images, predict_hexes_on_resnet

def slope(x1,y1,x2,y2):
    ###finding slope
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return None

def y_intercept(x1,y1,x2,y2):
    # y = mx+b OR b = y-mx
    m = slope(x1, y1, x2, y2) * x1
    if m:
        b = y1 - int(m)
    elif m == -0.0:
        b = y1
    else:
        b = None
    return b

def calc_intersection(m1,b1,m2,b2):
    """
    Returns the intersection points of two lines
    Special Cases:
        * One line is verticle (a solution is still returned)
        * Both lines are verticle (no solution)
        * The lines are parralel (no solution)
    If there is no solution, an empty tuple is returned
    """
    # Create the coefficient matrices
    if m1 == m2:
        # both lines are verticle or parallel, no solution 
        solution = tuple()
    if isinstance(m1, str):
        # The first line is verticle, solve using b1
        x = b1
        y = m2 * x + b2 # line 1's equation is x = b1
        solution = (x, y)
    elif isinstance(m2, str):
        # The second line is verticle, solve using b2
        x = b2
        y = m1 * x + b1 # line 2's equation is x = b2
        solution = (x, y)
    else:
        try:
            # set up solving matrices
            a = np.array([[-m1, 1], [-m2, 1]], dtype=np.float64)
            b = np.array([b1, b2], dtype=np.float64)
            solution = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            return tuple()

    return solution

def showImage(img, name=None):
    if not name:
        cv2.imshow("Image display", img)
    else:
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def saveImage(filename, img, dir):
    # Get full path
    full_path = f"{dir}/{filename}"
    cv2.imwrite(full_path, img)
    print(f"Image saved to {full_path}")

def homography_board(image, vis=False):
    if isinstance(image, str):
        image = cv2.imread(image)
    if vis:
        showImage(image, "Before")
    ## Get border image
    # Resize to 1052x1052
    image = cv2.resize(image, (1052, 1052))
    # Convert to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the color range of the border
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    # Get the mask of just the blue regions
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # Get the bitwise_and of the blue region (1 if blue, 0 if not)
    blue_region = cv2.bitwise_and(image, image, mask=mask)
    # Format image so guassian blur and canny edge can be performed
    gray_blue_region = cv2.cvtColor(blue_region, cv2.COLOR_BGR2GRAY)
    # Apply gaussian blur for edge detection
    blurred_blue_region = cv2.GaussianBlur(gray_blue_region, (5,5), 0)
    # Canny Edge Detection
    edges = cv2.Canny(blurred_blue_region, 50, 150)
    if vis:
        showImage(edges, "EDGES")

    ## Hough Transform for perimiter lines
    lines = []
    thresh = 200
    while len(lines) < 10:
        lines=cv2.HoughLines(edges, 1, np.pi / 180, thresh)
        if len(lines) >= 10:
            break
        else:
            thresh -= 5

    # Find overlapping lines from hough transform
    min_rho_diff = 10 # the minimum distance between two lines to determine if they are for the same side
    min_theta_diff = 0.05
    overlapping_indexes = set()

    while len(lines) - len(overlapping_indexes) > 6:
        for i in range(len(lines)):
            rho, theta = lines[i, 0]
            for j in range(i+1, len(lines)):
                rho2, theta2 = lines[j, 0]
                if abs(rho-rho2) < min_rho_diff and abs(theta-theta2) < min_theta_diff:
                    overlapping_indexes.add(j)
        min_rho_diff += 1
    perimiter_lines = [] # these are the actual perimiter lines after overlap is removed
    # Get the perimiter lines
    for i in range(len(lines)):
        if i not in overlapping_indexes:
            perimiter_lines.append(lines[i])

    ## Finding the perimiter POINTS from the perimiter lines
    line_coords = []
    for index, line in enumerate(perimiter_lines):
    # for index, line in enumerate(lines):
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        line_coords.append([(x1, y1), (x2, y2)])
        if vis:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    if vis:
        showImage(image)
    # Get the line equations
    line_equations = []
    # NEED TO FIX VERTICLE LINES
    for line in line_coords:
        try:
            m = slope(line[0][0], line[0][1], line[1][0], line[1][1])
            if m:
                b = y_intercept(line[0][0], line[0][1], line[1][0], line[1][1])
                line_equations.append((m, b))
            else:
                # This is if the line is verticle
                line_equations.append(('verticle line', line[0][0]))
        except Exception as e:
            print("at line 160", e)
    
    # Get perimiter points
    perimeter_points = []
    image_height, image_width, _ = image.shape
    for ind, line_one in enumerate(line_equations):
        for line_two in line_equations[ind+1:]:
            perimeter_point = calc_intersection(line_one[0], line_one[1],
                                                line_two[0], line_two[1])
            if len(perimeter_point) != 2:
                # calc_intersection returns an empty tuple if there is no intersection point
                print("No intersection point, moving on...")
                pass
            elif perimeter_point[0] >= 0 and perimeter_point[0] <= image_width and perimeter_point[1] >= 0 and perimeter_point[1] <= image_height:
                perimeter_points.append((round(perimeter_point[0]), 
                                          round(perimeter_point[1])))
                
    inverted_edges = 255 - edges
    dist_transform = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)

    good_pts = []

    dist_threshold = 25

    # This is an infinite loop
    while len(good_pts) < 6:
        good_pts = []
        for i in range(len(perimeter_points)):
            x, y = perimeter_points[i]
            dist_to_edge = dist_transform[y, x]
            if dist_to_edge < dist_threshold:
                if len(good_pts) < 6:
                    good_pts.append(perimeter_points[i])
        dist_threshold += 5
    
    # Order the points
    centroid = np.mean(good_pts, axis=0)
    sorted_points = sorted(good_pts, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
    
    # Visualize the points if wanted
    if vis:
        for pt in good_pts:
            cv2.circle(image, pt, 5, (255, 255, 0), -1)
        showImage(image, "Perimiter points")

    # Set up points for homography
    src_points = np.array(sorted_points)
    R = 526
    center = np.array([R, R])

    theta = np.pi / 2
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    hexagon_points = np.array([(center[0] + R * np.cos(2 * np.pi / 6 * i), center[1] + R * np.sin(2 * np.pi / 6 * i)) for i in range(6)])

    dst_points = []

    for point in hexagon_points:
        translated_point = point - center
        rotated_point = np.dot(rotation_matrix, translated_point)
        dst_points.append([int(rotated_point[0]+R), int(rotated_point[1]+R)])
    dst_points = np.array(dst_points)
    ## HOMOGRAPHY
    # Compute homography matrix
    H, _ = cv2.findHomography(src_points, dst_points)

    # Apply homography to warp the real test_image to the ideal Catan board's perspective
    warped_image = cv2.warpPerspective(image, H, (1052, 1052))
    if vis:
        showImage(warped_image, "After")

    return warped_image, dst_points

def create_bb(image, center_point, sl):
    """
    image: the image to draw the bounding box on
    center_point: the center point of the bounding box
    sl: side length of the square bounding box
    color: scalar color of the bounding box to be
    """
    tl_corner = (center_point[0] - int(sl/2), center_point[1] - int(sl/2)) # top left corner
    tr_corner = (center_point[0] + int(sl/2), center_point[1] - int(sl/2)) # top right corner
    bl_corner = (center_point[0] - int(sl/2), center_point[1] + int(sl/2)) # bottom left corner
    br_corner = (center_point[0] + int(sl/2), center_point[1] + int(sl/2)) # bottom right corner

    # draw the bounding box
    # cv2.line(image, tl_corner, tr_corner, color, 3)
    # cv2.line(image, tr_corner, br_corner, color, 3)
    # cv2.line(image, br_corner, bl_corner, color, 3)
    # cv2.line(image, bl_corner, tl_corner, color, 3)

    return tl_corner[0], tl_corner[1], br_corner[0], br_corner[1]

def save_num_images(image, dst_points, side_length, save_dir, ind):
    """Finds and saves all the numbers on the Catan board of a homogrophied image
    image: homographied image
    dst_points: perimiter points of the board
    sl: side length of bounding box for each number
    save_dir: directory for images to be saved to
    ind: how many images have already been saved to the directory
    """
    subimages = []
    R = 526
    # This is the actual center point
    x1, y1, x2, y2 = create_bb(image,
                               (R, R),
                               side_length)
    subimage_center = image[y1:y2, x1:x2]
    subimages.append(subimage_center)
    # This gets all the numbers that fall on the line between the center point of the board and the perimiter points
    for pt in dst_points:
        delta_x = R - pt[0]
        delta_y = R - pt[1]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (7/24)*delta_x)), 
                                    int((R + (7/24)*delta_y))), 
                                   side_length)
        subimage1 = image[y1:y2, x1:x2]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (3/5)*delta_x)), 
                                    int((R + (3/5)*delta_y))), 
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

        x1, y1, x2, y2 = create_bb(image, (shifted_x, shifted_y), sl=side_length)
        subimage = image[y1:y2, x1:x2]
        subimages.append(subimage)

    # Save the images
    for img in subimages:
        if ind < 10:
            saveImage(f"00{ind}.jpg", img, save_dir)
        elif ind < 100:
            saveImage(f"0{ind}.jpg", img, save_dir)
        else:
            saveImage(f"{ind}.jpg", img, save_dir)
        ind += 1
    
    return ind

def get_num_images(image, dst_points, side_length):
    """Finds and saves all the numbers on the Catan board of a homogrophied image
    image: source image
    dst_points: perimiter points of the board
    sl: side length of bounding box for each number
    save_dir: directory for images to be saved to
    ind: how many images have already been saved to the directory
    """
    subimages = []
    R = 526
    # This is the actual center point
    x1, y1, x2, y2 = create_bb(image,
                               (R, R),
                               side_length)
    subimage_center = image[y1:y2, x1:x2]
    subimages.append(subimage_center)
    # This gets all the numbers that fall on the line between the center point of the board and the perimiter points
    for pt in dst_points:
        delta_x = R - pt[0]
        delta_y = R - pt[1]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (7/24)*delta_x)), 
                                    int((R + (7/24)*delta_y))), 
                                   side_length)
        subimage1 = image[y1:y2, x1:x2]

        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (3/5)*delta_x)), 
                                    int((R + (3/5)*delta_y))), 
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

        x1, y1, x2, y2 = create_bb(image, (shifted_x, shifted_y), sl=side_length)
        subimage = image[y1:y2, x1:x2]
        subimages.append(subimage)

    return subimages

def pred_nums_on_resnet(img_dir) -> list:
    """
    img_dir: directory of images to be predicted upon

    returns a list of labels in alphabetical order (of the image names)
    """
    # Load in the image directories
    file_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]) # Must be sorted for 
    # Set up and load the model
    CLASS_NAMES = ['desert','eight','eleven','five','four','nine','six','ten','three','twelve','two']
    CLASS_CNT = len(CLASS_NAMES) # eleven classes to be predicted
    MODEL_SAVE_PATH = "./CATANIST/models/catanistv2_3.pth"
    LABELS = []
    PRED_PROBS = []

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
        transforms.Resize(size=(64, 64)),
        transforms.Grayscale(num_output_channels=3),
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
        PRED_PROBS.append(img_pred_probs.cpu())
        LABELS.append(img_label)
    
    # Ensure that the correct number of classes is predicted
    LABELS = validate_num_predictions(LABELS, torch.stack(PRED_PROBS), class_names=CLASS_NAMES)

    return LABELS

def pred_num_on_resnet(img_path):
    # Set up and load the model
    CLASS_NAMES = ['desert','eight','eleven','five','four','nine','six','ten','three','twelve','two']
    CLASS_CNT = len(CLASS_NAMES) # eleven classes to be predicted
    MODEL_SAVE_PATH = "./CATANIST/models/catanistv2_1.pth"
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
        transforms.Resize(size=(64, 64)),
        transforms.Grayscale(num_output_channels=3),
    ])
    # Put the model into eval mode
    resnet_model.eval()
    image = torchvision.io.read_image(str(img_path)).type(torch.float32)
    image /= 255
    transformed_img = input_transform(image[:3, :, :])
    with torch.inference_mode():
        img_pred = resnet_model((transformed_img.unsqueeze(0)).to(device))
    # Logits -> Predictions probabilites -> Prediction labels -> class name
    img_pred_probs = torch.softmax(img_pred, dim=1)
    img_pred_label = torch.argmax(img_pred_probs, dim=1)
    img_label = CLASS_NAMES[img_pred_label]
    return img_label

def validate_num_predictions(labels, pred_probs, class_names):
    valid_board = { 'two': 1,
                    'three': 2,
                    'four': 2,
                    'five': 2,
                    'six': 2, 
                    'eight': 2,
                    'nine': 2,
                    'ten': 2,
                    'eleven': 2,
                    'twelve': 1,
                    'desert': 1 }
    predicted_board = { 'two': 0,
                        'three': 0,
                        'four': 0,
                        'five': 0,
                        'six': 0, 
                        'eight': 0,
                        'nine': 0,
                        'ten': 0,
                        'eleven': 0,
                        'twelve': 0,
                        'desert': 0 }
    for label in labels:
        predicted_board[label] += 1
    
    # Compare the dictionaries of the valid board and the predicted board
    flag = 0
    for i in valid_board:
        if valid_board.get(i) != predicted_board.get(i):
            flag = 1
            break
    if flag == 0:
        return labels
    else:
        labels = adjust_labels(labels, pred_probs, class_names, valid_board, predicted_board)
    return labels

def adjust_labels(labels, pred_probs, class_names, valid_board, predicted_board):
    """This function is meant to fix number labels if invalid board predictions come through
    labels- current prediction labels (to be adjusted)
    pred_probs- the confidence values of each class for each label (this'll be used to adjust the labels)
    valid_board- dictionary of class names and how many of that class SHOULD BE predicted
    predicted_board- dictionary of class names and how many of that class WAS predicted
    """
    # Find where there is a discrepancy in the predicted numbers
    invalid_class = 'N/A'
    for pred_class, num_preds in predicted_board.items():
        # ind is the index within the dictionary that we're looking at
        # pred_class is the string class 
        # num_preds is the number of predictions for this class
        if num_preds > valid_board[pred_class]: # if a class has been predicted too many times
            invalid_class = pred_class
    if invalid_class == 'N/A':
        return labels
    # Compare the confidence values of each prediction for that class
    # Reduce the number of predictions in this class by 1 in the predicted_board dict
    # print(f"Number of {invalid_class} predictions: {predicted_board[invalid_class]}")
    invalid_indexes = [i for i, name in enumerate(labels) if name==invalid_class]
    # print(f"Possible invalid indexes: {invalid_indexes}")
    lowest_confidence = 1.0
    lowest_confidence_index = -1
    for i in invalid_indexes:
        confidence_of_prediction = pred_probs[i].max(dim=1)[0]
        if confidence_of_prediction < lowest_confidence:
            lowest_confidence = confidence_of_prediction
            lowest_confidence_index = i
    # print(f"Lowest confidence index for this class: {lowest_confidence_index}")
    # Take the lowest confidence and move the label to the next most confident
    index_to_modify = pred_probs[lowest_confidence_index].argmax(dim=1)[0]
    pred_probs[lowest_confidence_index][0][index_to_modify] = 0.0
    # print(f"Prediction probabilities of the invalid index:\n{pred_probs[lowest_confidence_index]}")
    new_class_pred = pred_probs[lowest_confidence_index].argmax(dim=1)
    new_class_pred_label = class_names[new_class_pred]
    # print(f"New prediction label: {new_class_pred_label}")
    labels[lowest_confidence_index] = new_class_pred_label
    
    # Recursive call validate_num_prediction to check if the new labels are now correct
    labels = validate_num_predictions(labels, pred_probs, class_names)

    # print(f"Correct labels: {labels}")

    # Return valid labels
    return labels

def show_predictions(subimages, labels):#
    for i in range(len(subimages)):
        showImage(subimages[i], str(labels[i]))

def main():
    # This is all to create more training images right now
    img_dir = "./images/v6/board07.jpeg"
    num_save_dir = "./images/eval numbers/"
    hex_save_dir = "./images/eval hexes"
    # image_dirs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    # ind = 190
    NUM_SIDE_LENGTH = 60
    HEX_SIDE_LENGTH = 40
    # print(image_dirs)
    image = cv2.imread(img_dir)
    hom_img, perimeter_pts = homography_board(image)
    save_num_images(hom_img, perimeter_pts, NUM_SIDE_LENGTH, num_save_dir, 0)
    save_hex_images(hom_img, perimeter_pts, HEX_SIDE_LENGTH)
    num_labels = pred_nums_on_resnet(num_save_dir)
    hex_labels = predict_hexes_on_resnet(hex_save_dir)

    # for image_dir in image_dirs:
    #     image = cv2.imread(image_dir)
    #     hom_img, perimeter_points = homography_board(image)
    #     ind = save_num_images(hom_img, perimeter_points, SIDE_LENGTH, save_dir, ind)
    # print("Done saving images")

if __name__ == "__main__":
    main()