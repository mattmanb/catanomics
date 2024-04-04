import torch
import torchvision
from torchvision import transforms
from resnet import BasicBlock, ResNet

import cv2
import numpy as np
from PIL import Image
import os
import random

def slope(x1,y1,x2,y2):
    ###finding slope
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'

def y_intercept(x1,y1,x2,y2):
    # y = mx+b OR b = y-mx
    b = y1 - int(slope(x1, y1, x2, y2) * x1)
    return b

def calc_intersection(m1,b1,m2,b2):
    # Create the coefficient matrices
    a = np.array([[-m1, 1], [-m2, 1]])
    b = np.array([b1, b2])
    try:
        solution = np.linalg.solve(a, b)
    except:
        solution = (0,0)

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
    for line in line_coords:
        try:
            line_equations.append((slope(line[0][0], line[0][1], line[1][0], line[1][1]), 
                                   y_intercept(line[0][0], line[0][1], line[1][0], line[1][1])))
        except:
            pass
    
    # Get perimiter points
    perimeter_points = []
    image_height, image_width, _ = image.shape
    for ind, line_one in enumerate(line_equations):
        for line_two in line_equations[ind+1:]:
            perimeter_point = calc_intersection(line_one[0], line_one[1],
                                                line_two[0], line_two[1])
            if perimeter_point[0] == 0 and perimeter_point[1] == 0:
                pass
            elif perimeter_point[0] >= 0 and perimeter_point[0] <= image_width and perimeter_point[1] >= 0 and perimeter_point[1] <= image_height:
                perimeter_points.append((round(perimeter_point[0]), 
                                          round(perimeter_point[1])))
                
    inverted_edges = 255 - edges
    dist_transform = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)

    good_pts = []

    dist_threshold = 25

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
    DST_PTS = dst_points
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
    images - list of subimages of the numbers of a Catan board

    returns a list of labels in the order the images are passed in
    """
    # Load in the image directories
    file_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]) # Must be sorted for 
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

def show_predictions(subimages, labels):
    for i in range(len(subimages)):
        showImage(subimages[i], str(labels[i]))

def main():
    img_dir = "./images/eval/eval00.jpg"
    save_dir = "./images/eval numbers/"
    homographied_board, dst_points = homography_board(image_dir=img_dir)
    # Show the homographied board
    showImage(homographied_board)
    number_imgs = get_num_images(homographied_board, dst_points, side_length=60)
    save_num_images(homographied_board, dst_points, side_length=60, save_dir=save_dir, ind=0)
    # Predict numbers for each image
    labels = pred_nums_on_resnet(save_dir)
    # Visualize the results
    show_predictions(number_imgs, labels)

if __name__ == "__main__":
    main()