"""
This file is the main hunk of code that performs a lot of the logic requires to pre-process the image and predict the numbers/hexes using a trained model that is saved in the project ("./CATANIST/models" and "./HEXIST/models")
"""

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
    # if x2 = x1 then the slope is infinity
    if x2!=x1:
        # change in y / change in x
        return((y2-y1)/(x2-x1))
    else:
        # The slope DNE (divide by 0 error)
        return None

def y_intercept(x1,y1,x2,y2):
    # y = mx+b OR b = y-mx
    m = slope(x1, y1, x2, y2) * x1
    # if the slope exists (`slope` function returns None if it doesn't)
    if m:
        # the y-intercept is y - mx
        b = y1 - int(m)
    # This is a weird result I was getting from slope, which basically means the line is horizontal, so this catches that result from the `slope` function
    elif m == -0.0:
        b = y1
    else:
        # if the slope doesn't exist, return None
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
            # use numpy to solve the matrices
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

def homography_board(image, vis=False, hom=False):
    """Process an image of a Catan board into a top-down version

    Keyword arguments:
    image -- cv2 image or image file path
    vis -- boolean to visualize the processes within homographying the image for error checking
    hom -- boolean to visualize the end product ONLY
    """
    # if the image passed in is a string, assume it is a file path; if it isn't, then we assume that image is already read in using cv2
    if isinstance(image, str):
        image = cv2.imread(image)
    if vis:
        showImage(image, "Before")
    ## Get border image
    # Resize to 1052x1052
    image = cv2.resize(image, (1052, 1052))
    # Convert to HSV
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the color range of the border (blue ocean border)
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    # Get the mask of just the blue regions
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    if vis:
        showImage(mask)
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
    # This threshold value will be reduced until 10 lines are found
    thresh = 200
    # Go until 10 lines are found (only 6 lines sometimes doesn't find all 6 perimeter edges because some overlap)
    while len(lines) < 10:
        # Call houghlines with the current threshold
        lines=cv2.HoughLines(edges, 1, np.pi / 180, thresh)
        # If we have 10 or more lines, leave the while loop
        if len(lines) >= 10:
            break
        # If 10 lines are NOT found, lower the threshold then go to the next iteration of the loop
        else:
            thresh -= 5

    # Find overlapping lines from hough transform
    min_rho_diff = 10 # the minimum distance between two lines to determine if they are for the same line
    min_theta_diff = 0.05 # the minimum angle between two lines to determine if they are for the same line
    # this set will hold the indexes of all `overlapping` lines
    overlapping_indexes = set()

    # Until we have 6 lines remaining, continue looking through lines for overlapping lines
    while len(lines) - len(overlapping_indexes) > 6:
        # Iterate through lines starting at the first line, then comparing side by side with the remaining lines to see if they overlap (using min rho and min theta)
        for i in range(len(lines)):
            # get the rho and theta values of the first line
            rho, theta = lines[i, 0]
            # iterate through remaining lines to compare to the line we are currently looking at
            for j in range(i+1, len(lines)):
                # Get the second line's rho and theta values
                rho2, theta2 = lines[j, 0]
                # check to see if the lines are within the min rho and min theta differences
                if abs(rho-rho2) < min_rho_diff and abs(theta-theta2) < min_theta_diff:
                    # if the lines are overlapping, add this line's index to `overlapping_indexes`
                    overlapping_indexes.add(j)
        # Add to the min rho and min theta difference in case we haven't narrowed it down to 6 lines
        min_rho_diff += 1
        min_theta_diff += 0.01
    perimiter_lines = [] # these are the actual perimiter lines after overlap is removed
    # Get the perimiter lines by not adding lines that are considered 'overlapping' (if the line's index is in `overlapping_indexes` it is considered 'overlapping')
    for i in range(len(lines)):
        if i not in overlapping_indexes:
            perimiter_lines.append(lines[i])

    ## Finding the perimiter POINTS from the perimiter lines
    line_coords = [] # This list will hold the xy starting points and ending points of each remaining line (this will form a lines 2000px long)
    # 'For' loop to iterate over each line remaining, pulling each line's polar data and converting it into xy start and end coordinates
    for index, line in enumerate(perimiter_lines):
        # Pull the polar coordinates out of this line
        rho, theta = line[0]
        # Calculate xy coordinates of a line following the polar data for the line
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # Add/subtract 1000 to extend the line for a more accurate slope calculation
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # Append the line's coordinates to `line_coords`
        line_coords.append([(x1, y1), (x2, y2)])
        if vis:
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
    if vis:
        showImage(image)
    # Get the line equations
    line_equations = [] # This list will store the slope and y-intercept of each line
    # Iterate through each line in `line_coords`
    for line in line_coords:
        # Was having trouble with this code block, try-catch is to catch if a line is verticle, which results in a divide by 0 error
        # Instead of throwing an error, I assume the line is verticle so I set the slope as 'verticle line' and the y-intercept as the x-value
        try:
            # Call 'slope' function and send in the start and end xy coords
            m = slope(line[0][0], line[0][1], line[1][0], line[1][1])
            # if the slope DNE (is None) then the line is verticle, in which case we catch it here
            if m:
                # calculate the y-intercept given the coordinates 
                b = y_intercept(line[0][0], line[0][1], line[1][0], line[1][1])
                # append the line's equation to `line_equations`
                line_equations.append((m, b))
            else:
                # This is if the line is verticle
                line_equations.append(('verticle line', line[0][0]))
        except Exception as e:
            # If this except block happens, then something is actually going wrong (not verticle line related)
            print("at line 160", e)
    
    # Get perimiter points
    perimeter_points = [] # list that will contain the actual xy coordinates of the board's perimeter
    # get the shape of the image
    image_height, image_width, _ = image.shape
    # iterate over all the lines, calculating intersections between the two to find perimeter points (a lot of the intersections won't matter, and that is handled in the for loop)
    for ind, line_one in enumerate(line_equations):
        # We have the first line, not iterate over the rest of the lines and calculate the intersections
        for line_two in line_equations[ind+1:]:
            # use `calc_intersection` function to find the intersection point between two lines
            perimeter_point = calc_intersection(line_one[0], line_one[1],
                                                line_two[0], line_two[1])
            # if `calc_intersection` doesn't return a tuple of two values (xy coords) then the lines are parallel
            if len(perimeter_point) != 2:
                # calc_intersection returns an empty tuple if there is no intersection point
                print("No intersection point, moving on...")
                pass
            # Check to make sure the perimeter point is inside the bounds of the image (if it isn't, then the intersection is meaningless)
            elif perimeter_point[0] >= 0 and perimeter_point[0] <= image_width and perimeter_point[1] >= 0 and perimeter_point[1] <= image_height:
                # append the perimeter point to `perimeter_points`
                perimeter_points.append((round(perimeter_point[0]), 
                                          round(perimeter_point[1])))
    
    # binary images of canny edge detection to get all the meaningful intersections 
    inverted_edges = 255 - edges
    # this transform calculates the distance from a passed in point to any turned on pixel of the canny edge detection
    dist_transform = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)

    # This list will hold all the ACTUAL perimeter points
    good_pts = []

    # This is the threshold of how close a point must be to a turned on pixel of the canny edge image
    dist_threshold = 25

    # Until 6 perimeter points are found, continuously increase the distance_threshold to find the closest intersection points to the Canny edge border
    while len(good_pts) < 6:
        # Empty `good_pts`
        good_pts = []
        # iterate over each perimeter point
        for i in range(len(perimeter_points)):
            # get the x and y values of the point
            x, y = perimeter_points[i]
            # calculate the distance to the closest turned on pixel using dist_transform
            dist_to_edge = dist_transform[y, x]
            # if the distance to the edge is less than the distance threshold, we found a valid point
            if dist_to_edge < dist_threshold:
                # Ensure no more than 6 points are added to the list
                if len(good_pts) < 6:
                    good_pts.append(perimeter_points[i])
        # Add 5 to the distance threshold for the next iteration of the loop in case 6 points weren't found
        dist_threshold += 5
    
    # Order the points
    centroid = np.mean(good_pts, axis=0) # this is the center point of the image
    # Using angles from the centroid, we can order the 6 points in a clockwise fashion
    sorted_points = sorted(good_pts, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
    
    # Visualize the points if wanted
    if vis:
        for pt in good_pts:
            cv2.circle(image, pt, 20, (0, 0, 255), -1)
        showImage(image, "Perimiter points")

    # Set up points for homography
    src_points = np.array(sorted_points) # this is 'source' points, or the actual perimeter points in the uploaded image
    # R is half of the width/height of the image
    R = 526
    # The center point is 526, 526 (half the width/height)
    center = np.array([R, R])

    # This is all to calculate the 'optimal' perimeter point locations, or the points that `src_points` will be mapped to (we need to rotate 90 degrees to get it in the format we want)
    theta = np.pi / 2
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
    hexagon_points = np.array([(center[0] + R * np.cos(2 * np.pi / 6 * i), center[1] + R * np.sin(2 * np.pi / 6 * i)) for i in range(6)])

    # This list will hold all the destination points, or the points where we WANT the perimeter points to be
    dst_points = []

    # Iterate over hexagon_points (the dst_points) and rotate the points into the orientation we want
    for point in hexagon_points:
        translated_point = point - center
        # rotate the point around the center 90 degrees
        rotated_point = np.dot(rotation_matrix, translated_point)
        # Append the point to `dst_points` for further use
        dst_points.append([int(rotated_point[0]+R), int(rotated_point[1]+R)])
    # Change dst_points into a numpy array to be sent into findHomography function
    dst_points = np.array(dst_points)
    ## HOMOGRAPHY
    # Compute homography matrix
    H, _ = cv2.findHomography(src_points, dst_points)

    # Apply homography to warp the real test_image to the ideal Catan board's perspective
    warped_image = cv2.warpPerspective(image, H, (1052, 1052))
    if vis or hom:
        showImage(warped_image, "After")

    # Return the warped image, as well as the designated point locations which will be used to get all the number and hex images from the warped image
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

def save_num_images(image, dst_points, side_length, save_dir, imgNum):
    """Finds and saves all the numbers on the Catan board of a homogrophied image
    image: homographied image (cv2 read in already)
    dst_points: perimiter points of the board
    sl: side length of bounding box for each number image
    save_dir: directory for images to be saved to
    ind: how many images have already been saved to the directory (for naming convention)
    """
    subimages = [] # list that will hold the number subimages
    R = 526 # half the width/height of the homographied image
    # get the number in the center of the board using `create_bb` (create_bb returns the top left and bottom right coordinates of the image)
    x1, y1, x2, y2 = create_bb(image,
                               (R, R),
                               side_length)
    # Get the subimage from the homographied image
    subimage_center = image[y1:y2, x1:x2] # calculate the center of the image
    # Append the center number to the subimages list
    subimages.append(subimage_center)
    # This gets all the numbers that fall on the line between the center point of the board and the perimiter points
    for pt in dst_points:
        delta_x = R - pt[0]
        delta_y = R - pt[1]
        # it just happens that the numbers surrounding the center hex are 7/24ths of the way between the center and a perimeter point (the center is 7/24ths of the way) NOTE*** this was found through experimentation as well as the second number along the line
        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (7/24)*delta_x)), 
                                    int((R + (7/24)*delta_y))), 
                                   side_length)
        # get the subimage from the passed in image
        subimage1 = image[y1:y2, x1:x2]

        # Get the second image across the line (it just happens to be 3/5ths of the way across)
        x1, y1, x2, y2 = create_bb(image, 
                                   (int((R + (3/5)*delta_x)), 
                                    int((R + (3/5)*delta_y))), 
                                    side_length)
        # get the subimage from the passed in image
        subimage2 = image[y1:y2, x1:x2]

        # append the two images found to `subimages`
        subimages.append(subimage1)
        subimages.append(subimage2)

    # This gets all the numbers not found by the center lines to the perimiter points using every other perimiter point to find the numbers
    for i in range(len(dst_points)):
        # Look at perimeter points two spaces away from each other (the line connecting these points has numbers that cannot be found from a line drawn from the center of the image)
        next_point_ind = i+2
        # If the next index is out of the range, wrap to the beginning of the list
        if i+2 >= len(dst_points):
            next_point_ind = i+2-len(dst_points)

        # Get the two perimeter points we're looking at currently
        pt1 = [dst_points[i][0], dst_points[i][1]]
        pt2 = [dst_points[next_point_ind][0], dst_points[next_point_ind][1]]

        # Calculate the distance from one point to the next
        delta_x = pt2[0] - pt1[0]
        delta_y = pt2[1] - pt1[1]

        ## Get the point 1/3 across the line connecting the two points (this doesn't work perfectly, and needs to be adjusted)
        cir_pt = (int((pt1[0] + (1/3)*delta_x)), int((pt1[1] + (1/3)*delta_y)))

        ### Shift the circle toward the center (this more accurately gets the image of the number)
        xfc = R - cir_pt[0] # x distance from center
        yfc = R - cir_pt[1] # y distance from center

        shift_factor = 0.10 # Shifts the center point closer to the center by a factor of this much

        # Shift the center point of the number image toward the center by a factor of the shift factor
        shifted_x = int(cir_pt[0] + shift_factor * xfc)
        shifted_y = int(cir_pt[1] + shift_factor * yfc)

        # Get the bounding box coords of the number (NOTE: only the first number is collected since the same numbers would be collected twice if the second number across the line was also collected)
        x1, y1, x2, y2 = create_bb(image, (shifted_x, shifted_y), sl=side_length)

        # Get the subimage using the bounding box coords and append to subimages
        subimage = image[y1:y2, x1:x2]
        subimages.append(subimage)

    # Save the images to `save_dir`
    for img in subimages:
        if imgNum < 10:
            saveImage(f"00{imgNum}.jpg", img, save_dir)
        elif imgNum < 100:
            saveImage(f"0{imgNum}.jpg", img, save_dir)
        else:
            saveImage(f"{imgNum}.jpg", img, save_dir)
        imgNum += 1
    
    return imgNum

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

    resnet_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device))
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
    # Set the list of class names in the correct order (same as the neural network)
    CLASS_NAMES = ['desert','eight','eleven','five','four','nine','six','ten','three','twelve','two']
    # Get the number of classes
    CLASS_CNT = len(CLASS_NAMES) # eleven classes to be predicted
    # Get the save path of the model file
    MODEL_SAVE_PATH = "./CATANIST/models/catanistv2_1.pth"
    # List where the predicted labels will be stored in addition to the confidence levels of each class
    LABELS = []

    # Device agnostic code
    device = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
            )
    
    print(f"DEVICE: {device}\n")

    # Instantiate the model (same as the trained model)
    resnet_model = ResNet(input_shape=3, 
                        block=BasicBlock,
                        layers=[2, 2, 2],
                        class_cnt=CLASS_CNT).to(device)

    # Load the state dictionary to get the pretrained model's parameters
    resnet_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    # Set the model to the device
    resnet_model.to(device)

    # Define the image transform (image pre-processing)
    input_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.Grayscale(num_output_channels=3),
    ])
    # Put the model into eval mode
    resnet_model.eval()
    # Read the image in using torchvision so it is the correct type/format
    image = torchvision.io.read_image(str(img_path)).type(torch.float32)
    # Get pixel values between 0 and 1 instead of 0-255
    image /= 255
    # put the image through the image transform defined earlier in the function
    transformed_img = input_transform(image[:3, :, :])
    # Enable torch inference mode
    with torch.inference_mode():
        # Get the image prediction (the model outputs raw logits)
        img_pred = resnet_model((transformed_img.unsqueeze(0)).to(device))
    # Interpret the output logits using softmax and then determining what class the model is most confident 
    # Logits -> Predictions probabilites -> Prediction labels -> class name
    img_pred_probs = torch.softmax(img_pred, dim=1)
    img_pred_label = torch.argmax(img_pred_probs, dim=1)
    # Get the actual class label 
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
        labels = adjust_num_labels(labels, pred_probs, class_names, valid_board, predicted_board)
    return labels

def adjust_num_labels(labels, pred_probs, class_names, valid_board, predicted_board):
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

def save_hex_images(image, save_dir, dst_points, side_length, num_offset, imgNum):
    """ Get all the images for a NN to identify hex types
    image: cv2 read in image of the homographied Catan board
    save_dir: directory to save the hex images
    dst_points: perimiter points of the board
    side_length: side length of bounding box for each number
    num_offset: the number of pixels above the number point
    """
    subimages = [] # list of where the hex images will be saved
    R = 526 # half the height/width
    # Get the hex subimage of the center hex
    x1, y1, x2, y2 = create_bb(image,
                               (R, R - num_offset),
                               side_length)
    # Collect the subimage from the passed in image
    subimage_center = image[y1:y2, x1:x2]
    # Append the subimage to the list of subimages
    subimages.append(subimage_center)
    # This section gets all the numbers that fall on the line between the center point of the board and the perimiter points
    for pt in dst_points:
        delta_x = R - pt[0]
        delta_y = R - pt[1]
        
        # NOTE: the subimages are collected almost identically to the number images in `save_num_images` except the center point is above the center point of the number image by `num_offset` which is a parameter passed into this function
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
        if imgNum < 10:
            saveImage(f"00{imgNum}.jpg", img, save_dir)
        elif imgNum < 100:
            saveImage(f"0{imgNum}.jpg", img, save_dir)
        else:
            saveImage(f"{imgNum}.jpg", img, save_dir)
        imgNum += 1

    return imgNum

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

    resnet_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH, map_location=device))
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
        PRED_PROBS.append(img_pred_probs.cpu())
        LABELS.append(img_label)

    LABELS = validate_hex_predictions(LABELS, torch.stack(PRED_PROBS), CLASS_NAMES)

    return LABELS

def validate_hex_predictions(labels, pred_probs, class_names):
    valid_board = { 'brick': 3,
                    'ore': 3,
                    'wood': 4,
                    'wheat': 4,
                    'sheep': 4,
                    'desert': 1 }
    predicted_board = { 'brick': 0,
                        'ore': 0,
                        'wood': 0,
                        'wheat': 0,
                        'sheep': 0,
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
        labels = adjust_hex_labels(labels, pred_probs, class_names, valid_board, predicted_board)
    return labels

def adjust_hex_labels(labels, pred_probs, class_names, valid_board, predicted_board):
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
    # labels = validate_hex_predictions(labels, pred_probs, class_names)

    # print(f"Correct labels: {labels}")

    # Return valid labels
    return labels

def order_labels(labels):
    """This function is to order the images and labels so that the board is read left to right in terms of hexes"""
    # This is the correct way to order the number/hex images to read left to right starting at the top most row of the homographied image
    order = [2, 15, 16, 12, 1, 4, 11, 3, 14, 0, 17, 9, 5, 10, 7, 6, 13, 18, 8]
    ordered_labels = [] # list that will store the labels in the correct order
    for i in order:
        ordered_labels.append(labels[i])
    return ordered_labels

def main():
    # This is all to create more training images right now
    img_dir = "./images/v6/board09.jpeg"
    num_save_dir = "./images/eval numbers/"
    hex_save_dir = "./images/eval hexes"
    # image_dirs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
    # ind = 190
    NUM_SIDE_LENGTH = 60
    HEX_SIDE_LENGTH = 40
    HEX_OFFSET = 55
    # print(image_dirs)
    image = cv2.imread(img_dir)
    hom_img, perimeter_pts = homography_board(image, vis=False, hom=True)
    save_num_images(hom_img, perimeter_pts, NUM_SIDE_LENGTH, num_save_dir, 0)
    save_hex_images(hom_img, hex_save_dir, perimeter_pts, HEX_SIDE_LENGTH, HEX_OFFSET, 0)
    num_labels = pred_nums_on_resnet(num_save_dir)
    hex_labels = predict_hexes_on_resnet(hex_save_dir)
    # for i in range(len(num_labels)):
    #     # if one model predicts desert, trust that prediction
    #     if num_labels[i] == 'desert' and hex_labels[i] != num_labels[i]:
    #         hex_labels[i] = 'desert'
    #     elif hex_labels[i] == 'desert' and num_labels[i] != hex_labels[i]:
    #         num_labels[i] = 'desert'
    #     print(f"{num_labels[i]} | {hex_labels[i]}")

    # for image_dir in image_dirs:
    #     image = cv2.imread(image_dir)
    #     hom_img, perimeter_points = homography_board(image)
    #     ind = save_num_images(hom_img, perimeter_points, SIDE_LENGTH, save_dir, ind)
    # print("Done saving images")

if __name__ == "__main__":
    main()