import torch
from torchvision import transforms
from resnet import BasicBlock, ResNet

import cv2
import numpy as np

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

def homography_board(image):
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
    overlapping_indexes = set()

    while len(lines) - len(overlapping_indexes) > 6:
        for i in range(len(lines)):
            rho, _ = lines[i, 0]
            for j in range(i+1, len(lines)):
                rho2, _ = lines[j, 0]
                if abs(rho-rho2) < min_rho_diff:
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

def pred_nums_on_resnet(images) -> list:
    """
    images - list of subimages of the numbers of a Catan board

    returns a list of labels in the order the images are passed in
    """
    # Set up and load the model
    CLASS_NAMES = ["two", "three", "four", "five", "six", 
                   "eight", "nine", "ten", "eleven", "twelve"]
    CLASS_CNT = len(CLASS_NAMES) # ten numbers to be predicted
    MODEL_SAVE_PATH = "./models/catanist_resnet.pth"
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
        transforms.ToTensor()
    ])
    # Put the model into eval mode
    resnet_model.eval()
    for image in images:
        transformed_img = input_transform(image)
        with torch.inference_mode():
            img_pred = resnet_model(transformed_img.unsqueezze(0).to(device))
        # Logits -> Predictions probabilites -> Prediction labels -> class name
        img_label = CLASS_NAMES[torch.argmax(torch.softmax(img_pred, dim=1), dim=1)]
        LABELS.append(img_label)

    return LABELS