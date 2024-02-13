# Import necessary libraries
import cv2
import numpy as np
import os

# Import helper computer vision function
from helper_functions import *

def findCatanBoardPerimiter(image):
    # Resize (assuming the images passed in are large)
    image = cv2.resize(image, (0,0), fx=.25, fy=.25)

    # Convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    showImage(hsv_image)

    # Define the color bounds for the perimiter space
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Create binary mask of the blue perimiter
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    showImage(mask)

    # Bitwise AND to keep blue parts of the image (gets rid of some noise)
    blue_regions = cv2.bitwise_and(image, image, mask=mask)
    showImage(blue_regions)

    # Perform edge detection
    # Gray scale
    gray_blue_region = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blurred_blue_regions = cv2.GaussianBlur(gray_blue_region, (5,5), 0)
    # Canny edge detection
    edges = cv2.Canny(blurred_blue_regions, 50, 150)
    showImage(edges)

    # Get two rotated images to find the side points
    LEFT_ANGLE = 4
    left_img = rotate_image(edges, LEFT_ANGLE)
    showImage(left_img)
    RIGHT_ANGLE = -4
    right_img = rotate_image(edges, RIGHT_ANGLE)
    showImage(right_img)

    # Find the top and bottom points
    edge_y, edge_x = np.nonzero(edges)
    bottom_y = np.max(edge_y)
    top_y = np.min(edge_y)
    x_values_at_max_y = edge_x[edge_y == bottom_y]
    x_values_at_min_y = edge_x[edge_y == top_y]
    bottom_x = int(np.median(x_values_at_max_y))
    top_x = int(np.median(x_values_at_min_y))

    # Find the top left and bottom right points of the perimiter
    ul_lr_y_edges, ul_lr_x_edges = np.nonzero(left_img)
    lr_x = np.max(ul_lr_x_edges)
    ul_x = np.min(ul_lr_x_edges)
    lr_y_values = ul_lr_y_edges[ul_lr_x_edges == lr_x]
    ul_y_values = ul_lr_y_edges[ul_lr_x_edges == ul_x]
    lr_y = int(np.median(lr_y_values))
    ul_y = int(np.median(ul_y_values))
    # Inverse rotation on these points to get the points on the original image
    height, width = left_img.shape[:2]
    ul_lr_center = (width / 2, height / 2)
    upper_left_x, upper_left_y = rotate_point_back(
                                    x_prime = ul_x, 
                                    y_prime=ul_y, 
                                    angle_deg=LEFT_ANGLE, 
                                    center=ul_lr_center
                                 )
    lower_right_x, lower_right_y = rotate_point_back(
                                    x_prime=lr_x,
                                    y_prime=lr_y,
                                    angle_deg=LEFT_ANGLE,
                                    center=ul_lr_center
                                   )
    
    # Find the top right and bottom left points on the perimiter of the board
    ur_ll_y_edges, ur_ll_x_edges = np.nonzero(right_img)
    ur_x = np.max(ur_ll_x_edges)
    ll_x = np.min(ur_ll_x_edges)
    ll_y_values = ur_ll_y_edges[ur_ll_x_edges == ll_x]
    ur_y_values = ur_ll_y_edges[ur_ll_x_edges == ur_x]
    ll_y = int(np.median(ll_y_values))
    ur_y = int(np.median(ur_y_values))
    height, width = right_img.shape[:2]
    ur_ll_center = (width / 2, height / 2)
    upper_right_x, upper_right_y = rotate_point_back(
                                    x_prime = ur_x,
                                    y_prime=ur_y,
                                    angle_deg = RIGHT_ANGLE,
                                    center = ur_ll_center
                                   )
    lower_left_x, lower_left_y = rotate_point_back(
                                    x_prime = ll_x,
                                    y_prime = ll_y,
                                    angle_deg = RIGHT_ANGLE,
                                    center = ur_ll_center
                                 )
    
    # draw the perimiter of the board
    cv2.line(image, (top_x, top_y), (upper_right_x, upper_right_y), (0,0,255), 5)
    cv2.line(image, (upper_right_x, upper_right_y), (lower_right_x, lower_right_y), (0,0,255), 5)
    cv2.line(image, (lower_right_x, lower_right_y), (bottom_x, bottom_y), (0,0,255), 5)
    cv2.line(image, (bottom_x, bottom_y), (lower_left_x, lower_left_y), (0,0,255), 5)
    cv2.line(image, (lower_left_x, lower_left_y), (upper_left_x, upper_left_y), (0,0,255), 5)
    cv2.line(image, (upper_left_x, upper_left_y), (top_x, top_y), (0,0,255), 5)

    showImage(image)
    return image

def main():
    save_dir = "./images/perimiter"
    img_dir = "v0"
    read_pth = f"../images/{img_dir}" 
    files = os.listdir(read_pth)
    image_number=0
    
    # Iterate through images save the perimiter images
    for file_name in files:
        file_path = os.path.join(read_pth, file_name)
        if os.path.isfile(file_path):
            # Read the image
            image = cv2.imread(file_path)
            if image is not None:
                if image_number < 10:
                    image_name = f"{img_dir}_perimiter_0{image_number}.jpg"
                else:
                    image_name = f"{img_dir}_perimiter_ {image_number}.jpg"
                perimiter_image = findCatanBoardPerimiter(image)
                saveImage(image_name, perimiter_image, save_dir)
            else:
                print(f"Error reading in image {file_path}")

if __name__=="__main__":
    main()