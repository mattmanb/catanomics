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

    # Define the color bounds for the perimiter space
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Create binary mask of the blue perimiter
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Bitwise AND to keep blue parts of the image (gets rid of some noise)
    blue_regions = cv2.bitwise_and(image, image, mask=mask)

    # Perform edge detection
    # Gray scale
    gray_blue_region = cv2.cvtColor(blue_regions, cv2.COLOR_BGR2GRAY)
    # Gaussian Blur
    blurred_blue_regions = cv2.GaussianBlur(gray_blue_region, (5,5), 0)
    # Canny edge detection
    edges = cv2.Canny(blurred_blue_regions, 50, 150)

    # Perform Hough Transform using `HoughLines`
    lines = []
    thresh = 200
    while len(lines) < 10:
        lines = cv2.HoughLines(edges, 1, np.pi / 180, thresh)
        print(f"Trying threshold {thresh} | Got {len(lines)} lines.")
        if len(lines) >= 10:
            break
        else:
            thresh -= 5

    print(f"Results: {thresh} threshold | {len(lines)} lines")

    # Draw lines on the image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    showImage(image)
    return image

def main():
    save_dir = "../images/perimiter/v1"
    img_dir = "v3"
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