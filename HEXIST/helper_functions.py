# Import necessary libraries
import cv2
import numpy as np

# Show image function for later use
def showImage(img, name=None):
    if not name:
        cv2.imshow("Image display", img)
    else:
        cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Rotate the image to find the maximum y_value (straighten the image as much as possible)
def rotate_image(image, angle):
    # Get image size
    height, width = image.shape[:2]
    # Calculate the center of the image
    center = (width / 2, height / 2)
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def rotate_point_back(x_prime, y_prime, angle_deg, center):
    """Return the coordinates of a point rotated about a center point

    Keyword arguments:
    x_prime: x-coordinate of a point on a rotated image
    y_prime: y-coordinate of a point on a rotated image
    angle_deg: the degree in which the target image was rotated to get x and y prime
    center: the center of the rotated image
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)
    # Rotate matrix
    rotation_matrix = np.array( [ [np.cos(angle_rad), -np.sin(angle_rad)],
                                  [np.sin(angle_rad), np.cos(angle_rad)] ])
    # Original center
    original_center = np.array(center)

    # Translate points back to origin for rotation
    translated_point = np.array([x_prime, y_prime]) - original_center

    # Apply inverse rotation
    rotated_point = np.dot(rotation_matrix, translated_point)

    # Translate back after rotation
    original_point = rotated_point + original_center

    return int(original_point[0]), int(original_point[1])

# Save the image as `filename` at `dir`
def saveImage(filename, img, dir):
    # Get full path
    full_path = f"{dir}/{filename}"
    cv2.imwrite(full_path, img)
    print(f"Image saved to {full_path}")