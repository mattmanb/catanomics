from predict_numbers import homography_board, get_num_images, showImage
import os

def main():
    dir_path = "./images/v5"
    image_number = "0"
    image_path = f"{dir_path}/board0{image_number}.jpg"
    homographied_image, corners = homography_board(image_path, True)
    num_imgs = get_num_images(homographied_image, corners, side_length=60)
    order = [2, 15, 16, 12, 1, 4, 11, 3, 14, 0, 17, 9, 5, 10, 7, 6, 13, 18, 8]
    for i in order:
        showImage(num_imgs[i])
    

if __name__ == "__main__":
    main()