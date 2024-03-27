from predict_numbers import homography_board
import os

def main():
    dir_path = "./images/v5"
    image_number = input("Enter image number to examine: ")
    image_path = f"{dir_path}/board0{image_number}.jpg"
    homographied_image = homography_board(image_path, True)

if __name__ == "__main__":
    main()