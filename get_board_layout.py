from misc_functions import order_labels
from predict_numbers import homography_board, showImage, get_num_images, save_num_images, pred_nums_on_resnet, slope, y_intercept, calc_intersection
from predict_hexes import get_hex_images, save_hex_images, predict_hexes_on_resnet

def main():
    # Constants
    NUM_IMG_SIDE_LENGTH = 60
    HEX_IMG_SIDE_LENGTH = 40
    HEX_IMG_OFFSET = 55 # number of pixels above the number images to fetch hex imgs
    input_image = "./images/eval/eval00.jpg"
    num_imgs_dir = "./images/eval numbers"
    hex_imgs_dir = "./images/eval hexes"
    ind = 0 # This is for naming images
    board_data = []

    # Get the homographied board
    hom_img, corners = homography_board(input_image)
    # Get the number and hex images
    save_num_images(hom_img, corners, NUM_IMG_SIDE_LENGTH, save_dir=num_imgs_dir, ind=0)
    save_hex_images(hom_img, save_dir=hex_imgs_dir, 
                    dst_points=corners, 
                    side_length=HEX_IMG_SIDE_LENGTH, 
                    num_offset=HEX_IMG_OFFSET,
                    ind=0)
    # Predict all the numbers and hex types
    num_labels = pred_nums_on_resnet(num_imgs_dir)
    hex_labels = predict_hexes_on_resnet(hex_imgs_dir)
    # Order the labels to the layout of the board (top down)
    num_labels = order_labels(num_labels)
    hex_labels = order_labels(hex_labels)

    # for i in range(len(num_labels)):
    #     print(num_labels[i], hex_labels[i])

    board_data.append([[num_labels[0], hex_labels[0]]])
    board_data.append([[num_labels[1], hex_labels[1]], 
                       [num_labels[2], hex_labels[2]]])
    board_data.append([[num_labels[3], hex_labels[3]], 
                       [num_labels[4], hex_labels[4]],
                       [num_labels[5], hex_labels[5]]])
    board_data.append([[num_labels[6], hex_labels[6]], 
                       [num_labels[7], hex_labels[7]]])
    board_data.append([[num_labels[8], hex_labels[8]], 
                       [num_labels[9], hex_labels[9]],
                       [num_labels[10], hex_labels[10]]])
    board_data.append([[num_labels[11], hex_labels[11]], 
                       [num_labels[12], hex_labels[12]]])
    board_data.append([[num_labels[13], hex_labels[13]], 
                       [num_labels[14], hex_labels[14]],
                       [num_labels[15], hex_labels[15]]])
    board_data.append([[num_labels[16], hex_labels[16]], 
                       [num_labels[17], hex_labels[17]]])
    board_data.append([[num_labels[18], hex_labels[18]]])
    print(board_data)

if __name__=="__main__":
    main()