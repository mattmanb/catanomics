from predict_board import *
import numpy as np

def score_junctions(junctions):
    # These are the scores of each number (probability wise)
    number_scores = {}
    number_scores['two'], number_scores['twelve'] = 1, 1
    number_scores['three'], number_scores['eleven'] = 2, 2
    number_scores['four'], number_scores['ten'] = 3, 3
    number_scores['five'], number_scores['nine'] = 4, 4
    number_scores['six'], number_scores['eight'] = 5, 5
    number_scores['desert'] = 0
    junction_scores = []
    for junction in junctions:
        junction_score = 0
        resources = set()
        for hex in junction:
            number = hex[0]
            if hex[1] != 'desert':
                resources.add(hex[1])
            junction_score += number_scores[number]
        junction_score += 0.5 * len(resources)
        junction_scores.append([junction_score, junction])
    return junction_scores

def get_board_layout(input_image):
    # Constants
    NUM_IMG_SIDE_LENGTH = 60
    HEX_IMG_SIDE_LENGTH = 40
    HEX_IMG_OFFSET = 55 # number of pixels above the number images to fetch hex imgs
    num_imgs_dir = "./images/eval numbers"
    hex_imgs_dir = "./images/eval hexes"
    ind = 0 # This is for naming images
    board_data = []
    junctions = []

    # Get the homographied board
    hom_img, corners = homography_board(input_image)
    cv2.imwrite("./static/uploads/hom_img.jpg", hom_img)
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
    ### Now lets create all the junctions (access via [row][column])
    ## The comments next to junction appends is on a sample board to help me keep track of all junctions
    # First row of placement spots
    junctions.append([board_data[0][0]]) # 6 sheep, COAST
    junctions.append([board_data[0][0]]) # 6 sheep, COAST
    # Second row
    junctions.append([board_data[1][0]]) # 9 brick COAST
    junctions.append([board_data[0][0], board_data[1][0]]) # 9 brick, 6 sheep, COAST
    junctions.append([board_data[0][0], board_data[1][1]]) # 9 wood, 6 sheep, COAST
    junctions.append([board_data[1][1]]) # 9 wood, COAST
    # Third row
    junctions.append([board_data[2][0]]) # 4 wheat COAST
    junctions.append([board_data[1][0], board_data[2][0]]) # 9 brick, 4 wheat, COAST
    junctions.append([board_data[0][0], board_data[1][0], board_data[2][1]]) # 6 sheep, 9 brick, DESERT
    junctions.append([board_data[0][0], board_data[1][1], board_data[2][1]]) # 6 sheep, 9 wood, DESERT
    junctions.append([board_data[1][1], board_data[2][2]]) # 9 wood, 8 wheat, COAST
    junctions.append([board_data[2][2]]) # 8 wheat, COAST
    # Fourth row
    junctions.append([board_data[2][0]]) # 4 wheat, COAST
    junctions.append([board_data[1][0], board_data[2][0], board_data[3][0]]) # 9 brick, 4 wheat, 10 ore
    junctions.append([board_data[1][0], board_data[2][1], board_data[3][0]]) # 9 brick, DESERT, 10 ore
    junctions.append([board_data[1][1], board_data[2][1], board_data[3][1]]) # 9 wood, DESERT, 3 sheep
    junctions.append([board_data[1][1], board_data[2][2], board_data[3][1]]) # 9 wood, 8 wheat, 3 sheep
    junctions.append([board_data[2][2]]) # 8 wheat, COAST
    # Fifth row
    junctions.append([board_data[2][0], board_data[4][0]]) # 4 wheat, 10 sheep, COAST
    junctions.append([board_data[2][0], board_data[3][0], board_data[4][0]]) # 4 wheat, 10 ore, 10 sheep
    junctions.append([board_data[2][1], board_data[3][0], board_data[4][1]]) # DESERT, 10 ore, 5 wood
    junctions.append([board_data[2][1], board_data[3][1], board_data[4][1]]) # DESERT, 3 sheep, 5 wood
    junctions.append([board_data[2][2], board_data[3][1], board_data[4][2]]) # 8 wheat, 3 sheep, 2 wood
    junctions.append([board_data[2][2], board_data[4][2]]) # 8 wheat, 2 wood, COAST
    # Sixth row
    junctions.append([board_data[4][0]]) # 10 sheep, COAST
    junctions.append([board_data[3][0], board_data[4][0], board_data[5][0]]) # 10 ore, 10 sheep, 11 brick
    junctions.append([board_data[3][0], board_data[4][1], board_data[5][0]]) # 10 ore, 5 wood, 11 brick
    junctions.append([board_data[3][1], board_data[4][1], board_data[5][1]]) # 3 sheep, 5 wood, 5 wheat
    junctions.append([board_data[3][1], board_data[4][2], board_data[5][1]]) # 3 sheep, 2 wood, 5 wheat
    junctions.append([board_data[4][2]]) # 2 wood, COAST
    # Seventh row
    junctions.append([board_data[4][0], board_data[6][0]]) # 10 sheep, 3 wheat, COAST
    junctions.append([board_data[4][0], board_data[5][0], board_data[6][0]]) # 10 sheep, 11 brick, 3 wheat
    junctions.append([board_data[4][1], board_data[5][0], board_data[6][1]]) # 5 wood, 11 brick, 12(3) ore
    junctions.append([board_data[4][1], board_data[5][1], board_data[6][1]]) # 5 wood, 5 wheat, 12(3) ore
    junctions.append([board_data[4][2], board_data[5][1], board_data[6][2]]) # 2 wood, 5 wheat, 11 ore
    junctions.append([board_data[4][2], board_data[6][2]]) # 2 wood, 11 ore, COAST
    # Eighth row
    junctions.append([board_data[6][0]]) # 3 wheat, COAST
    junctions.append([board_data[5][0], board_data[6][0], board_data[7][0]]) # 11 brick, 3 wheat, 8 sheep
    junctions.append([board_data[5][0], board_data[6][1], board_data[7][0]]) # 11 brick, 12(3) ore, 8 sheep
    junctions.append([board_data[5][1], board_data[6][1], board_data[7][1]]) # 5 wheat, 12(3) ore, 4 wood
    junctions.append([board_data[5][1], board_data[6][2], board_data[7][1]]) # 5 wheat, 11 ore, 4 wood
    junctions.append([board_data[6][2]]) # 11 ore
    # Ninth row
    junctions.append([board_data[6][0]]) # 3 wheat, COAST
    junctions.append([board_data[6][0], board_data[7][0]]) # 3 wheat, 8 sheep, COAST
    junctions.append([board_data[6][1], board_data[7][0], board_data[8][0]]) # 12(3) ore, 8 sheep, 6 brick
    junctions.append([board_data[6][1], board_data[7][1], board_data[8][0]]) # 12(3) ore, 4 wood, 6 brick
    junctions.append([board_data[6][2], board_data[7][1]]) # 11 ore, 4 wood, COAST
    junctions.append([board_data[6][2]]) # 11 ore, COAST
    # Tenth row
    junctions.append([board_data[7][0]]) # 8 sheep, COAST
    junctions.append([board_data[7][0], board_data[8][0]]) # 8 sheep, 6 brick, COAST
    junctions.append([board_data[7][1], board_data[8][0]]) # 4 wood, 6 brick, COAST
    junctions.append([board_data[7][1]]) # 4 wood, COAST
    # Eleventh (last) row
    junctions.append([board_data[8][0]]) # 6 brick, COAST
    junctions.append([board_data[8][0]]) # 6 brick, COAST

    return junctions