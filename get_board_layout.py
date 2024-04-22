from predict_board import *
import numpy as np

def score_junctions(junctions):
    # These are the scores of each number (probability wise)
    number_scores = {}
    # The scores of each number depends on the number of dots beneath the number on the Catan board (each dot represents a 1/36 chance for the sum to be rolled with 2d6)
    number_scores['two'], number_scores['twelve'] = 1, 1
    number_scores['three'], number_scores['eleven'] = 2, 2
    number_scores['four'], number_scores['ten'] = 3, 3
    number_scores['five'], number_scores['nine'] = 4, 4
    number_scores['six'], number_scores['eight'] = 5, 5
    # The desert doesn't produce resources
    number_scores['desert'] = 0 
    junction_scores = [] # list that will store the score of each junction as well as the junction itself per element i.e. [junction_score, junction]
    for junction in junctions:
        # start at 0 for the junction
        junction_score = 0
        # store each unique resource that gets produced at the junction
        resources = set()
        # Iterate through each hex adjecent to the junction
        for hex in junction:
            # get the number for the hex
            number = hex[0]
            # add the resource to the set of resources this junction has if it isn't 'desert'
            if hex[1] != 'desert':
                resources.add(hex[1])
            # add the score of the number to the total junction_score
            junction_score += number_scores[number]
        # for each unique resource that lies adjacent to the junction, add 0.5 to the score
        junction_score += 0.5 * len(resources)
        # Add the junction score and junction itself as a list to the junction_scores list
        junction_scores.append([junction_score, junction])
    # Return all the junction scores paired with the junction itself
    return junction_scores

def get_board_layout(input_image):
    # Constants
    NUM_IMG_SIDE_LENGTH = 60 # This is the side length of the subimages of numbers on the Catan board (60 is enough to get the whole number consistently as long as the numbers are placed close to the center of each hex)
    HEX_IMG_SIDE_LENGTH = 40 # This is the side length of the subimages of hex numbers on the Catan board
    HEX_IMG_OFFSET = 55 # number of pixels above the number images to fetch hex imgs
    num_imgs_dir = "./images/eval numbers" # directory where the cropped number images will be saved (temporarily)
    hex_imgs_dir = "./images/eval hexes" # directory for saving cropped hex images
    imgNum = 0 # The number appending to the image (numbered to ensure the correct number of images are collected)
    # `board_data` will store each hex's information in the order it is collected (so it is NOT sorted)
    board_data = [] 
    # `junctions` will store the information reguarding each junction, which is 1-3 hexes
    junctions = []

    # Get the homographied board
    hom_img, corners = homography_board(input_image)
    # Save the homographied image in case a function later reads in a file path instead of a cv2 image (either can be used, previous uploads get overwritten)
    cv2.imwrite("./static/uploads/hom_img.jpg", hom_img)
    # Get the number and hex images
    save_num_images(hom_img, corners, NUM_IMG_SIDE_LENGTH, save_dir=num_imgs_dir, imgNum=0)
    save_hex_images(hom_img, save_dir=hex_imgs_dir, 
                    dst_points=corners, 
                    side_length=HEX_IMG_SIDE_LENGTH, 
                    num_offset=HEX_IMG_OFFSET,
                    imgNum=0)
    # Predict all the numbers and hex types
    num_labels = pred_nums_on_resnet(num_imgs_dir)
    hex_labels = predict_hexes_on_resnet(hex_imgs_dir)
    # Order the labels to the layout of the board (top down)
    num_labels = order_labels(num_labels)
    hex_labels = order_labels(hex_labels)


    # append all the predicted number and hex labels to board_data in a left-right top-bottom order looking at the homographied image
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

    ### Now we create all the junctions (access via [row][column])
    ## The comments next to junction appends is on a SAMPLE BOARD to help me keep track of all junctions
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

    # Return the list of junctions
    return junctions