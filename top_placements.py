from get_board_layout import *

def get_top_placements(file_path):
    junctions = get_board_layout(input_image=file_path)
    junction_scores = score_junctions(junctions)
    # print(junction_scores)
    top_ten_settle_spots = sorted(junction_scores, reverse=True)[:10]
    output_string = "Top ten settlement spots:\n"
    for ind, spot in enumerate(top_ten_settle_spots):
        output_string += f"{ind+1}: |"
        for hex in spot[1]:
            output_string += f" {hex[0]}-{hex[1]} |"
        output_string += '\n'
    return output_string