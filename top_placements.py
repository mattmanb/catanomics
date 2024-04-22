from get_board_layout import *

def get_top_placements(file_path):
    # call `get_board_layout` to get a list of all the junctions of the uploaded Catan board
    junctions = get_board_layout(input_image=file_path)
    # score all the junctions using `score_junctions`
    junction_scores = score_junctions(junctions)
    # Sort all the junctions based on score, and take the ten highest scores
    top_ten_settle_spots = sorted(junction_scores, reverse=True)[:10]
    # Start formatting the output string which will be passed to the Flask app for display
    output_string = "Top ten settlement spots:\n"
    # Format output string by appending each of the top 10 settlement spots
    for ind, spot in enumerate(top_ten_settle_spots):
        output_string += f"{ind+1}: |"
        for hex in spot[1]:
            output_string += f" {hex[0]}-{hex[1]} |"
        output_string += '\n'
    # Return the output string
    return output_string