def order_labels(labels):
    """This function is to order the images and labels so that the board is read left to right in terms of hexes"""
    # This is the correct way to order the number/hex images to read left to right starting at the top most row
    order = [2, 15, 16, 12, 1, 4, 11, 3, 14, 0, 17, 9, 5, 10, 7, 6, 13, 18, 8]
    ordered_labels = []
    for i in order:
        ordered_labels.append(labels[i])
    return ordered_labels