import turtle
from PIL import Image
import os

# Init some constants
HEXAGON_LENGTHS = 200

# Create a turtle object
pen = turtle.Turtle()

# Set the speed of the turtle
pen.speed(6)  # Speeds can be set from 1 (slowest) to 10 (fastest)

# Move cursor to the left side before beginning
pen.left(180)
pen.forward(HEXAGON_LENGTHS)
pen.left(180)

# Loop to draw each side of the first hexagon
for _ in range(6):
    pen.forward(HEXAGON_LENGTHS)  # Move the turtle forward by 100 units
    pen.right(60)     # Turn the turtle right by 60 degrees

# Get to starting position of second hexagon
pen.forward(HEXAGON_LENGTHS)
pen.left(60)
pen.forward(HEXAGON_LENGTHS)
pen.right(60)

# Draw second hexagon
for _ in range(4):
    pen.forward(HEXAGON_LENGTHS)  # Move the turtle forward by 100 units
    pen.right(60)     # Turn the turtle right by 60 degrees

# Draw third hexagon
pen.left(60)
for _ in range(5):
    pen.left(60)
    pen.forward(HEXAGON_LENGTHS)

# Go to the center
pen.left(60)
pen.forward(HEXAGON_LENGTHS)

# Circle the 3-way junction
pen.right(60)
pen.forward(50)
pen.left(90)
pen.circle(50)

# Hide the turtle
pen.hideturtle()

# filepaths
ps_output_dir = "./postscript_files"
full_path = os.path.join(ps_output_dir, "hexagon_junction.ps")

# Save the image as PostScript file
canvas = turtle.getcanvas()
canvas.postscript(file=full_path)

# To prevent the window from closing immediately
turtle.done()
