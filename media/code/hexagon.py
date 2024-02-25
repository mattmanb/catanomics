import turtle

# Create a turtle object
pen = turtle.Turtle()

# Set the speed of the turtle
pen.speed(1)  # Speeds can be set from 1 (slowest) to 10 (fastest)

# Loop to draw each side of the hexagon
for _ in range(6):
    pen.forward(100)  # Move the turtle forward by 100 units
    pen.right(60)     # Turn the turtle right by 60 degrees

# To prevent the window from closing immediately
turtle.done()
