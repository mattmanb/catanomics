import turtle
from turtle import Screen, Turtle

screen = Screen()
screen.setup(1000,1000)
screen.screensize(1200,960)

pen = turtle.Turtle()
pen.speed(100)
pen.width(5)


def drawHexagon():
    """
    Assume the pen is at the correct starting angle
    """
    for i in range(6):
        pen.forward(SIDE_LENGTH)
        pen.left(60)

SIDE_LENGTH = 45

# Get the correct starting angle
pen.right(30)

# Draw perimiter hexes
for i in range(6):
    drawHexagon()
    for j in range(5):
        # if j % 2 == 0:
        #     drawHexagon()
        pen.forward(SIDE_LENGTH)
        pen.left(60 * ((-1)**j))

# Draw inner layer hexes
pen.up()
pen.forward(SIDE_LENGTH)
pen.left(60)
pen.forward(SIDE_LENGTH)
pen.left(60)
pen.forward(SIDE_LENGTH)
pen.left(60)
pen.forward(SIDE_LENGTH)
pen.left(180)
pen.down()
# left right left x 6
for i in range(6):
    drawHexagon()
    pen.forward(SIDE_LENGTH)
    pen.left(60)
    pen.forward(SIDE_LENGTH)
    pen.right(60)
    pen.forward(SIDE_LENGTH)
    pen.left(60)

# Draw boarder
pen.up()
pen.forward(SIDE_LENGTH)
pen.right(60)
pen.forward(SIDE_LENGTH)
pen.left(60)
pen.forward(SIDE_LENGTH)
pen.right(60)
pen.forward(50)
pen.left(90)
pen.forward(135)
pen.left(60)
pen.down()

colors = ["red", "orange", "yellow", "green", "blue", "purple"]
edge_coordinates = []

for i in range(6):
    # Draw a circle of the coordinate
    pen.color(colors[i])
    # pen.up()
    # pen.forward(3)
    # pen.left(90)
    # pen.down()
    pen.circle(2)
    # pen.up()
    # pen.right(90)
    # pen.forward(3)
    # pen.right(180)
    # pen.down()

    # Log the coordinate and its color
    edge_coordinates.append([pen.xcor()+600, pen.ycor()+480, colors[i]])

    # Draw edge line
    pen.color("black")
    pen.forward(270)
    pen.left(60)

for point in edge_coordinates:
    print(f"({round(point[0])}, {round(point[1])})")

pen.up()
pen.speed(1)
pen.right(60)
pen.goto(0,0)
pen.forward(600)
print(screen.screensize())



# canvas = screen.getcanvas()

# canvas.postscript(file="board.eps", x=-225, y=-500, width=700, height=700)