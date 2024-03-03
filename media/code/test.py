from turtle import Screen, Turtle

screen = Screen()
screen.setup(600, 480)  # what's visible
screen.screensize(1200, 960)  # total backing store

turtle = Turtle()
turtle.hideturtle()
turtle.speed('fastest')

turtle.penup()
turtle.sety(-100)
turtle.pendown()
turtle.circle(100)  # visible

turtle.penup()
turtle.sety(-400)
turtle.pendown()
turtle.circle(400)  # offscreen

canvas = screen.getcanvas()

canvas.postscript(file="test.eps", x=-600, y=-480, width=1200, height=960)

# wait for program to quit, then examine file 'test.eps'