import turtle

def draw_hexagon_and_write_number(side_length, number):
    # Setup turtle environment
    window = turtle.Screen()
    window.title("Hexagon with Number")
    hex_turtle = turtle.Turtle()
    
    # Function to draw hexagon
    def draw_hexagon():
        hex_turtle.penup()
        hex_turtle.goto(0, -side_length / 2)  # Starting point at bottom of hexagon
        hex_turtle.pendown()
        hex_turtle.left(30)  # Orient for hexagon drawing
        for _ in range(6):
            hex_turtle.forward(side_length)
            hex_turtle.left(60)
        hex_turtle.right(30)  # Reset orientation
    
    # Function to write number in center
    def write_number():
        hex_turtle.penup()
        hex_turtle.goto(0, -15)  # Roughly center; adjust based on side_length for exact centering
        hex_turtle.pendown()
        hex_turtle.write(number, move=False, align="center", font=("Arial", 16, "normal"))
        hex_turtle.penup()  # Stop drawing
    
    # Execute drawing and writing
    draw_hexagon()
    write_number()

    # Complete the drawing
    window.mainloop()

# Example usage: Draw a hexagon with side length 100 and write '5' in the center
draw_hexagon_and_write_number(25, 5)
