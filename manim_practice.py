from manim import *

# Scenes are within classes

# self.play() Plays animations
# self.wait() Waits
# self.add() Adds an object
# self.remove() Removes an object

class Test(Scene):
    def construct(self):
        c = Circle(2, color=RED, fill_opacity = 0.1)

        self.play(DrawBorderThenFill(c), run_time = 0.5)

        title = Text("Catanomics", font_size = 48, slant="ITALIC")
        self.play(Write(title))

        a = Arc(2.2, TAU * 1 / 4, -TAU * 2.6 / 4, color = YELLOW, stroke_width=15)
        self.play(Create(a))

        self.wait(3)