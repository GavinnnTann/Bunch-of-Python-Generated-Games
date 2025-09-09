import tkinter as tk
import random
import math

WIDTH, HEIGHT = 800, 600
CROSSHAIR_SIZE = 20
TARGET_RADIUS = 30
TARGET_SPEED = 10

class FPSGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple FPS Game")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()
        self.score = 0
        self.targets = []
        self.crosshair_x = WIDTH // 2
        self.crosshair_y = HEIGHT // 2
        self.canvas.bind("<Motion>", self.move_crosshair)
        self.canvas.bind("<Button-1>", self.shoot)
        self.spawn_target()
        self.update()
        self.draw_crosshair()
        self.score_text = self.canvas.create_text(70, 30, text=f"Score: {self.score}", fill="white", font=("Arial", 20))

    def spawn_target(self):
        x = random.randint(TARGET_RADIUS, WIDTH - TARGET_RADIUS)
        y = random.randint(TARGET_RADIUS, HEIGHT - TARGET_RADIUS)
        angle = random.uniform(0, 2 * math.pi)
        dx = TARGET_SPEED * math.cos(angle)
        dy = TARGET_SPEED * math.sin(angle)
        self.targets.append({'x': x, 'y': y, 'dx': dx, 'dy': dy})

    def move_crosshair(self, event):
        self.crosshair_x = event.x
        self.crosshair_y = event.y

    def shoot(self, event):
        for target in self.targets:
            dist = math.hypot(event.x - target['x'], event.y - target['y'])
            if dist <= TARGET_RADIUS:
                self.targets.remove(target)
                self.score += 1
                self.canvas.itemconfig(self.score_text, text=f"Score: {self.score}")
                self.spawn_target()
                break

    def update(self):
        self.canvas.delete("target")
        for target in self.targets:
            target['x'] += target['dx']
            target['y'] += target['dy']
            # Bounce off walls
            if target['x'] < TARGET_RADIUS or target['x'] > WIDTH - TARGET_RADIUS:
                target['dx'] *= -1
            if target['y'] < TARGET_RADIUS or target['y'] > HEIGHT - TARGET_RADIUS:
                target['dy'] *= -1
            self.canvas.create_oval(
                target['x'] - TARGET_RADIUS, target['y'] - TARGET_RADIUS,
                target['x'] + TARGET_RADIUS, target['y'] + TARGET_RADIUS,
                fill="red", tags="target"
            )
        self.draw_crosshair()
        self.root.after(20, self.update)

    def draw_crosshair(self):
        self.canvas.delete("crosshair")
        x, y = self.crosshair_x, self.crosshair_y
        self.canvas.create_line(x - CROSSHAIR_SIZE, y, x + CROSSHAIR_SIZE, y, fill="white", width=2, tags="crosshair")
        self.canvas.create_line(x, y - CROSSHAIR_SIZE, x, y + CROSSHAIR_SIZE, fill="white", width=2, tags="crosshair")

if __name__ == "__main__":
    root = tk.Tk()
    game = FPSGame(root)
    root.mainloop()