import tkinter as tk
import math
import random

WIDTH, HEIGHT = 600, 600
BALL_RADIUS = 10
GRAVITY = 0.5
FRICTION = 0.99
BOUNCE = 0.9
KICKBACK = 10  # Kickback velocity magnitude

class Ball:
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def update(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy
        self.vx *= FRICTION
        self.vy *= FRICTION

class Polygon:
    def __init__(self, sides, cx, cy, radius):
        self.sides = sides
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.vertices = self.calculate_vertices()

    def calculate_vertices(self):
        vertices = []
        for i in range(self.sides):
            angle = 2 * math.pi * i / self.sides
            x = self.cx + self.radius * math.cos(angle)
            y = self.cy + self.radius * math.sin(angle)
            vertices.append((x, y))
        return vertices

    def increase_sides(self):
        self.sides += 1
        self.vertices = self.calculate_vertices()

    def get_edges(self):
        edges = []
        for i in range(self.sides):
            p1 = self.vertices[i]
            p2 = self.vertices[(i + 1) % self.sides]
            edges.append((p1, p2))
        return edges

def reflect(vx, vy, nx, ny):
    dot = vx * nx + vy * ny
    rx = vx - 2 * dot * nx
    ry = vy - 2 * dot * ny
    return rx * BOUNCE, ry * BOUNCE

def point_line_distance(px, py, x1, y1, x2, y2):
    line_mag = math.hypot(x2 - x1, y2 - y1)
    if line_mag < 1e-8:
        return math.hypot(px - x1, py - y1), (x1, y1)
    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
    u = max(0, min(1, u))
    ix = x1 + u * (x2 - x1)
    iy = y1 + u * (y2 - y1)
    return math.hypot(px - ix, py - iy), (ix, iy)

def is_inside_polygon(x, y, vertices):
    # Ray casting algorithm
    n = len(vertices)
    inside = False
    px, py = x, y
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[(i + 1) % n]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-8) + xi):
            inside = not inside
    return inside

MAX_BALLS = 15  # Set your desired maximum number of balls

class App:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack()
        self.polygon = Polygon(3, WIDTH // 2, HEIGHT // 2, 250)
        angle = random.uniform(0, 2 * math.pi)
        self.balls = [
            Ball(self.polygon.cx + 100 * math.cos(angle),
                 self.polygon.cy + 100 * math.sin(angle),
                 random.uniform(-5, 5), random.uniform(-5, 5))
        ]
        self.animate()

    def animate(self):
        new_balls = []
        hit = False
        for ball in self.balls:
            ball.update()
            # Prevent ball from escaping: check if inside polygon, if not, reflect back
            if not is_inside_polygon(ball.x, ball.y, self.polygon.vertices):
                # Find closest edge and reflect
                min_dist = float('inf')
                closest = None
                for (x1, y1), (x2, y2) in self.polygon.get_edges():
                    dist, (ix, iy) = point_line_distance(ball.x, ball.y, x1, y1, x2, y2)
                    if dist < min_dist:
                        min_dist = dist
                        closest = (x1, y1, x2, y2, ix, iy)
                if closest:
                    x1, y1, x2, y2, ix, iy = closest
                    dx, dy = x2 - x1, y2 - y1
                    nx, ny = -dy, dx
                    norm = math.hypot(nx, ny)
                    if norm != 0:
                        nx /= norm
                        ny /= norm
                        ball.vx, ball.vy = reflect(ball.vx, ball.vy, nx, ny)
                        ball.x = ix + nx * (BALL_RADIUS + 1)
                        ball.y = iy + ny * (BALL_RADIUS + 1)
            for (x1, y1), (x2, y2) in self.polygon.get_edges():
                dist, (ix, iy) = point_line_distance(ball.x, ball.y, x1, y1, x2, y2)
                if dist <= BALL_RADIUS:
                    dx, dy = x2 - x1, y2 - y1
                    nx, ny = -dy, dx
                    norm = math.hypot(nx, ny)
                    if norm == 0:
                        continue
                    nx /= norm
                    ny /= norm
                    ball.vx, ball.vy = reflect(ball.vx, ball.vy, nx, ny)
                    # Kickback: add random velocity
                    kick_angle = random.uniform(0, 2 * math.pi)
                    ball.vx += KICKBACK * math.cos(kick_angle)
                    ball.vy += KICKBACK * math.sin(kick_angle)
                    # Move ball out of edge
                    ball.x = ix + nx * (BALL_RADIUS + 1)
                    ball.y = iy + ny * (BALL_RADIUS + 1)
                    hit = True
                    # Add new ball at collision point with random velocity, if limit not reached
                    if len(self.balls) + len(new_balls) < MAX_BALLS:
                        new_angle = random.uniform(0, 2 * math.pi)
                        new_balls.append(Ball(
                            ix + nx * (BALL_RADIUS + 1),
                            iy + ny * (BALL_RADIUS + 1),
                            random.uniform(-5, 5), random.uniform(-5, 5)
                        ))
                    break
        if hit:
            self.polygon.increase_sides()
            self.balls.extend(new_balls)

        self.draw()
        self.root.after(16, self.animate)

    def draw(self):
        self.canvas.delete("all")
        self.canvas.create_polygon(self.polygon.vertices, outline="black", fill="", width=2)
        for ball in self.balls:
            self.canvas.create_oval(
                ball.x - BALL_RADIUS, ball.y - BALL_RADIUS,
                ball.x + BALL_RADIUS, ball.y + BALL_RADIUS,
                fill="red"
            )
        self.canvas.create_text(50, 30, text=f"Sides: {self.polygon.sides}", font=("Arial", 16))
        self.canvas.create_text(50, 60, text=f"Balls: {len(self.balls)}", font=("Arial", 16))

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Bouncing Balls in Growing Polygon")
    App(root)
    root.mainloop()