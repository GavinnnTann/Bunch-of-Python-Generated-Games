import tkinter as tk
import random
import heapq
def astar_search(snake, food, obstacles, width, height, cell_size):
    # A* that allows stepping into the current tail (which will vacate)
    def neighbors(pos, blocked):
        x, y = pos
        for dx, dy in [(-cell_size,0),(cell_size,0),(0,-cell_size),(0,cell_size)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles and (nx, ny) not in blocked:
                yield (nx, ny)

    def heuristic(a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    start = snake[0]
    # treat all snake segments except the tail as blocked (tail will move)
    blocked = set(snake[:-1])

    def run_astar(goal):
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
        visited = set()
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            if current == goal:
                return path
            if current in visited:
                continue
            visited.add(current)
            for n in neighbors(current, blocked):
                if n not in visited:
                    g = cost + 1
                    f = g + heuristic(n, goal)
                    heapq.heappush(open_set, (f, g, n, path + [n]))
        return None

    # try to find a path to the food first
    path_to_food = run_astar(food)
    if path_to_food:
        return path_to_food

    # if no path to food, try to find a path to the tail to survive
    tail = snake[-1]
    path_to_tail = run_astar(tail)
    return path_to_tail

def dijkstra_search(snake, food, obstacles, width, height, cell_size):
    def neighbors(pos):
        x, y = pos
        for dx, dy in [(-cell_size,0),(cell_size,0),(0,-cell_size),(0,cell_size)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles and (nx, ny) not in snake:
                yield (nx, ny)
    queue = []
    heapq.heappush(queue, (0, snake[0], [snake[0]]))
    visited = set()
    while queue:
        cost, current, path = heapq.heappop(queue)
        if current == food:
            return path
        if current in visited:
            continue
        visited.add(current)
        for n in neighbors(current):
            if n not in visited:
                heapq.heappush(queue, (cost+1, n, path+[n]))
    return None

def get_next_direction(snake, path):
    if len(path) < 2:
        return None
    head, next_pos = path[0], path[1]
    dx, dy = next_pos[0] - head[0], next_pos[1] - head[1]
    if dx > 0:
        return 'Right'
    elif dx < 0:
        return 'Left'
    elif dy > 0:
        return 'Down'
    elif dy < 0:
        return 'Up'
    return None

def auto_play(game, algorithm='astar'):
    if not game.running:
        return

    # helper to call selected planner
    def plan():
        if algorithm == 'astar':
            return astar_search(game.snake, game.food, game.obstacles, game.width, game.height, game.cell_size)
        return dijkstra_search(game.snake, game.food, game.obstacles, game.width, game.height, game.cell_size)

    # initialize planned path on the game object
    if not hasattr(game, 'planned_path'):
        game.planned_path = []

    path = game.planned_path

    head = game.snake[0]

    # If no plan or plan exhausted, compute a new one
    if not path or len(path) < 2:
        path = plan() or []
        game.planned_path = path

    # If the plan doesn't align with current head, try to re-sync or replan
    if path:
        if path[0] != head:
            if head in path:
                idx = path.index(head)
                path = path[idx:]
                game.planned_path = path
            else:
                path = plan() or []
                game.planned_path = path

    # Determine next move from the current plan
    next_pos = path[1] if len(path) >= 2 else None

    # If next position is blocked (new obstacle or snake body except tail), replan
    blocked_by_body = False
    if next_pos:
        # allow stepping into the tail because it will vacate
        blocked_by_body = next_pos in set(game.snake[:-1])
        blocked_by_obstacle = next_pos in set(game.obstacles)
        if blocked_by_body or blocked_by_obstacle:
            path = plan() or []
            game.planned_path = path
            next_pos = path[1] if len(path) >= 2 else None

    # If we have a valid next_pos, follow it
    if next_pos:
        direction = get_next_direction(game.snake, path)
        if direction:
            game.direction = direction
    else:
        # No planned path found: pick a safe fallback move (prefer continuing current direction)
        x, y = head
        candidates = [
            ('Up', (x, y - game.cell_size)),
            ('Down', (x, y + game.cell_size)),
            ('Left', (x - game.cell_size, y)),
            ('Right', (x + game.cell_size, y)),
        ]
        # prefer current direction first
        candidates.sort(key=lambda t: 0 if t[0] == game.direction else 1)
        safe_dir = None
        for d, (nx, ny) in candidates:
            if not (0 <= nx < game.width and 0 <= ny < game.height):
                continue
            if (nx, ny) in game.obstacles:
                continue
            # allow stepping into tail
            if (nx, ny) in set(game.snake[:-1]):
                continue
            safe_dir = d
            break
        if safe_dir:
            game.direction = safe_dir
        # if no safe move, do nothing and let the game detect collision

    # schedule next planning step
    game.root.after(game.speed, lambda: auto_play(game, algorithm))

# To use auto-play, call auto_play(game, 'astar') or auto_play(game, 'dijkstra') after creating the game instance.
class SnakeGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Complicated Snake Game")
        self.width = 600
        self.height = 400
        self.cell_size = 20
        self.direction = 'Right'
        self.score = 0
        self.level = 1
        self.speed = 100
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.food = None
        self.obstacles = []
        self.create_widgets()
        self.place_food()
        self.place_obstacles()
        self.root.bind("<Key>", self.change_direction)
        self.running = True
        self.update()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black')
        self.canvas.pack()
        self.score_label = tk.Label(self.root, text=f"Score: {self.score}  Level: {self.level}", font=("Arial", 14))
        self.score_label.pack()

    def place_food(self):
        while True:
            x = random.randrange(0, self.width, self.cell_size)
            y = random.randrange(0, self.height, self.cell_size)
            if (x, y) not in self.snake and (x, y) not in self.obstacles:
                self.food = (x, y)
                break

    def place_obstacles(self):
        self.obstacles.clear()
        for _ in range(self.level * 3):
            while True:
                x = random.randrange(0, self.width, self.cell_size)
                y = random.randrange(0, self.height, self.cell_size)
                if (x, y) not in self.snake and (x, y) != self.food and (x, y) not in self.obstacles:
                    self.obstacles.append((x, y))
                    break

    def change_direction(self, event):
        key = event.keysym
        opposites = {'Up':'Down', 'Down':'Up', 'Left':'Right', 'Right':'Left'}
        if key in ['Up', 'Down', 'Left', 'Right'] and opposites[key] != self.direction:
            self.direction = key

    def move_snake(self):
        x, y = self.snake[0]
        if self.direction == 'Up':
            y -= self.cell_size
        elif self.direction == 'Down':
            y += self.cell_size
        elif self.direction == 'Left':
            x -= self.cell_size
        elif self.direction == 'Right':
            x += self.cell_size
        new_head = (x, y)
        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 10
            if self.score % 50 == 0:
                self.level += 1
                self.speed = max(30, self.speed - 10)
                self.place_obstacles()
            self.place_food()
        else:
            self.snake.pop()

    def check_collisions(self):
        head = self.snake[0]
        # Wall collision
        if not (0 <= head[0] < self.width and 0 <= head[1] < self.height):
            return True
        # Self collision
        if head in self.snake[1:]:
            return True
        # Obstacle collision
        if head in self.obstacles:
            return True
        return False

    def draw(self):
        self.canvas.delete("all")
        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            color = 'green' if i == 0 else 'lightgreen'
            self.canvas.create_rectangle(x, y, x+self.cell_size, y+self.cell_size, fill=color)
        # Draw food
        fx, fy = self.food
        self.canvas.create_oval(fx, fy, fx+self.cell_size, fy+self.cell_size, fill='red')
        # Draw obstacles
        for ox, oy in self.obstacles:
            self.canvas.create_rectangle(ox, oy, ox+self.cell_size, oy+self.cell_size, fill='gray')
        self.score_label.config(text=f"Score: {self.score}  Level: {self.level}")

    def update(self):
        if not self.running:
            return
        self.move_snake()
        if self.check_collisions():
            self.running = False
            self.canvas.create_text(self.width//2, self.height//2, text="Game Over", fill="white", font=("Arial", 32))
            return
        self.draw()
        self.root.after(self.speed, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    game = SnakeGame(root)
    # start autoplay before entering the Tk event loop so it runs while the GUI is active
    auto_play(game, 'dijkstra')  # or 'astar'
    root.mainloop()