import tkinter as tk
from tkinter import messagebox
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
        # Limit search to prevent excessive computation
        max_steps = 1000
        steps = 0
        
        open_set = []
        heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
        visited = set()
        g_score = {start: 0}
        
        while open_set and steps < max_steps:
            steps += 1
            _, cost, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path
                
            if current in visited:
                continue
                
            visited.add(current)
            
            for n in neighbors(current, blocked):
                if n in visited:
                    continue
                    
                tentative_g = cost + 1
                
                if n not in g_score or tentative_g < g_score[n]:
                    g_score[n] = tentative_g
                    f_score = tentative_g + heuristic(n, goal)
                    heapq.heappush(open_set, (f_score, tentative_g, n, path + [n]))
        
        # If we couldn't find a path to the goal but explored some nodes,
        # return a path to the node that got closest to the goal
        if visited:
            best_node = None
            best_h = float('inf')
            
            for node in visited:
                h = heuristic(node, goal)
                if h < best_h:
                    best_h = h
                    best_node = node
            
            if best_node and best_node != start:
                # Reconstruct the path to this node
                current = best_node
                path = [current]
                while current in g_score and current != start:
                    for n in neighbors(current, blocked):
                        if n in g_score and g_score[n] == g_score[current] - 1:
                            path.insert(0, n)
                            current = n
                            break
                    else:
                        break
                
                if path[0] == start:
                    return path
        
        return None

    # try to find a path to the food first
    path_to_food = run_astar(food)
    if path_to_food:
        return path_to_food

    # if no path to food, try to find a path to the tail to survive
    tail = snake[-1]
    path_to_tail = run_astar(tail)
    
    # If we can't find a path to tail either, return None and let the fallback logic handle it
    return path_to_tail

def dijkstra_search(snake, food, obstacles, width, height, cell_size):
    def neighbors(pos):
        x, y = pos
        for dx, dy in [(-cell_size,0),(cell_size,0),(0,-cell_size),(0,cell_size)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in obstacles:
                # Allow moving into tail position since it will move
                if (nx, ny) not in snake[:-1]:
                    yield (nx, ny)
    
    # Limit search to prevent excessive computation
    max_steps = 1000
    steps = 0
    
    queue = []
    heapq.heappush(queue, (0, snake[0], [snake[0]]))
    visited = set()
    
    while queue and steps < max_steps:
        steps += 1
        cost, current, path = heapq.heappop(queue)
        
        if current == food:
            return path
            
        if current in visited:
            continue
            
        visited.add(current)
        
        for n in neighbors(current):
            if n not in visited:
                heapq.heappush(queue, (cost+1, n, path+[n]))
    
    # If no path to food is found but we've explored some nodes,
    # return a path to the farthest explored node as a fallback
    if visited and food not in visited:
        # Try to find a path to any safe location
        max_dist = 0
        best_path = None
        
        for node_path in queue:
            if len(node_path[2]) > max_dist:
                max_dist = len(node_path[2])
                best_path = node_path[2]
                
        if best_path:
            return best_path
    
    # Last resort: just return a path with the current position
    # This will trigger the fallback in auto_play
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

    try:
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
    except Exception as e:
        print(f"Error in auto_play: {e}")
        # Make a random safe move as fallback
        try:
            directions = ['Up', 'Down', 'Left', 'Right']
            # Try to avoid the opposite direction
            opposites = {'Up':'Down', 'Down':'Up', 'Left':'Right', 'Right':'Left'}
            if game.direction in opposites:
                directions.remove(opposites[game.direction])
            # Pick a random direction
            game.direction = random.choice(directions)
        except:
            pass  # If even the fallback fails, do nothing

    # schedule next planning step if game is still running
    if game.running:
        game.root.after(game.speed, lambda: auto_play(game, algorithm))

class StartupDialog:
    def __init__(self, root):
        self.root = root
        self.result = None
        
        # Create a new toplevel window
        self.dialog = tk.Toplevel(root)
        self.dialog.title("Snake Game - Choose Mode")
        self.dialog.geometry("400x300")
        self.dialog.resizable(False, False)
        self.dialog.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Make sure dialog stays on top and grabs focus
        self.dialog.transient(root)
        self.dialog.grab_set()
        self.dialog.focus_set()
        
        # Make sure the main window doesn't show
        root.withdraw()
        
        # Center the dialog
        self.center_window()
        
        # Create widgets
        tk.Label(self.dialog, text="Welcome to Snake Game!", font=("Arial", 16, "bold")).pack(pady=20)
        
        # Description frame
        desc_frame = tk.Frame(self.dialog)
        desc_frame.pack(fill="x", padx=20, pady=10)
        tk.Label(desc_frame, text="Choose your play mode:", font=("Arial", 12)).pack(anchor="w")
        
        # Options frame
        options_frame = tk.Frame(self.dialog)
        options_frame.pack(fill="x", padx=30, pady=10)
        
        # Play manually option
        manual_btn = tk.Button(options_frame, text="Play Manually", font=("Arial", 11),
                               command=lambda: self.set_result("manual"), width=20, height=2)
        manual_btn.pack(pady=5)
        
        # Watch A* play option
        astar_btn = tk.Button(options_frame, text="Watch A* Algorithm", font=("Arial", 11),
                            command=lambda: self.set_result("astar"), width=20, height=2)
        astar_btn.pack(pady=5)
        
        # Watch Dijkstra play option
        dijkstra_btn = tk.Button(options_frame, text="Watch Dijkstra Algorithm", font=("Arial", 11),
                               command=lambda: self.set_result("dijkstra"), width=20, height=2)
        dijkstra_btn.pack(pady=5)
        
        # Description text
        tk.Label(self.dialog, text="A* and Dijkstra are pathfinding algorithms\nthat can play the game automatically.",
                 font=("Arial", 10)).pack(pady=10)
        
        root.wait_window(self.dialog)
    
    def center_window(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
    
    def set_result(self, result):
        self.result = result
        self.dialog.destroy()
        self.root.deiconify()  # Show the main window after dialog is closed
        
    def on_closing(self):
        self.result = "manual"  # Default to manual if closed
        self.dialog.destroy()
        self.root.deiconify()  # Show the main window after dialog is closed

class SnakeGame:
    def __init__(self, root, play_mode='manual'):
        self.root = root
        self.root.title("Snake Game")
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
        self.play_mode = play_mode  # 'manual', 'astar', or 'dijkstra'
        
        self.create_widgets()
        self.place_food()
        self.place_obstacles()
        
        # Only bind keys for manual play
        if self.play_mode == 'manual':
            self.root.bind("<Key>", self.change_direction)
            
        self.running = True
        self.update()

    def create_widgets(self):
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black')
        self.canvas.pack()
        
        # Create frame for info display
        info_frame = tk.Frame(self.root)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        # Score and level display
        self.score_label = tk.Label(info_frame, text=f"Score: {self.score}  Level: {self.level}", font=("Arial", 14))
        self.score_label.pack(side=tk.LEFT)
        
        # Mode display
        mode_text = "Manual Play"
        if self.play_mode == 'astar':
            mode_text = "AI Mode: A* Search"
        elif self.play_mode == 'dijkstra':
            mode_text = "AI Mode: Dijkstra's Algorithm"
            
        self.mode_label = tk.Label(info_frame, text=mode_text, font=("Arial", 12), fg="blue")
        self.mode_label.pack(side=tk.RIGHT)

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
            
            # Display final score
            self.canvas.create_text(self.width//2, self.height//2 + 50, 
                                   text=f"Final Score: {self.score}", 
                                   fill="white", font=("Arial", 18))
            
            # Add restart button
            restart_button = tk.Button(self.canvas, text="Play Again", font=("Arial", 12),
                                      command=self.restart_game, bg="green", fg="white")
            restart_button_window = self.canvas.create_window(self.width//2, self.height//2 + 100, 
                                                            window=restart_button)
            return
        self.draw()
        self.root.after(self.speed, self.update)
        
    def restart_game(self):
        # Reset game state
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = 'Right'
        self.score = 0
        self.level = 1
        self.speed = 100
        self.obstacles = []
        
        # Clear and recreate game elements
        self.canvas.delete("all")
        self.place_food()
        self.place_obstacles()
        
        # Update display
        self.score_label.config(text=f"Score: {self.score}  Level: {self.level}")
        
        # Restart the game loop
        self.running = True
        
        # If we're in auto mode, restart the algorithm
        if self.play_mode != 'manual':
            # Clear any previous planned path
            if hasattr(self, 'planned_path'):
                self.planned_path = []
            auto_play(self, self.play_mode)
        
        # Start the update loop again
        self.update()

if __name__ == "__main__":
    try:
        # Create the root window
        root = tk.Tk()
        root.title("Snake Game")
        
        # Force root to initialize but stay hidden
        root.withdraw()
        
        # Show startup dialog to select game mode
        dialog = StartupDialog(root)
        play_mode = dialog.result
        
        # Default to manual if dialog was closed or no selection was made
        if not play_mode:
            play_mode = "manual"
            
        # Create the game with the selected mode
        game = SnakeGame(root, play_mode=play_mode)
        
        # Start autoplay if an algorithm was selected
        if play_mode != "manual":
            auto_play(game, play_mode)
            
        # Start the main event loop
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {e}")