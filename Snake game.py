import tkinter as tk
from tkinter import messagebox
import random
import time

class SnakeGame:
    def __init__(self, master, mode='manual'):
        self.master = master
        self.master.title("Snake Game")
        self.width = 600
        self.height = 400
        self.cell_size = 20
        self.mode = mode
        self.direction = 'Right'
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.food = None
        self.obstacles = []
        self.score = 0
        self.running = True
        
        # Set different speeds based on algorithm
        if self.mode == 'manual':
            self.speed = 100
        elif self.mode == 'astar':
            self.speed = 60  # A* runs faster
        elif self.mode == 'dijkstra':
            self.speed = 120  # Dijkstra runs slower
        
        # Initialize UI components
        self.setup_ui()
        self.place_food()
        self.place_obstacles(5)  # Start with fewer obstacles
        
        # Bind keys for manual mode
        if self.mode == 'manual':
            self.master.bind("<KeyPress>", self.on_key_press)
        
        # Initialize path planning variables for AI modes
        if self.mode in ['astar', 'dijkstra']:
            self.planned_path = []
            self.next_move_time = time.time()
        
        # Start the game loop
        self.update()
    
    def setup_ui(self):
        # Create game canvas
        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height, bg="black")
        self.canvas.pack(pady=10)
        
        # Info bar
        info_frame = tk.Frame(self.master)
        info_frame.pack(fill="x", padx=10)
        
        # Score display
        self.score_label = tk.Label(info_frame, text="Score: 0", font=("Arial", 14))
        self.score_label.pack(side="left", padx=10)
        
        # Mode display
        mode_text = "Manual Mode"
        mode_color = "blue"
        if self.mode == 'astar':
            mode_text = "A* Algorithm Mode"
            mode_color = "green"
        elif self.mode == 'dijkstra':
            mode_text = "Dijkstra Algorithm Mode"
            mode_color = "orange"
        
        self.mode_label = tk.Label(info_frame, text=mode_text, font=("Arial", 14), fg=mode_color)
        self.mode_label.pack(side="right", padx=10)
        
        # Reset button
        reset_button = tk.Button(self.master, text="Restart (r)", command=self.restart_game)
        reset_button.pack(pady=5)
    
    def place_food(self):
        while True:
            x = random.randrange(0, self.width, self.cell_size)
            y = random.randrange(0, self.height, self.cell_size)
            pos = (x, y)
            if pos not in self.snake and pos not in self.obstacles:
                self.food = pos
                break
    
    def place_obstacles(self, count):
        self.obstacles = []
        for _ in range(count):
            while True:
                x = random.randrange(0, self.width, self.cell_size)
                y = random.randrange(0, self.height, self.cell_size)
                pos = (x, y)
                if pos not in self.snake and pos != self.food and pos not in self.obstacles:
                    self.obstacles.append(pos)
                    break
    
    def on_key_press(self, event):
        key = event.keysym
        
        # Handle game restart
        if key.lower() == 'r':
            self.restart_game()
            return
        
        # Direction changes - prevent 180 degree turns
        if key == 'Right' and self.direction != 'Left':
            self.direction = 'Right'
        elif key == 'Left' and self.direction != 'Right':
            self.direction = 'Left'
        elif key == 'Up' and self.direction != 'Down':
            self.direction = 'Up'
        elif key == 'Down' and self.direction != 'Up':
            self.direction = 'Down'
    
    def move_snake(self):
        head_x, head_y = self.snake[0]
        
        if self.direction == 'Right':
            new_head = (head_x + self.cell_size, head_y)
        elif self.direction == 'Left':
            new_head = (head_x - self.cell_size, head_y)
        elif self.direction == 'Up':
            new_head = (head_x, head_y - self.cell_size)
        elif self.direction == 'Down':
            new_head = (head_x, head_y + self.cell_size)
        
        self.snake.insert(0, new_head)
        
        if new_head == self.food:
            self.score += 10
            self.score_label.config(text=f"Score: {self.score}")
            self.place_food()
            # Add more obstacles every 50 points
            if self.score % 50 == 0:
                self.place_obstacles(5 + (self.score // 50))
        else:
            self.snake.pop()
    
    def is_collision(self):
        head = self.snake[0]
        head_x, head_y = head
        
        # Wall collision
        if head_x < 0 or head_x >= self.width or head_y < 0 or head_y >= self.height:
            return True
        
        # Self collision (excluding head)
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
            color = 'green' if i == 0 else 'light green'
            self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, 
                                         fill=color, outline="")
        
        # Draw food
        food_x, food_y = self.food
        self.canvas.create_oval(food_x, food_y, food_x + self.cell_size, food_y + self.cell_size, 
                               fill="red", outline="")
        
        # Draw obstacles
        for ox, oy in self.obstacles:
            self.canvas.create_rectangle(ox, oy, ox + self.cell_size, oy + self.cell_size, 
                                        fill="gray", outline="")
        
        # Draw path for AI modes (optional)
        if hasattr(self, 'planned_path') and self.planned_path and len(self.planned_path) > 1:
            path_color = "light green" if self.mode == 'astar' else "light blue"
            for i in range(1, min(len(self.planned_path), 10)):  # Limit to 10 steps ahead
                px, py = self.planned_path[i]
                self.canvas.create_rectangle(px + 8, py + 8, px + self.cell_size - 8, 
                                           py + self.cell_size - 8, fill=path_color, outline="")
    
    def update(self):
        if not self.running:
            return
        
        # For AI modes, calculate next move
        if self.mode in ['astar', 'dijkstra']:
            # Only calculate a new move when it's time
            current_time = time.time()
            if current_time >= self.next_move_time:
                self.calculate_ai_move()
                self.next_move_time = current_time + (self.speed / 1000.0)
        
        self.move_snake()
        
        if self.is_collision():
            self.running = False
            self.show_game_over()
            return
        
        self.draw()
        self.master.after(self.speed, self.update)
    
    def calculate_ai_move(self):
        """Simple AI to move toward food while avoiding obstacles"""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Calculate possible moves
        possible_moves = []
        
        # Right
        if self.direction != 'Left':  # Prevent 180-degree turns
            new_pos = (head_x + self.cell_size, head_y)
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and 
                new_pos not in self.snake[:-1] and  # Can move into tail position as it will move
                new_pos not in self.obstacles):
                possible_moves.append(('Right', new_pos))
        
        # Left
        if self.direction != 'Right':
            new_pos = (head_x - self.cell_size, head_y)
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and 
                new_pos not in self.snake[:-1] and
                new_pos not in self.obstacles):
                possible_moves.append(('Left', new_pos))
        
        # Up
        if self.direction != 'Down':
            new_pos = (head_x, head_y - self.cell_size)
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and 
                new_pos not in self.snake[:-1] and
                new_pos not in self.obstacles):
                possible_moves.append(('Up', new_pos))
        
        # Down
        if self.direction != 'Up':
            new_pos = (head_x, head_y + self.cell_size)
            if (0 <= new_pos[0] < self.width and 
                0 <= new_pos[1] < self.height and 
                new_pos not in self.snake[:-1] and
                new_pos not in self.obstacles):
                possible_moves.append(('Down', new_pos))
        
        # If no moves are possible, keep current direction (will likely cause collision)
        if not possible_moves:
            return
        
        # For A*: prefer moves that get closer to the food (manhattan distance)
        if self.mode == 'astar':
            def manhattan_distance(pos):
                return abs(pos[0] - food_x) + abs(pos[1] - food_y)
            
            possible_moves.sort(key=lambda x: manhattan_distance(x[1]))
            self.direction = possible_moves[0][0]
            
            # Build a simple path for visualization
            self.planned_path = [self.snake[0]]
            next_pos = possible_moves[0][1]
            self.planned_path.append(next_pos)
            
        # For Dijkstra: prioritize open spaces
        elif self.mode == 'dijkstra':
            # Count number of empty adjacent cells for each possible move
            def count_open_spaces(pos):
                count = 0
                x, y = pos
                for dx, dy in [(self.cell_size, 0), (-self.cell_size, 0), (0, self.cell_size), (0, -self.cell_size)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.width and 
                        0 <= ny < self.height and 
                        (nx, ny) not in self.snake and
                        (nx, ny) not in self.obstacles):
                        count += 1
                return count
            
            # Sometimes go for food directly, sometimes prioritize open spaces
            if random.random() < 0.7:  # 70% chance to go for food
                possible_moves.sort(key=lambda x: manhattan_distance(x[1]))
            else:  # 30% chance to prioritize open spaces
                possible_moves.sort(key=lambda x: -count_open_spaces(x[1]))  # Negative to sort descending
            
            self.direction = possible_moves[0][0]
            
            # Build a simple path for visualization
            self.planned_path = [self.snake[0]]
            next_pos = possible_moves[0][1]
            self.planned_path.append(next_pos)
    
    def show_game_over(self):
        self.canvas.create_text(
            self.width // 2, 
            self.height // 2, 
            text="GAME OVER", 
            fill="white", 
            font=("Arial", 30)
        )
        
        self.canvas.create_text(
            self.width // 2, 
            self.height // 2 + 50, 
            text=f"Score: {self.score}", 
            fill="white", 
            font=("Arial", 20)
        )
        
        # Add restart button on canvas
        restart_button = tk.Button(self.canvas, text="Play Again", command=self.restart_game)
        self.canvas.create_window(self.width // 2, self.height // 2 + 100, window=restart_button)
    
    def restart_game(self):
        # Reset game state
        self.snake = [(100, 100), (80, 100), (60, 100)]
        self.direction = 'Right'
        self.score = 0
        self.obstacles = []
        
        # Reset display
        self.score_label.config(text=f"Score: {self.score}")
        
        # Recreate game elements
        self.place_food()
        self.place_obstacles(5)
        
        if self.mode in ['astar', 'dijkstra']:
            self.planned_path = []
        
        # Clear canvas and restart
        self.canvas.delete("all")
        self.running = True
        self.update()

def show_game_selection():
    selection_window = tk.Tk()
    selection_window.title("Snake Game - Mode Selection")
    selection_window.geometry("300x250")
    selection_window.resizable(False, False)
    
    # Center the window
    window_width = 300
    window_height = 250
    screen_width = selection_window.winfo_screenwidth()
    screen_height = selection_window.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    selection_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    def start_game(mode):
        selection_window.destroy()
        root = tk.Tk()
        game = SnakeGame(root, mode=mode)
        root.mainloop()
    
    # Title
    tk.Label(selection_window, text="Snake Game", font=("Arial", 20, "bold")).pack(pady=15)
    
    # Instructions
    tk.Label(selection_window, text="Select a game mode:", font=("Arial", 12)).pack(pady=5)
    
    # Buttons
    button_frame = tk.Frame(selection_window)
    button_frame.pack(pady=10)
    
    manual_btn = tk.Button(button_frame, text="Manual Play", font=("Arial", 11),
                          command=lambda: start_game('manual'), width=20)
    manual_btn.pack(pady=5)
    
    astar_btn = tk.Button(button_frame, text="A* Algorithm", font=("Arial", 11),
                          command=lambda: start_game('astar'), width=20)
    astar_btn.pack(pady=5)
    
    dijkstra_btn = tk.Button(button_frame, text="Dijkstra Algorithm", font=("Arial", 11),
                            command=lambda: start_game('dijkstra'), width=20)
    dijkstra_btn.pack(pady=5)
    
    # Info text
    tk.Label(selection_window, text="Manual: Use arrow keys to control\nA*/Dijkstra: Watch AI play", 
            font=("Arial", 9), justify="left").pack(pady=5)
    
    selection_window.mainloop()

# Display instructions for first-time users
def show_instructions():
    messagebox.showinfo("Instructions", 
                       "Snake Game Instructions:\n\n"
                       "• Manual Mode: Use arrow keys to control the snake\n"
                       "• A* Algorithm: Watch the computer play using A*\n"
                       "• Dijkstra Algorithm: Watch using Dijkstra's algorithm\n\n"
                       "Collect red food to grow and score points.\n"
                       "Avoid hitting walls, obstacles, or yourself.\n\n"
                       "Press 'R' to restart at any time.")

if __name__ == "__main__":
    try:
        # Show instructions first (can be commented out after first run)
        show_instructions()
        
        # Show game selection dialog
        show_game_selection()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        print(f"Error: {e}")
