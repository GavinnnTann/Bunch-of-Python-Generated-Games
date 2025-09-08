import tkinter as tk
import random

class Tetris:
    def __init__(self, master):
        self.master = master
        self.master.title("Tetris")
        self.width = 10
        self.height = 20
        self.cell_size = 30
        
        # Create main frame for all widgets
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(pady=10)
        
        # Create game area
        self.canvas = tk.Canvas(self.main_frame, width=self.width*self.cell_size, height=self.height*self.cell_size, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        # Create sidebar frame for score and next piece
        self.sidebar = tk.Frame(self.main_frame)
        self.sidebar.pack(side=tk.LEFT, padx=10)
        
        # Create score display
        self.score_frame = tk.Frame(self.sidebar, height=100)
        self.score_frame.pack(fill=tk.X, pady=20)
        tk.Label(self.score_frame, text="SCORE", font=("Arial", 16)).pack()
        self.score_var = tk.StringVar(value="0")
        tk.Label(self.score_frame, textvariable=self.score_var, font=("Arial", 24)).pack()
        
        # Create next piece preview
        tk.Label(self.sidebar, text="NEXT", font=("Arial", 16)).pack(pady=(20, 0))
        self.preview_canvas = tk.Canvas(self.sidebar, width=4*self.cell_size, height=4*self.cell_size, bg='black')
        self.preview_canvas.pack(pady=5)
        
        # Add instructions
        instructions = tk.Frame(self.sidebar)
        instructions.pack(pady=20, fill=tk.X)
        tk.Label(instructions, text="Controls:", font=("Arial", 12)).pack(anchor=tk.W)
        tk.Label(instructions, text="Arrows: Move/Rotate", font=("Arial", 10)).pack(anchor=tk.W)
        tk.Label(instructions, text="Space: Drop", font=("Arial", 10)).pack(anchor=tk.W)
        tk.Label(instructions, text="R: Reset Game", font=("Arial", 10)).pack(anchor=tk.W)
        
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.shapes = [
            [[1, 1, 1, 1]],  # I
            [[1, 1], [1, 1]],  # O
            [[0, 1, 0], [1, 1, 1]],  # T
            [[1, 0, 0], [1, 1, 1]],  # J
            [[0, 0, 1], [1, 1, 1]],  # L
            [[1, 1, 0], [0, 1, 1]],  # S
            [[0, 1, 1], [1, 1, 0]]   # Z
        ]
        self.colors = ['cyan', 'yellow', 'purple', 'blue', 'orange', 'green', 'red']
        self.score = 0
        
        # Generate first and next pieces
        self.next_shape_id = random.randint(0, len(self.shapes)-1)
        self.reset_piece()
        self.game_over = False
        self.master.bind("<Key>", self.key_press)
        self.update()

    def reset_piece(self):
        # Use the next shape that was prepared
        if hasattr(self, 'next_shape_id'):
            self.shape_id = self.next_shape_id
        else:
            self.shape_id = random.randint(0, len(self.shapes)-1)
        
        # Prepare the next shape
        self.next_shape_id = random.randint(0, len(self.shapes)-1)
        self.shape = self.shapes[self.shape_id]
        self.color = self.colors[self.shape_id]
        self.x = self.width // 2 - len(self.shape[0]) // 2
        self.y = 0
        
        # Draw the next piece preview
        if hasattr(self, 'preview_canvas'):
            self.draw_preview()

    def rotate(self):
        self.shape = [list(row) for row in zip(*self.shape[::-1])]
        if self.collide(self.x, self.y, self.shape):
            self.shape = [list(row) for row in zip(*self.shape)][::-1]  # undo

    def collide(self, x, y, shape):
        for i, row in enumerate(shape):
            for j, cell in enumerate(row):
                if cell:
                    if (x+j < 0 or x+j >= self.width or y+i >= self.height or
                        (y+i >= 0 and self.board[y+i][x+j])):
                        return True
        return False

    def freeze(self):
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell and self.y+i >= 0:
                    self.board[self.y+i][self.x+j] = self.color
        
        # Add points for placing a piece
        self.score += 5
        self.score_var.set(str(self.score))
        
        self.clear_lines()
        self.reset_piece()
        if self.collide(self.x, self.y, self.shape):
            self.game_over = True

    def clear_lines(self):
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        lines_cleared = self.height - len(new_board)
        for _ in range(lines_cleared):
            new_board.insert(0, [0 for _ in range(self.width)])
        self.board = new_board
        
        # Update score based on lines cleared
        if lines_cleared > 0:
            # Score more points for clearing multiple lines at once
            points = {1: 100, 2: 300, 3: 500, 4: 800}
            self.score += points.get(lines_cleared, 100 * lines_cleared)
            self.score_var.set(str(self.score))

    def move(self, dx, dy):
        if not self.collide(self.x+dx, self.y+dy, self.shape):
            self.x += dx
            self.y += dy
        elif dy == 1:
            self.freeze()

    def key_press(self, event):
        if self.game_over:
            # Allow reset even when game is over
            if event.keysym.lower() == 'r':
                self.reset_game()
            return
            
        if event.keysym == 'Left':
            self.move(-1, 0)
        elif event.keysym == 'Right':
            self.move(1, 0)
        elif event.keysym == 'Down':
            self.move(0, 1)
        elif event.keysym == 'Up':
            self.rotate()
        elif event.keysym == 'space':
            # Hard drop - move piece all the way down
            while not self.collide(self.x, self.y+1, self.shape):
                self.y += 1
            self.freeze()
            # Add points for hard drop
            self.score += 10
            self.score_var.set(str(self.score))
        elif event.keysym.lower() == 'r':
            self.reset_game()
        self.draw()
        
    def reset_game(self):
        # Clear the board
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        # Reset score
        self.score = 0
        self.score_var.set("0")
        # Generate new pieces
        self.next_shape_id = random.randint(0, len(self.shapes)-1)
        self.reset_piece()
        self.game_over = False
        # Restart game loop if needed
        if not hasattr(self, '_update_after_id'):
            self.update()

    def update(self):
        if not self.game_over:
            self.move(0, 1)
            self.draw()
            # Store the after ID so we can cancel it if needed
            self._update_after_id = self.master.after(400, self.update)
        else:
            # Game over message
            self.canvas.create_text(self.width*self.cell_size//2, self.height*self.cell_size//2,
                                    text="GAME OVER", fill="white", font=("Arial", 24))
            self.canvas.create_text(self.width*self.cell_size//2, self.height*self.cell_size//2 + 40,
                                    text="Press 'R' to restart", fill="white", font=("Arial", 16))

    def draw(self):
        self.canvas.delete("all")
        
        # Draw grid lines
        for x in range(self.width + 1):
            x_pos = x * self.cell_size
            self.canvas.create_line(x_pos, 0, x_pos, self.height * self.cell_size, fill="gray", width=1)
        for y in range(self.height + 1):
            y_pos = y * self.cell_size
            self.canvas.create_line(0, y_pos, self.width * self.cell_size, y_pos, fill="gray", width=1)
            
        # Draw board
        for y in range(self.height):
            for x in range(self.width):
                color = self.board[y][x]
                if color:
                    self.draw_cell(self.canvas, x, y, color)
        
        # Draw current piece
        for i, row in enumerate(self.shape):
            for j, cell in enumerate(row):
                if cell and self.y+i >= 0:
                    self.draw_cell(self.canvas, self.x+j, self.y+i, self.color)
                    
        # Draw shadow (ghost piece) to show where piece will land
        if not self.game_over:
            shadow_y = self.y
            while not self.collide(self.x, shadow_y + 1, self.shape):
                shadow_y += 1
                
            if shadow_y > self.y:
                for i, row in enumerate(self.shape):
                    for j, cell in enumerate(row):
                        if cell and shadow_y+i >= 0:
                            self.draw_shadow_cell(self.x+j, shadow_y+i)

    def draw_cell(self, canvas, x, y, color):
        x0 = x * self.cell_size + 1
        y0 = y * self.cell_size + 1
        x1 = x0 + self.cell_size - 2
        y1 = y0 + self.cell_size - 2
        
        # Create 3D effect with lighter and darker borders
        light_color = self.lighten_color(color)
        dark_color = self.darken_color(color)
        
        # Main rectangle
        canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        
        # Highlight (top-left)
        canvas.create_line(x0, y0, x1, y0, fill=light_color, width=2)
        canvas.create_line(x0, y0, x0, y1, fill=light_color, width=2)
        
        # Shadow (bottom-right)
        canvas.create_line(x0, y1, x1, y1, fill=dark_color, width=2)
        canvas.create_line(x1, y0, x1, y1, fill=dark_color, width=2)
    
    def draw_shadow_cell(self, x, y):
        # Draw a ghost version of a cell to show where piece will land
        x0 = x * self.cell_size + 1
        y0 = y * self.cell_size + 1
        x1 = x0 + self.cell_size - 2
        y1 = y0 + self.cell_size - 2
        
        # Draw just the outline
        self.canvas.create_rectangle(x0, y0, x1, y1, outline="white", width=1)
    
    def lighten_color(self, color):
        # Simple lightening for common colors
        color_map = {
            'cyan': '#AAFFFF', 'yellow': '#FFFFAA', 'purple': '#FFAAFF',
            'blue': '#AAAAFF', 'orange': '#FFDDAA', 'green': '#AAFFAA', 'red': '#FFAAAA'
        }
        return color_map.get(color, color)
    
    def darken_color(self, color):
        # Simple darkening for common colors
        color_map = {
            'cyan': '#008888', 'yellow': '#888800', 'purple': '#880088',
            'blue': '#000088', 'orange': '#884400', 'green': '#008800', 'red': '#880000'
        }
        return color_map.get(color, color)
    
    def draw_preview(self):
        # Clear the preview canvas
        self.preview_canvas.delete("all")
        
        # Draw the next piece
        next_shape = self.shapes[self.next_shape_id]
        next_color = self.colors[self.next_shape_id]
        
        # Calculate centering
        width = len(next_shape[0])
        height = len(next_shape)
        start_x = (4 - width) // 2
        start_y = (4 - height) // 2
        
        # Draw the next piece in the preview canvas
        for i, row in enumerate(next_shape):
            for j, cell in enumerate(row):
                if cell:
                    self.draw_cell(self.preview_canvas, start_x + j, start_y + i, next_color)

if __name__ == "__main__":
    root = tk.Tk()
    game = Tetris(root)
    root.mainloop()