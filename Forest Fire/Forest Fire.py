import tkinter as tk
import random
from tkinter import messagebox
from collections import deque
from tkinter.simpledialog import askstring
from datetime import datetime
import json
import os

# Path to the local highscores file
# Use os.path.dirname(__file__) to get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIGHSCORES_FILE = os.path.join(SCRIPT_DIR, "forest_fire_highscores.json")

# Initialize highscores
highscores = []

# Load highscores if file exists
def load_highscores():
    global highscores
    try:
        if os.path.exists(HIGHSCORES_FILE):
            with open(HIGHSCORES_FILE, 'r') as file:
                highscores = json.load(file)
    except Exception as e:
        print(f"Error loading highscores: {e}")
        highscores = []

# Save highscores to file
def save_highscores():
    try:
        with open(HIGHSCORES_FILE, 'w') as file:
            json.dump(highscores, file)
    except Exception as e:
        print(f"Error saving highscores: {e}")

# Load highscores at startup
load_highscores()

start_time = datetime.now()

class ForestFireGame:
    def __init__(self, root, difficulty, custom_size=None):
        self.root = root
        self.player_name = player_name
        self.difficulty = difficulty
        self.size = 11  # Default difficulty will be normal at 11

        if difficulty == 'easy':
            self.size = 13
        elif difficulty == 'normal':
            self.size = 11
        elif difficulty == 'hard':
            self.size = 9
        elif difficulty == 'impossible':
            self.size = 8
        elif difficulty == 'custom':
            self.size = custom_size or self.get_custom_size()

        self.grid = [[0] * self.size for _ in range(self.size)]

        self.buttons = [[None] * self.size for _ in range(self.size)]
        self.create_board()
        self.show_instructions()

        # Create a reset button
        self.create_reset_button()
        self.create_leaderboard()

        # Create a scoreboard label and defining variable to count moves
        self.moves_count = 0
        self.create_scoreboard()


    def create_leaderboard(self):
        global highscores
        
        if not highscores:
            # Displays the following when no one played the game yet
            self.highscore_label = tk.Label(
            self.root, text="Highscores", font=("Bell Gothic Std Black",10))
            self.highscore_label.grid(row=1, columnspan=3)

            self.score_label = tk.Label(
            self.root, text="No scores yet!")
            self.score_label.grid(row=0, columnspan=self.size)
            self.score_label = tk.Label(
            self.root, text="Complete the game to get first")
            self.score_label.grid(row=1, columnspan=self.size)
        else:
            # Displays the highscore from the local storage
            self.highscore_label = tk.Label(
            self.root, text="Highscores", font=("Bell Gothic Std Black",15))
            self.highscore_label.grid(row=1, columnspan=3)

            no_of_scores = len(highscores)

            if no_of_scores >= 3:
                self.display_leaderboard(3, highscores)
            else:
                self.display_leaderboard(no_of_scores, highscores)


    # Displays the top 3 highscores, sorted by the number of moves,
    # followed by the time taken to complete the game
    def display_leaderboard(self, count, all_highscores):
        # Sort highscores by number of moves followed by time taken to complete the game
        sorted_highscores = sorted(all_highscores, key=lambda x: (x["moves"], x["time"]))

        # Display the top 3 highscores
        for i in range(min(count, len(sorted_highscores))):
            self.firstname_label = tk.Label(
            self.root, text=f"{i+1}. {sorted_highscores[i]['name']}:{sorted_highscores[i]['moves']}      Time: {sorted_highscores[i]['time']}s"
            )
            self.firstname_label.grid(row=i, column=3, columnspan=self.size-3)


    # Creating functionality of reset button
    def create_reset_button(self):
        self.reset_button = tk.Button(
            self.root, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=self.size + 3, columnspan=self.size)


    def create_scoreboard(self):
        self.score_label = tk.Label(
            self.root,  text=f"Moves: {self.moves_count}")
        self.score_label.grid(row=self.size + 4, columnspan=self.size)


    def update_scoreboard(self):
        self.score_label.config(text=f"Moves: {self.moves_count}")


    def show_instructions(self):
        instructions = (
            "Welcome to the Forest Fire Challenge!\n\n"
            "Your goal is to prevent the forest fire from spreading.\n\n"
            "Blue tiles represent water barriers.\n"
            "The red tile represents the dangerous fire.\n\n"
            "If the forest fire reaches any of the brown tiles, you have failed to put out the fire.\n\n"
            "To win the game, click on the blank tiles to create water barriers and try to surround the fire.\n\n"
            "Be strategic, as the fire will try to escape.\n\n"
            "Good luck and protect the forest!"
        )
        messagebox.showinfo("Forest Fire Challenge Instructions", instructions)

        # Place the fire at a random position near the middle of the grid
        fire_row = self.size // 2
        fire_col = self.size // 2
        self.fire_position = (fire_row, fire_col)
        self.grid[self.fire_position[0]][self.fire_position[1]] = 1
        self.update_board()


    def create_board(self):
        for i in range(self.size):
            for j in range(self.size):
                button = tk.Button(self.root, width=4, height=2,
                                   command=lambda i=i, j=j: self.on_click(i, j))
                button.grid(row=i+3, column=j)
                self.buttons[i][j] = button


    def update_board(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                        # Brown for the forest border
                        self.buttons[i][j].config(bg='#8B4513')
                    else:
                        # Light green for unmarked spots
                        self.buttons[i][j].config(bg='#90EE90')
                elif self.grid[i][j] == 1:  # Fire position
                    self.buttons[i][j].config(bg='#FF0000')  # Red for the fire
                elif self.grid[i][j] == 2:  # Positive action (marked spot)
                    # Blue for water barriers
                    self.buttons[i][j].config(bg='#0000FF')
                elif self.grid[i][j] == 3:  # Burnt woods
                    self.buttons[i][j].config(
                        bg='#FF0000')  # Red for burnt woods
                elif self.grid[i][j] == 4:  # Yellow center of the flower
                    self.buttons[i][j].config(bg='#FFFF00')  # Yellow for the flower center
                elif self.grid[i][j] == 5:  # Pink petals of the flower
                    self.buttons[i][j].config(bg='#FFC0CB')  # Pink for the flower petals


    def on_click(self, i, j):
        if self.grid[i][j] == 0 and not (i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1):
            # Player takes a positive action (marks the spot with water)
            self.grid[i][j] = 2
            self.move_fire()
            self.moves_count += 1
            self.update_scoreboard()
            self.update_board()
            self.check_game_state()


    def get_neighbors(self, x, y):
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = [(x + dx, y + dy) for dx, dy in moves]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.size and 0 <= ny < self.size]


    def bfs(self, start):
        visited = set()
        queue = deque([(start, 0)])

        while queue:
            (x, y), distance = queue.popleft()

            if (x == 0 or x == self.size - 1 or y == 0 or y == self.size - 1) and self.grid[x][y] == 0:
                return distance

            for nx, ny in self.get_neighbors(x, y):
                if (nx, ny) not in visited and self.grid[nx][ny] == 0:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), distance + 1))

        return float('inf')


    def move_fire(self):
        # Find the valid neighbors (unmarked spots) of the fire
        valid_neighbors = [(i, j) for i, j in self.get_neighbors(
            *self.fire_position) if self.grid[i][j] == 0]

        if valid_neighbors:
            # Move the fire towards the closest unmarked spot using BFS
            best_move = min(valid_neighbors, key=lambda pos: self.bfs(pos))
            self.grid[self.fire_position[0]][self.fire_position[1]] = 0
            self.fire_position = best_move
            self.grid[self.fire_position[0]][self.fire_position[1]] = 1


    def check_game_state(self):
        # Check if the fire is surrounded by water barriers
        surrounded = all(self.grid[i][j] == 2 for i,
                         j in self.get_neighbors(*self.fire_position))

        if surrounded:
            self.show_game_result(
                f"Congratulations, {self.player_name}! You've successfully contained the forest fire. \n\nThe flowers are blooming because of you!! \n", self.prompt_restart)
        elif (self.fire_position[0] == 0 or self.fire_position[0] == self.size - 1 or
              self.fire_position[1] == 0 or self.fire_position[1] == self.size - 1):
            self.show_game_result(
                "Oh no! The forest fire has escaped and is spreading!", self.prompt_restart)


    def show_game_result(self, message, callback):
        if "successfully contained" in message:
            messagebox.showinfo(
                "Game Over", f"{message}\nYou've saved the forest!")

            end_time = datetime.now()
            time_taken = end_time - start_time
            seconds = float(str(time_taken).split(':')[-1])

            # Add score to highscores
            global highscores
            highscores.append({
                "name": self.player_name,
                "moves": self.moves_count,
                "time": seconds
            })
            
            # Save highscores to file
            save_highscores()
            
            self.animate_win(callback)
            self.create_leaderboard()
            
        elif "forest fire has escaped" in message:
            # Check if the message contains the specific prompt
            if "The forest fire has escaped and is spreading!" in message:
                # Add burning animation with a callback to prompt restart after animation
                messagebox.showinfo(
                    "Game Over", f"{message}\nThe forest has been completely destroyed.")
                self.animate_loss(callback)

    def animate_win(self, callback):
        
        self.reset_game_screen() #reset the screen
        # Get all positions in the grid
        all_positions = [(i, j) for i in range(self.size) for j in range(self.size)]

        # Ensure the number of flowers does not exceed the size of the game
        num_flowers = min(len(all_positions), self.size)

        # Shuffle the list of all positions
        random.shuffle(all_positions)

        # Schedule the animation with a callback
        self.animate_win_helper(all_positions, num_flowers, callback)

    def animate_win_helper(self, positions, num_flowers, callback):
        if num_flowers <= 0:
            # Execute the callback function after the animation completes
            callback()
            return

        # Select a position randomly and create a flower
        flower_position = positions.pop(0)
        self.create_flower(flower_position)

        # Update the board
        self.update_board()

        # Schedule the next update
        self.root.after(100, self.animate_win_helper, positions, num_flowers - 1, callback)

    def create_flower(self, position):
        i, j = position

        # Skip water barrier positions
        if self.grid[i][j] == 2:
            return

        # Set the center of the flower to yellow
        self.grid[i][j] = 4

        # Set the surrounding petals to pink
        petals_positions = self.get_neighbors(i, j)
        for petal_pos in petals_positions:
            pi, pj = petal_pos
            # Overwrite any existing block, including blue water barriers
            self.grid[pi][pj] = 5
            
    def animate_loss(self, callback):
        # Get the list of unmarked positions
        unmarked_positions = [(i, j) for i in range(self.size)
                              for j in range(self.size) if self.grid[i][j] == 0]

        # Schedule the animation with a callback
        self.animate_loss_helper(unmarked_positions, callback)


    def animate_loss_helper(self, positions, callback):
        if not positions:
            # Execute the callback function after the animation completes
            callback()
            return

        # Update multiple positions at a time for a more outward-spreading effect
        num_updates = min(5, len(positions))
        updated_positions = positions[:num_updates]

        for i, j in updated_positions:
            self.grid[i][j] = 3  # Set cell to burnt woods

        self.update_board()

        # Schedule the next update
        self.root.after(100, self.animate_loss_helper,
                        positions[num_updates:], callback)


    def prompt_restart(self):
        # Prompt the user to restart the game
        result = messagebox.askquestion(
            "Game Over", "Do you want to play again?")
        if result == 'yes':
            ask_again = messagebox.askquestion(
            "Restarting game", "Do you want to change the difficulty level?")
            # gives the user the option to change difficulty level
            if ask_again == 'yes':
                self.root.destroy()
                if __name__ == "__main__":

                    # Prompt the user for difficulty level
                    difficulty_level = askstring(
                        "Difficulty", "Choose difficulty: easy, normal, or hard")
                    if difficulty_level and difficulty_level.lower() in ['easy', 'normal', 'hard']:
                        root_window = tk.Tk()
                        root_window.title("Forest Fire Challenge")

                        game = ForestFireGame(root_window, difficulty_level.lower())

                        root_window.mainloop()
                    else:
                        messagebox.showinfo(
                            "Invalid Input", "Please enter a valid difficulty level (easy, normal, or hard).")

            else:
                self.reset_game()

        else:
            self.root.destroy()
            
    def reset_game_screen(self):
        # Reset the game by reinitializing the grid and fire position
        self.grid = [[0] * self.size for _ in range(self.size)]


    def reset_game(self):
        # Reset the game by reinitializing the grid and fire position
        global start_time
        start_time = datetime.now()
        self.grid = [[0] * self.size for _ in range(self.size)]
        fire_row = self.size // 2
        fire_col = self.size // 2
        self.fire_position = (fire_row, fire_col)
        self.grid[self.fire_position[0]][self.fire_position[1]] = 1
        self.update_board()
        self.moves_count = 0
        self.update_scoreboard()


# Checks whether player name already exists in highscores.
# Returns True if player name doesn't exist or player wants to continue using existing name
# Returns False if player wants to use a different name
def check_player_name(player_name):
    global highscores
    
    # If no highscores yet, no need to check
    if not highscores:
        return True
    
    # Check if player name exists in highscores
    player_names = [score["name"] for score in highscores]
    
    if player_name in player_names:
        result = messagebox.askquestion(
            "The player name already exists", "Do you want to continue using this player name?")
        if result == 'yes':
            return True
        else:
            return False
    
    return True


def main():
    # Prompt the user for player name
    global player_name
    player_name = askstring("Player Name", "Enter your name:")

    # Handle empty player name
    if not player_name:
        player_name = "Anonymous"
    
    # Check if player name exists and handle accordingly
    while not check_player_name(player_name):
        player_name = askstring("Player Name", "Enter your name:")
        if not player_name:
            player_name = "Anonymous"
            break

    # Prompt the user for difficulty level
    game_modes = ['easy', 'normal', 'hard', 'impossible', 'custom']
    difficulty_level = askstring(
        "Game Mode", f"Choose game mode: {', '.join(game_modes)}")

    if difficulty_level and difficulty_level.lower() in game_modes:
        # If custom mode, ask for custom size
        custom_size = None
        if difficulty_level.lower() == 'custom':
            custom_size = askstring(
                "Custom Size", "Enter custom grid size (e.g, 10):")
            try:
                custom_size = int(custom_size) + 2
            except ValueError:
                custom_size = 11

        root_window = tk.Tk()
        root_window.title("Forest Fire Challenge")

        game = ForestFireGame(root_window, difficulty_level.lower(), custom_size)

        root_window.mainloop()
    else:
        messagebox.showinfo(
            "Invalid Input", f"Please enter a valid game mode: {', '.join(game_modes)}.")

if __name__ == "__main__":
    main()
