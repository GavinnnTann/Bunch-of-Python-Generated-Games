import pygame
import sys
import tkinter as tk
from tkinter import messagebox
import numpy as np
import time
import random

# Game constants
BOARD_SIZE = 15  # 15x15 board
CELL_SIZE = 40   # Size of each board cell in pixels
STONE_RADIUS = 18  # Size of stones
BOARD_COLOR = (220, 179, 92)  # Wooden board color
LINE_COLOR = (0, 0, 0)  # Black lines
MARGIN = 30  # Board margin
GRID_WIDTH = BOARD_SIZE * CELL_SIZE + 2 * MARGIN  # Window size with margin
GRID_HEIGHT = BOARD_SIZE * CELL_SIZE + 2 * MARGIN
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


class OmokGame:
    def __init__(self, game_mode="PvP"):
        """
        Initialize the Omok game.
        
        Args:
            game_mode (str): "PvP" for player vs player, "PvE" for player vs AI
        """
        pygame.init()
        self.screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
        pygame.display.set_caption("Omok (Gomoku)")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)  # 0: empty, 1: black, 2: white
        self.current_player = 1  # Black starts
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.game_mode = game_mode
        self.move_count = 0  # Track the number of moves made
        
        # AI difficulty (for PvE mode)
        self.ai_difficulty = 2  # 1: Easy, 2: Medium, 3: Hard
        
        # Opening book strategies
        self.opening_book = {
            # Common Omok opening patterns: (x, y) coordinates
            'center': [(BOARD_SIZE//2, BOARD_SIZE//2)],
            'star_points': [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)],
            'near_center': [(7, 6), (7, 8), (6, 7), (8, 7), (6, 6), (8, 8), (6, 8), (8, 6)],
            
            # Standard opening sequences (first move depends on player color)
            'standard_black': [(BOARD_SIZE//2, BOARD_SIZE//2)],  # Black usually starts center
            'standard_white': [(BOARD_SIZE//2-1, BOARD_SIZE//2-1), (BOARD_SIZE//2+1, BOARD_SIZE//2+1),
                              (BOARD_SIZE//2-1, BOARD_SIZE//2+1), (BOARD_SIZE//2+1, BOARD_SIZE//2-1)]
        }
    
    def draw_board(self):
        """Draw the Omok board with grid lines."""
        # Draw the wooden board background
        self.screen.fill(BOARD_COLOR)
        
        # Draw grid lines
        for i in range(BOARD_SIZE):
            # Horizontal lines
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (MARGIN, MARGIN + i * CELL_SIZE),
                (GRID_WIDTH - MARGIN, MARGIN + i * CELL_SIZE),
                2
            )
            
            # Vertical lines
            pygame.draw.line(
                self.screen, LINE_COLOR,
                (MARGIN + i * CELL_SIZE, MARGIN),
                (MARGIN + i * CELL_SIZE, GRID_HEIGHT - MARGIN),
                2
            )
        
        # Draw star points (traditionally at specific intersections)
        star_points = [3, 7, 11]
        for y in star_points:
            for x in star_points:
                pygame.draw.circle(
                    self.screen, BLACK,
                    (MARGIN + x * CELL_SIZE, MARGIN + y * CELL_SIZE),
                    4
                )
    
    def draw_stones(self):
        """Draw all stones on the board."""
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 1:  # Black stone
                    pygame.draw.circle(
                        self.screen, BLACK,
                        (MARGIN + x * CELL_SIZE, MARGIN + y * CELL_SIZE),
                        STONE_RADIUS
                    )
                elif self.board[y][x] == 2:  # White stone
                    pygame.draw.circle(
                        self.screen, WHITE,
                        (MARGIN + x * CELL_SIZE, MARGIN + y * CELL_SIZE),
                        STONE_RADIUS
                    )
                    # Draw black outline for white stones for better visibility
                    pygame.draw.circle(
                        self.screen, BLACK,
                        (MARGIN + x * CELL_SIZE, MARGIN + y * CELL_SIZE),
                        STONE_RADIUS, 1
                    )
        
        # Highlight the last move
        if self.last_move:
            last_x, last_y = self.last_move
            pygame.draw.circle(
                self.screen, RED,
                (MARGIN + last_x * CELL_SIZE, MARGIN + last_y * CELL_SIZE),
                5
            )
    
    def draw_game_state(self):
        """Draw current game state information."""
        font = pygame.font.Font(None, 30)
        player_text = "Black's Turn" if self.current_player == 1 else "White's Turn"
        
        if self.game_over:
            if self.winner == 1:
                player_text = "Black Wins!"
            elif self.winner == 2:
                player_text = "White Wins!"
        
        text = font.render(player_text, True, BLACK)
        self.screen.blit(text, (10, 10))
    
    def draw(self):
        """Draw all game elements."""
        self.draw_board()
        self.draw_stones()
        self.draw_game_state()
        pygame.display.update()
    
    def get_board_position(self, mouse_pos):
        """Convert mouse position to board coordinates."""
        x, y = mouse_pos
        
        # Calculate grid position considering margin
        grid_x = round((x - MARGIN) / CELL_SIZE)
        grid_y = round((y - MARGIN) / CELL_SIZE)
        
        # Check if the position is valid
        if 0 <= grid_x < BOARD_SIZE and 0 <= grid_y < BOARD_SIZE:
            return grid_x, grid_y
        else:
            return None
    
    def place_stone(self, pos):
        """Place a stone at the given position."""
        if self.board[pos[1]][pos[0]] == 0 and not self.game_over:  # Empty position
            self.board[pos[1]][pos[0]] = self.current_player
            self.last_move = pos
            self.move_count += 1  # Increment move counter
            
            # Check for win
            if self.check_win(pos):
                self.game_over = True
                self.winner = self.current_player
                return True
            
            # Switch players
            self.current_player = 3 - self.current_player  # Toggle between 1 and 2
            return True
        return False
    
    def check_win(self, pos):
        """Check if the current player has won after placing a stone."""
        directions = [
            [(0, 1), (0, -1)],  # Vertical
            [(1, 0), (-1, 0)],  # Horizontal
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)]   # Diagonal \
        ]
        
        x, y = pos
        player = self.board[y][x]
        
        for dir_pair in directions:
            count = 1  # Start with the stone just placed
            
            # Check both directions
            for dx, dy in dir_pair:
                nx, ny = x, y
                while True:
                    nx, ny = nx + dx, ny + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny][nx] == player:
                        count += 1
                    else:
                        break
            
            # If 5 or more in a row, win
            if count >= 5:
                return True
        
        return False
    
    def ai_move(self):
        """Make an AI move based on difficulty level."""
        # Use opening book for early game if playing hard mode
        if self.ai_difficulty == 3 and self.move_count < 6:
            opening_move = self.get_opening_move()
            if opening_move:
                return self.place_stone(opening_move)
        
        # Standard difficulty-based move selection
        if self.ai_difficulty == 1:
            # Easy: Random valid move
            return self.ai_easy_move()
        elif self.ai_difficulty == 2:
            # Medium: Mix of random and defensive moves
            return self.ai_medium_move()
        else:
            # Hard: More advanced strategy
            return self.ai_hard_move()
    
    def get_opening_move(self):
        """Get a strategic opening move from the opening book."""
        # Determine which opening strategy to use based on AI's color and game state
        if self.current_player == 1:  # AI is black (first player)
            if self.move_count == 0:
                # First move of the game, place at center or a strategic point
                return random.choice(self.opening_book['standard_black'])
            elif self.move_count == 2:
                # Third move (second black move)
                center = (BOARD_SIZE//2, BOARD_SIZE//2)
                if self.board[center[1]][center[0]] == 1:
                    # We played center, now play a strategic second move
                    options = self.opening_book['near_center']
                    # Filter out any occupied positions
                    options = [(x, y) for x, y in options if self.board[y][x] == 0]
                    if options:
                        return random.choice(options)
        else:  # AI is white (second player)
            if self.move_count == 1:
                # Second move of the game (first white move)
                center = (BOARD_SIZE//2, BOARD_SIZE//2)
                if self.board[center[1]][center[0]] == 1:
                    # Opponent played center, respond with a near-center move
                    return random.choice(self.opening_book['standard_white'])
                else:
                    # Opponent didn't play center, we take it
                    return (BOARD_SIZE//2, BOARD_SIZE//2)
            
        # Check if we can play at or near a star point
        star_options = []
        for x, y in self.opening_book['star_points']:
            if self.board[y][x] == 0:
                star_options.append((x, y))
        
        if star_options:
            return random.choice(star_options)
            
        # If no opening moves are valid, return None and let regular AI take over
        return None
    
    def ai_easy_move(self):
        """Make a random valid move."""
        empty_cells = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:
                    empty_cells.append((x, y))
        
        if empty_cells:
            x, y = random.choice(empty_cells)
            return self.place_stone((x, y))
        return False
    
    def ai_medium_move(self):
        """Medium AI: Looks for defensive moves or makes a random move."""
        # First, check if AI can win in the next move
        winning_move = self.find_winning_move(self.current_player)
        if winning_move:
            return self.place_stone(winning_move)
        
        # Then check if need to block opponent's winning move
        opponent = 3 - self.current_player
        blocking_move = self.find_winning_move(opponent)
        if blocking_move:
            return self.place_stone(blocking_move)
        
        # Otherwise make a random move near existing stones
        return self.ai_easy_move()
    
    def find_winning_move(self, player):
        """Find a winning move for the given player."""
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:  # Empty cell
                    # Try placing a stone and check if it's a winning move
                    self.board[y][x] = player
                    if self.check_win((x, y)):
                        self.board[y][x] = 0  # Undo the move
                        return (x, y)
                    self.board[y][x] = 0  # Undo the move
        return None
    
    def ai_hard_move(self):
        """Hard AI: More strategic moves using minimax with alpha-beta pruning."""
        # First, check for immediate win
        winning_move = self.find_winning_move(self.current_player)
        if winning_move:
            return self.place_stone(winning_move)
        
        # Then check if need to block opponent's winning move
        opponent = 3 - self.current_player
        blocking_move = self.find_winning_move(opponent)
        if blocking_move:
            return self.place_stone(blocking_move)
        
        # Use minimax with alpha-beta pruning for deeper analysis
        depth = 3  # Look ahead depth - higher values make stronger but slower AI
        best_move = self.minimax_move(depth)
        
        if best_move:
            return self.place_stone(best_move)
        
        # Fallback to smart move evaluation if minimax didn't find a good move
        # This handles early game better when the board is mostly empty
        best_score = -float('inf')
        best_move = None
        
        # Focus search around existing stones for efficiency
        candidates = self.get_candidate_moves()
        for x, y in candidates:
            if self.board[y][x] == 0:  # Empty cell
                score = self.evaluate_move((x, y))
                if score > best_score:
                    best_score = score
                    best_move = (x, y)
        
        if best_move:
            return self.place_stone(best_move)
        
        # Final fallback to random move
        return self.ai_easy_move()
    
    def minimax_move(self, depth):
        """Use minimax algorithm to find the best move with time limit."""
        best_score = -float('inf')
        best_move = None
        
        # Get candidate moves for efficiency instead of checking every cell
        candidates = self.get_candidate_moves()
        
        # Limit the number of candidate moves to prevent hanging
        if len(candidates) > 10:
            # If too many candidates, evaluate them quickly and pick top 10
            move_scores = []
            for x, y in candidates:
                if self.board[y][x] == 0:  # Empty cell
                    # Quick evaluation
                    score = self.evaluate_move_quick((x, y), self.current_player)
                    move_scores.append((score, (x, y)))
            
            # Sort by score and take top 10
            move_scores.sort(reverse=True)
            candidates = [move for _, move in move_scores[:10]]
        
        # Set a maximum time for minimax search (2 seconds)
        start_time = time.time()
        max_time = 1.0  # 1 second max search time
        
        for x, y in candidates:
            if self.board[y][x] == 0:  # Empty cell
                # Check if we're running out of time
                if time.time() - start_time > max_time:
                    break  # Time limit exceeded, use best move found so far
                
                # Place the stone temporarily
                self.board[y][x] = self.current_player
                
                # Calculate score via minimax
                score = self.minimax(depth - 1, -float('inf'), float('inf'), False, start_time, max_time)
                
                # Remove the stone
                self.board[y][x] = 0
                
                if score > best_score:
                    best_score = score
                    best_move = (x, y)
                
                # If we found a winning move, no need to continue searching
                if score >= 9000:
                    return best_move
        
        return best_move
    
    def minimax(self, depth, alpha, beta, is_maximizing, start_time, max_time):
        """Minimax algorithm with alpha-beta pruning and time limit."""
        # Time check - abort if we're taking too long
        if time.time() - start_time > max_time:
            # Return current evaluation if out of time
            return self.evaluate_board() if is_maximizing else -self.evaluate_board()
        
        # Check for terminal states or maximum depth
        if depth == 0:
            return self.evaluate_board()
            
        # Quickly check if there's a winner
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] != 0:
                    if self.check_win_at((x, y)):
                        if self.board[y][x] == self.current_player:
                            return 10000 + depth  # Win (prefer quicker wins)
                        else:
                            return -10000 - depth  # Loss (avoid quicker losses)
        
        # Get move candidates - limit to fewer moves for deeper levels
        candidates = self.get_candidate_moves()
        if len(candidates) > 20 - depth*5:  # Reduce candidates at deeper levels
            candidates = candidates[:20 - depth*5]
        
        if is_maximizing:
            max_eval = -float('inf')
            for x, y in candidates:
                if self.board[y][x] == 0:
                    self.board[y][x] = self.current_player
                    eval = self.minimax(depth - 1, alpha, beta, False, start_time, max_time)
                    self.board[y][x] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 3 - self.current_player
            for x, y in candidates:
                if self.board[y][x] == 0:
                    self.board[y][x] = opponent
                    eval = self.minimax(depth - 1, alpha, beta, True, start_time, max_time)
                    self.board[y][x] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
            return min_eval
            
    def check_win_at(self, pos):
        """Check if there's a win at the specific position."""
        x, y = pos
        player = self.board[y][x]
        if player == 0:
            return False
            
        directions = [
            [(0, 1), (0, -1)],  # Vertical
            [(1, 0), (-1, 0)],  # Horizontal
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)]   # Diagonal \
        ]
        
        for dir_pair in directions:
            count = 1  # Start with the stone at the position
            
            # Check both directions
            for dx, dy in dir_pair:
                nx, ny = x, y
                while True:
                    nx, ny = nx + dx, ny + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny][nx] == player:
                        count += 1
                    else:
                        break
            
            if count >= 5:
                return True
                
        return False
    
    def check_winner(self):
        """Check if there's a winner on the current board."""
        # Check horizontal, vertical, and diagonal lines
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] != 0:  # Not empty
                    player = self.board[y][x]
                    # Check all 4 directions
                    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
                    
                    for dx, dy in directions:
                        count = 1  # Current stone
                        # Look ahead in this direction
                        for i in range(1, 5):  # Need 5 in a row
                            nx, ny = x + i*dx, y + i*dy
                            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny][nx] == player:
                                count += 1
                            else:
                                break
                                
                        if count >= 5:
                            return player
                            
        return 0  # No winner
    
    def get_candidate_moves(self):
        """Get list of candidate moves (empty cells near existing stones)."""
        candidates = set()
        
        # Consider cells adjacent to existing stones
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] != 0:  # Stone exists here
                    # Check all 8 adjacent cells
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and self.board[ny][nx] == 0:
                                candidates.add((nx, ny))
        
        # If no stones on board or no candidates found, use center and surrounding area
        if not candidates:
            center = BOARD_SIZE // 2
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    nx, ny = center + dx, center + dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        candidates.add((nx, ny))
        
        return list(candidates)
    
    def evaluate_board(self):
        """Evaluate the entire board state."""
        score = 0
        
        # Evaluate all possible moves for current player
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] == 0:  # Empty cell
                    # Temporarily place stone and evaluate
                    self.board[y][x] = self.current_player
                    move_score = self.evaluate_move_quick((x, y), self.current_player)
                    self.board[y][x] = 0
                    score += move_score
                    
                    # Also evaluate opponent's potential at this position
                    self.board[y][x] = 3 - self.current_player
                    opp_score = self.evaluate_move_quick((x, y), 3 - self.current_player)
                    self.board[y][x] = 0
                    score -= opp_score * 0.9  # Slightly less weight on opponent's potential
        
        return score
    
    def evaluate_move_quick(self, pos, player):
        """A faster version of evaluate_move for use in minimax."""
        x, y = pos
        
        # Check patterns in all 4 directions
        directions = [
            [(1, 0), (-1, 0)],  # Horizontal
            [(0, 1), (0, -1)],  # Vertical
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)]   # Diagonal \
        ]
        
        # Pattern scores
        scores = {5: 100000, 4: 10000, 3: 1000, 2: 100, 1: 10}
        open_bonus = 2  # Multiplier for open-ended patterns
        
        total_score = 0
        
        for dir_pair in directions:
            # Count consecutive stones in this direction
            consecutive = 1
            open_ends = 0
            
            for dx, dy in dir_pair:
                # Count in this direction
                nx, ny = x, y
                for _ in range(4):  # Look up to 4 steps away
                    nx += dx
                    ny += dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if self.board[ny][nx] == player:
                            consecutive += 1
                        elif self.board[ny][nx] == 0:
                            open_ends += 1
                            break
                        else:
                            break
                    else:
                        break
            
            # Calculate score based on pattern
            if consecutive >= 5:
                return 100000  # Immediate win
            
            # Adjust score based on openness of the pattern
            pattern_score = scores.get(consecutive, 0)
            if open_ends > 0:
                pattern_score *= (1 + (open_ends * 0.5))
            
            total_score += pattern_score
        
        return total_score
    
    def evaluate_move(self, pos):
        """Evaluate how good a move is based on patterns around it."""
        x, y = pos
        player = self.current_player
        opponent = 3 - player
        score = 0
        
        # Temporarily place the stone
        self.board[y][x] = player
        
        # Check patterns in all 4 directions (horizontal, vertical, 2 diagonals)
        directions = [
            [(1, 0), (-1, 0)],  # Horizontal
            [(0, 1), (0, -1)],  # Vertical
            [(1, 1), (-1, -1)],  # Diagonal /
            [(1, -1), (-1, 1)]   # Diagonal \
        ]
        
        # Pattern scores - higher values for more valuable patterns
        pattern_scores = {
            'win': 1000000,       # Five in a row - win
            'open_four': 50000,   # Open four (can win next move)
            'four': 10000,        # Closed four
            'open_three': 5000,   # Open three
            'three': 1000,        # Closed three
            'open_two': 500,      # Open two
            'two': 100,           # Closed two
            'one': 10             # Single stone
        }
        
        # Evaluate player's patterns
        player_score = self.evaluate_patterns(pos, player, directions, pattern_scores)
        
        # Remove the stone and place opponent's stone to evaluate blocking value
        self.board[y][x] = 0
        self.board[y][x] = opponent
        
        # Evaluate opponent's patterns (defensive value)
        opponent_score = self.evaluate_patterns(pos, opponent, directions, pattern_scores)
        
        # Remove opponent stone
        self.board[y][x] = 0
        
        # Calculate final score
        # Prioritize winning moves, then blocking opponent's wins, then creating threats
        if player_score >= pattern_scores['win']:
            score = player_score
        elif opponent_score >= pattern_scores['win']:
            score = opponent_score * 0.9  # Slightly prioritize winning over blocking
        else:
            # Balance between offense and defense, with emphasis on offense
            score = player_score * 1.1 + opponent_score * 0.9
        
        # Prefer center positions in early game (count stones to determine game phase)
        stone_count = np.count_nonzero(self.board)
        if stone_count < 20:  # Early game
            center_x, center_y = BOARD_SIZE // 2, BOARD_SIZE // 2
            distance_to_center = abs(x - center_x) + abs(y - center_y)
            score -= distance_to_center * 5  # Stronger center preference
        
        return score
    
    def evaluate_patterns(self, pos, player, directions, pattern_scores):
        """Evaluate patterns for a specific player at the given position."""
        x, y = pos
        total_score = 0
        
        for dir_pair in directions:
            # For each direction pair (e.g., left-right), count consecutive stones and open ends
            consecutive_stones = 1  # Start with the current stone
            open_ends = 0
            
            # Check each direction in the pair
            for direction_index, (dx, dy) in enumerate(dir_pair):
                # Look in this direction
                nx, ny = x, y
                for step in range(1, 5):  # Look up to 4 steps away
                    nx += dx
                    ny += dy
                    if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                        if self.board[ny][nx] == player:
                            consecutive_stones += 1
                        elif self.board[ny][nx] == 0:
                            open_ends += 1
                            break
                        else:  # Opponent stone - blocked
                            break
                    else:  # Edge of board - blocked
                        break
            
            # Calculate score based on the pattern formed
            # Five in a row = win
            if consecutive_stones >= 5:
                total_score += pattern_scores['win']
                continue  # No need to check further patterns in this direction
                
            # Four in a row
            if consecutive_stones == 4:
                if open_ends >= 1:  # Even one open end is dangerous
                    if open_ends == 2:
                        total_score += pattern_scores['open_four']  # Very dangerous
                    else:
                        total_score += pattern_scores['four']       # Dangerous
                    
            # Three in a row
            elif consecutive_stones == 3:
                if open_ends >= 1:
                    if open_ends == 2:
                        total_score += pattern_scores['open_three']  # Very threatening
                    else:
                        total_score += pattern_scores['three']       # Somewhat threatening
            
            # Two in a row
            elif consecutive_stones == 2:
                if open_ends >= 1:
                    if open_ends == 2:
                        total_score += pattern_scores['open_two']
                    else:
                        total_score += pattern_scores['two']
            
            # Single stone
            elif consecutive_stones == 1 and open_ends > 0:
                total_score += pattern_scores['one']
            
            # Bonus for creating threats in multiple directions
            # This makes positions that create multiple threats more valuable
            if consecutive_stones >= 2 and open_ends > 0:
                total_score += consecutive_stones * open_ends * 25
        
        return total_score
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    if self.current_player == 1 or (self.game_mode == "PvP" and self.current_player == 2):
                        # Human player's turn
                        mouse_pos = pygame.mouse.get_pos()
                        board_pos = self.get_board_position(mouse_pos)
                        
                        if board_pos:
                            self.place_stone(board_pos)
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game with 'R'
                        self.__init__(self.game_mode)
                    elif event.key == pygame.K_q:  # Quit with 'Q'
                        running = False
            
            # AI move if it's PvE mode and AI's turn
            if self.game_mode == "PvE" and self.current_player == 2 and not self.game_over:
                pygame.time.delay(500)  # Small delay for better user experience
                self.ai_move()
            
            # Draw everything
            self.draw()
            
            # Update the display
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        pygame.quit()


class OmokMenu:
    def __init__(self):
        """Initialize the Tkinter menu for Omok game."""
        self.root = tk.Tk()
        self.root.title("Omok Game")
        
        # Calculate window position to center it
        window_width = 400
        window_height = 450
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int((screen_width - window_width) / 2)
        center_y = int((screen_height - window_height) / 2)
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.root.resizable(False, False)
        
        # Title label
        title_label = tk.Label(self.root, text="Omok (Five in a Row)", font=("Arial", 24))
        title_label.pack(pady=20)
        
        # Game mode selection
        self.game_mode_var = tk.StringVar(value="PvP")
        
        # Radio buttons for game mode
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=10)
        
        tk.Label(mode_frame, text="Game Mode:", font=("Arial", 14)).pack()
        
        tk.Radiobutton(mode_frame, text="Player vs Player", variable=self.game_mode_var,
                      value="PvP", font=("Arial", 12)).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="Player vs Computer", variable=self.game_mode_var,
                      value="PvE", font=("Arial", 12)).pack(anchor=tk.W)
        
        # AI difficulty selection (visible only in PvE mode)
        self.difficulty_var = tk.IntVar(value=2)
        self.difficulty_frame = tk.Frame(self.root)
        self.difficulty_frame.pack(pady=10)
        
        tk.Label(self.difficulty_frame, text="AI Difficulty:", font=("Arial", 14)).pack()
        
        tk.Radiobutton(self.difficulty_frame, text="Easy", variable=self.difficulty_var,
                      value=1, font=("Arial", 12)).pack(anchor=tk.W)
        tk.Radiobutton(self.difficulty_frame, text="Medium", variable=self.difficulty_var,
                      value=2, font=("Arial", 12)).pack(anchor=tk.W)
        tk.Radiobutton(self.difficulty_frame, text="Hard", variable=self.difficulty_var,
                      value=3, font=("Arial", 12)).pack(anchor=tk.W)
        
        # Update visibility of difficulty options based on game mode
        self.game_mode_var.trace("w", self.update_difficulty_visibility)
        self.update_difficulty_visibility()
        
        # Start button
        start_button = tk.Button(self.root, text="Start Game", command=self.start_game,
                               font=("Arial", 16), bg="#4CAF50", fg="white")
        start_button.pack(pady=20)
    
    def update_difficulty_visibility(self, *args):
        """Show or hide difficulty options based on game mode."""
        if self.game_mode_var.get() == "PvE":
            self.difficulty_frame.pack(pady=10)
        else:
            self.difficulty_frame.pack_forget()
    
    def start_game(self):
        """Start the Omok game with selected options."""
        game_mode = self.game_mode_var.get()
        
        # Hide the Tkinter window
        self.root.withdraw()
        
        # Start the Pygame-based game
        game = OmokGame(game_mode=game_mode)
        
        # Set AI difficulty if in PvE mode
        if game_mode == "PvE":
            game.ai_difficulty = self.difficulty_var.get()
        
        try:
            game.run()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            # Show the Tkinter window again after the game ends
            self.root.deiconify()
    
    def run(self):
        """Run the Tkinter menu."""
        self.root.mainloop()


# Main entry point
if __name__ == "__main__":
    menu = OmokMenu()
    menu.run()
