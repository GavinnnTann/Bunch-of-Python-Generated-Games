import random
import sys
from typing import List, Tuple

#!/usr/bin/env python3

class Minesweeper:
    MINE = -1

    def __init__(self, rows=9, cols=9, mines=10):
        self.rows = rows
        self.cols = cols
        self.total_mines = max(1, min(mines, rows * cols - 1))
        self.board: List[List[int]] = [[0]*cols for _ in range(rows)]
        self.revealed = [[False]*cols for _ in range(rows)]
        self.flagged = [[False]*cols for _ in range(rows)]
        self.first_move = True
        self.game_over = False
        self.exploded: Tuple[int,int] = (-1,-1)

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors(self, r, c):
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if self.in_bounds(nr, nc):
                    yield nr, nc

    def place_mines(self, safe_r, safe_c):
        banned = {(safe_r, safe_c)}
        for nr,nc in self.neighbors(safe_r, safe_c):
            banned.add((nr,nc))
        choices = [(r,c) for r in range(self.rows) for c in range(self.cols) if (r,c) not in banned]
        if len(choices) < self.total_mines:
            choices = [(r,c) for r in range(self.rows) for c in range(self.cols)]
        mines = set(random.sample(choices, self.total_mines))
        for r,c in mines:
            self.board[r][c] = Minesweeper.MINE
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == Minesweeper.MINE:
                    continue
                cnt = sum(1 for nr,nc in self.neighbors(r,c) if self.board[nr][nc] == Minesweeper.MINE)
                self.board[r][c] = cnt

    def reveal(self, r, c):
        if not self.in_bounds(r,c) or self.revealed[r][c] or self.flagged[r][c] or self.game_over:
            return
        if self.first_move:
            self.place_mines(r,c)
            self.first_move = False
        if self.board[r][c] == Minesweeper.MINE:
            self.revealed[r][c] = True
            self.game_over = True
            self.exploded = (r,c)
            return
        # flood fill for zeros
        stack = [(r,c)]
        while stack:
            cr, cc = stack.pop()
            if not self.in_bounds(cr,cc) or self.revealed[cr][cc] or self.flagged[cr][cc]:
                continue
            self.revealed[cr][cc] = True
            if self.board[cr][cc] == 0:
                for nr,nc in self.neighbors(cr,cc):
                    if not self.revealed[nr][nc] and not self.flagged[nr][nc]:
                        stack.append((nr,nc))

    def toggle_flag(self, r, c):
        if not self.in_bounds(r,c) or self.revealed[r][c] or self.game_over:
            return
        self.flagged[r][c] = not self.flagged[r][c]

    def check_win(self):
        if self.game_over:
            return False
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != Minesweeper.MINE and not self.revealed[r][c]:
                    return False
        self.game_over = True
        return True

    def reveal_all(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.revealed[r][c] = True

    def render(self, show_coords=True):
        cols = self.cols
        rows = self.rows
        lines = []
        if show_coords:
            header = "   " + " ".join(f"{c+1:2d}" for c in range(cols))
            lines.append(header)
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                if self.revealed[r][c]:
                    if self.board[r][c] == Minesweeper.MINE:
                        ch = '*' if (r,c) == self.exploded else '*'
                    elif self.board[r][c] == 0:
                        ch = ' '
                    else:
                        ch = str(self.board[r][c])
                else:
                    if self.flagged[r][c]:
                        ch = 'F'
                    else:
                        ch = '.'
                row_cells.append(f"{ch:>2}")
            prefix = f"{r+1:2d}" if show_coords else ""
            lines.append(prefix + " " + " ".join(row_cells))
        return "\n".join(lines)

def prompt_int(prompt, default):
    try:
        v = input(prompt).strip()
        if v == "":
            return default
        return int(v)
    except Exception:
        return default

def parse_command(s: str):
    parts = s.strip().split()
    if not parts:
        return None
    cmd = parts[0].lower()
    if cmd in ('r','reveal','open'):
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            return ('reveal', int(parts[1])-1, int(parts[2])-1)
    if cmd in ('f','flag'):
        if len(parts) >= 3 and parts[1].isdigit() and parts[2].isdigit():
            return ('flag', int(parts[1])-1, int(parts[2])-1)
    if cmd in ('q','quit','exit'):
        return ('quit',)
    if cmd in ('h','help','?'):
        return ('help',)
    return None

def print_help():
    print("Commands:")
    print("  r ROW COL   - reveal cell (row,col) (1-based)")
    print("  f ROW COL   - toggle flag on cell")
    print("  q           - quit")
    print("  h           - show help")

def main():
    print("Minesweeper (text). Press Enter to accept defaults.")
    rows = prompt_int("Rows [9]: ", 9)
    cols = prompt_int("Cols [9]: ", 9)
    mines = prompt_int("Mines [10]: ", 10)
    game = Minesweeper(rows, cols, mines)
    print_help()
    while True:
        print()
        print(game.render())
        if game.game_over:
            if game.exploded != (-1,-1):
                game.reveal_all()
                print("\nBOOM! You hit a mine.")
                print(game.render())
            elif game.check_win():
                print("\nCongratulations! You cleared the field.")
                game.reveal_all()
                print(game.render())
            else:
                print("\nGame over.")
            break
        if game.check_win():
            print("\nCongratulations! You cleared the field.")
            game.reveal_all()
            print(game.render())
            break
        cmdline = input("cmd> ")
        parsed = parse_command(cmdline)
        if not parsed:
            print("Invalid command. Type h for help.")
            continue
        if parsed[0] == 'quit':
            print("Quit.")
            break
        if parsed[0] == 'help':
            print_help()
            continue
        action, r, c = parsed
        if not game.in_bounds(r,c):
            print("Coordinates out of bounds.")
            continue
        if action == 'reveal':
            game.reveal(r,c)
        elif action == 'flag':
            game.toggle_flag(r,c)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)