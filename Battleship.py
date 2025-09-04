import random
import string
import sys

#!/usr/bin/env python3
"""
Battleship - single player vs computer
Save this file as Battleship.py and run with: python Battleship.py

Instructions:
- You play against the computer on a 10x10 grid labeled A-J (rows) and 1-10 (columns).
- You may place your ships manually or have them placed randomly.
- Enter attack coordinates like "A5" or "j10".
- The goal is to sink all enemy ships before the computer sinks yours.
- Ship types and sizes:
    Carrier      - 5
    Battleship   - 4
    Cruiser      - 3
    Submarine    - 3
    Destroyer    - 2
"""


GRID_SIZE = 10
ROWS = list(string.ascii_uppercase[:GRID_SIZE])
COLS = list(range(1, GRID_SIZE + 1))
SHIP_TYPES = [
    ("Carrier", 5),
    ("Battleship", 4),
    ("Cruiser", 3),
    ("Submarine", 3),
    ("Destroyer", 2),
]


def create_empty_grid():
    return [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


def print_boards(player_board, enemy_view):
    # Show player's board and enemy view side by side
    def row_to_str(row):
        return " ".join(row)

    header = "   " + " ".join(f"{c:2}" for c in COLS)
    print("Your board".ljust(35) + "Enemy board")
    print(header.ljust(35) + header)
    for r_idx, r_label in enumerate(ROWS):
        left = f"{r_label:2} {row_to_str(player_board[r_idx])}"
        right = f"{r_label:2} {row_to_str(enemy_view[r_idx])}"
        print(left.ljust(35) + right)
    print()


def coord_to_index(coord):
    coord = coord.strip().upper()
    if len(coord) < 2:
        return None
    row_char = coord[0]
    col_str = coord[1:]
    if row_char not in ROWS:
        return None
    try:
        col = int(col_str)
    except ValueError:
        return None
    if col not in COLS:
        return None
    return (ROWS.index(row_char), col - 1)


def valid_placement(board, start, length, orientation):
    r, c = start
    if orientation == "H":
        if c + length > GRID_SIZE:
            return False
        for i in range(length):
            if board[r][c + i] != ".":
                return False
    else:
        if r + length > GRID_SIZE:
            return False
        for i in range(length):
            if board[r + i][c] != ".":
                return False
    return True


def place_ship(board, ship_coords_map, ship_name, length, start, orientation):
    coords = []
    r, c = start
    for i in range(length):
        rr, cc = (r, c + i) if orientation == "H" else (r + i, c)
        board[rr][cc] = "O"
        coords.append((rr, cc))
    ship_coords_map[ship_name] = coords


def random_place_all_ships(board):
    ships = {}
    for name, length in SHIP_TYPES:
        placed = False
        tries = 0
        while not placed and tries < 1000:
            tries += 1
            orientation = random.choice(["H", "V"])
            r = random.randrange(GRID_SIZE)
            c = random.randrange(GRID_SIZE)
            if valid_placement(board, (r, c), length, orientation):
                place_ship(board, ships, name, length, (r, c), orientation)
                placed = True
        if not placed:
            raise RuntimeError("Failed to place ships randomly")
    return ships


def manual_place_all_ships(board):
    ships = {}
    print("\nManual ship placement:")
    print("Enter coordinates like A1, B10. Orientation H or V.")
    print_boards(board, create_empty_grid())
    for name, length in SHIP_TYPES:
        while True:
            try:
                inp = input(f"Place your {name} (size {length}). Start coord: ").strip()
                start = coord_to_index(inp)
                if start is None:
                    print("Invalid coordinate. Try again.")
                    continue
                orientation = input("Orientation (H or V): ").strip().upper()
                if orientation not in ("H", "V"):
                    print("Invalid orientation. Use H or V.")
                    continue
                if not valid_placement(board, start, length, orientation):
                    print("Invalid placement (overlap or out of bounds). Try again.")
                    continue
                place_ship(board, ships, name, length, start, orientation)
                print_boards(board, create_empty_grid())
                break
            except KeyboardInterrupt:
                print("\nPlacement cancelled.")
                sys.exit(0)
    return ships


def print_instructions():
    print(__doc__)
    print("Ready? Let's play!\n")


def attack(board, ships_map, target, record_board):
    r, c = target
    cell = board[r][c]
    if cell == "X" or cell == "-":
        return "repeat", None
    if cell == "O":
        board[r][c] = "X"
        # find which ship received hit
        sunk_ship = None
        for name, coords in ships_map.items():
            if (r, c) in coords:
                coords.remove((r, c))
                if not coords:
                    sunk_ship = name
                break
        record_board[r][c] = "X"
        if sunk_ship:
            return "sunk", sunk_ship
        else:
            return "hit", None
    else:
        board[r][c] = "-"
        record_board[r][c] = "-"
        return "miss", None


def all_sunk(ships_map):
    return all(len(coords) == 0 for coords in ships_map.values())


def computer_choose_move(available_moves, last_hits):
    # Simple logic: if we have a last hit, try adjacent; else random
    if last_hits:
        # try neighbors of the most recent hit
        r, c = last_hits[-1]
        neighbors = [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]
        random.shuffle(neighbors)
        for rr, cc in neighbors:
            if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE and (rr, cc) in available_moves:
                available_moves.remove((rr, cc))
                return (rr, cc)
    move = random.choice(tuple(available_moves))
    available_moves.remove(move)
    return move


def main():
    random.seed()

    print_instructions()

    # Setup boards
    player_board = create_empty_grid()
    enemy_board = create_empty_grid()
    enemy_view = create_empty_grid()  # what player sees of enemy (hits/misses)
    player_ships = {}
    enemy_ships = {}

    choice = input("Place ships manually? (y/N): ").strip().lower()
    if choice == "y":
        player_ships = manual_place_all_ships(player_board)
    else:
        player_ships = random_place_all_ships(player_board)
        print("Your ships placed randomly.")
    enemy_ships = random_place_all_ships(enemy_board)

    # Track computer move choices
    comp_moves = set((r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE))
    comp_last_hits = []

    turn = "player"

    try:
        while True:
            print_boards(player_board, enemy_view)
            if turn == "player":
                inp = input("Enter attack coordinate (or 'quit'): ").strip()
                if inp.lower() in ("quit", "exit"):
                    print("Game aborted.")
                    break
                target = coord_to_index(inp)
                if target is None:
                    print("Invalid coordinate. Use format like A5.")
                    continue
                res, sunk_ship = attack(enemy_board, enemy_ships, target, enemy_view)
                if res == "repeat":
                    print("You already attacked that coordinate. Try again.")
                    continue
                if res == "hit":
                    print("Hit!")
                elif res == "sunk":
                    print(f"You sunk the enemy's {sunk_ship}!")
                else:
                    print("Miss.")
                if all_sunk(enemy_ships):
                    print_boards(player_board, enemy_view)
                    print("Congratulations — you sank all enemy ships. You win!")
                    break
                turn = "computer"
            else:
                # computer turn
                target = computer_choose_move(comp_moves, comp_last_hits)
                res, sunk_ship = attack(player_board, player_ships, target, create_empty_grid())  # not storing a visible enemy view
                # We need to mark player's board display accordingly (attack mutates player_board)
                r, c = target
                if res == "hit" or res == "sunk":
                    comp_last_hits.append((r, c))
                if res == "sunk":
                    print(f"Computer sank your {sunk_ship} at {ROWS[r]}{c+1}!")
                    # remove hits related to that ship from last_hits
                    comp_last_hits = [h for h in comp_last_hits if h != (r, c)]
                elif res == "hit":
                    print(f"Computer hit at {ROWS[r]}{c+1}.")
                elif res == "miss":
                    print(f"Computer missed at {ROWS[r]}{c+1}.")
                elif res == "repeat":
                    # should not happen due to available moves set, but handle
                    pass

                if all_sunk(player_ships):
                    print_boards(player_board, enemy_view)
                    print("All your ships have been sunk. You lose.")
                    break
                turn = "player"
    except KeyboardInterrupt:
        print("\nGame interrupted. Goodbye.")


if __name__ == "__main__":
    main()