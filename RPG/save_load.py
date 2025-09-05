import time
# save_load.py
import json
import os
import os
SAVE_FILE = os.path.join(os.path.dirname(__file__), "rpg_save.json")

def save_game(player):
    data = player.to_dict()
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)
    print("Game saved.")

def load_game(Player):
    if not os.path.exists(SAVE_FILE):
        return None
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)
    return Player.from_dict(data)
