# enemy.py
import random
from .player import Difficulty
ENEMIES = [
    {"name": "Rat", "hp": 8, "atk": 2, "def": 0, "exp": 8, "gold": 3},
    {"name": "Bandit", "hp": 14, "atk": 4, "def": 1, "exp": 18, "gold": 8},
    {"name": "Wolf", "hp": 18, "atk": 6, "def": 2, "exp": 28, "gold": 12},
    {"name": "Goblin", "hp": 24, "atk": 8, "def": 3, "exp": 40, "gold": 18},
    {"name": "Orc", "hp": 36, "atk": 10, "def": 4, "exp": 60, "gold": 30},
]

def scale_enemy_spec(spec, player_level, difficulty=Difficulty.NORMAL):
    base_hp = spec.get("hp", 10) + player_level * 2
    base_atk = spec.get("atk", 2) + player_level // 2
    base_def = spec.get("def", 0)
    multiplier = difficulty.value
    return {
        "name": spec.get("name", "Foe"),
        "hp": int(max(1, base_hp * multiplier)),
        "atk": int(max(1, base_atk * multiplier)),
        "def": int(max(0, base_def * multiplier)),
        "exp": int(spec.get("exp", 0) * multiplier),
        "gold": int(spec.get("gold", 0) * multiplier),
    }
