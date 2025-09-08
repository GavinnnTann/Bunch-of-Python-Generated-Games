from RPG.events import random_world_event
from RPG.player import Player, Skill, Actor
from RPG.item import Item
from RPG.quest import Quest
import time
# --- Required imports ---
import threading
import random
import uuid
import json
import os
import sys
import atexit
from enum import Enum
from RPG.shop import sell_menu


def _patch_shop(delay=0.05):
    """Wait until the original shop(), Item class and SHOP_ITEMS are defined later in this module, then wrap shop to add a sell menu."""
    # ensure required symbols exist before patching to avoid timing/name errors
    if "shop" not in globals() or "sell_menu" not in globals() or "SHOP_ITEMS" not in globals() or "Item" not in globals():
        threading.Timer(delay, lambda: _patch_shop(min(delay * 2, 1.0))).start()
        return
    orig_shop = globals().get("shop")
    ItemCls = globals().get("Item")
    SHOP_ITEMS_REF = globals().get("SHOP_ITEMS", [])

    def enhanced_shop(player):
        while True:
            print("\n--- Marketplace ---")
            print("1) Buy Items")
            print("2) Sell Items")
            print("b) Back")
            choice = input("> ").strip().lower()
            if choice == "1":
                # delegate to original shop implementation if possible
                try:
                    orig_shop(player)
                except Exception:
                    # fallback: replicate simple buy flow if original fails (use safe globals)
                    print("Marketplace temporarily unavailable. Showing simple buy list.")
                    for i, it in enumerate(SHOP_ITEMS_REF, 1):
                        print(f"{i}. {it.name} ({it.type}) - {it.price} gold")
                    sel = input("Choose item number or b) back: ").strip().lower()
                    if sel == "b" or not sel:
                        continue
                    try:
                        idx = int(sel) - 1
                        it = SHOP_ITEMS_REF[idx]
                        if player.gold < it.price:
                            print("You cannot afford that.")
                        else:
                            player.gold -= it.price
                            # prefer constructing a new Item instance if Item class is available,
                            # otherwise append the existing template object as a fallback.
                            if ItemCls:
                                player.inventory.append(ItemCls(it.name, it.type, getattr(it, "power", 0), getattr(it, "heal", 0), getattr(it, "price", 0)))
                            else:
                                player.inventory.append(it)
                            print(f"Purchased {it.name}.")
                    except Exception:
                        print("Invalid choice.")
            elif choice == "2":
                # call sell_menu which is now guaranteed to exist when this wrapper is installed
                sell_menu(player)
            elif choice == "b" or choice == "":
                return
            else:
                print("Invalid selection.")

    globals()["shop"] = enhanced_shop


# schedule shop patch so it replaces the later-defined shop() with an enhanced one
threading.Timer(0.05, _patch_shop).start()

# Enhanced marketplace & rarer finds
RARE_ITEM_TEMPLATES = [
    {"name": "Enchanted Short Sword", "type": "weapon", "power": 14, "heal": 0, "price": 180},
    {"name": "Elven Bow", "type": "weapon", "power": 12, "heal": 0, "price": 150},
    {"name": "Mystic Robe", "type": "armor", "power": 6, "heal": 0, "price": 140},
    {"name": "Greater Potion", "type": "consumable", "power": 0, "heal": 80, "price": 120},
    {"name": "Golden Amulet", "type": "quest", "power": 0, "heal": 0, "price": 0},
    {"name": "Runed Dagger", "type": "weapon", "power": 9, "heal": 0, "price": 95},
    {"name": "Reinforced Mail", "type": "armor", "power": 9, "heal": 0, "price": 135},
]

EXTRA_FINDABLES = ["Silver Ring", "Ancient Coin", "Crystal Shard", "Elixir Residue", "Enchanted Bark", "Iron Ingot"]
_world_expanded = False

def _build_item_from_tpl(tpl):
    # Construct Item at runtime (Item class defined later)
    return Item(tpl["name"], tpl["type"], power=tpl.get("power", 0), heal=tpl.get("heal", 0), price=tpl.get("price", 0))

def _generate_traveling_merchant():
    # Create a merchant NPC carrying a small, rotated selection of rare wares
    names = ["Maris", "Tob", "Selra", "Borin", "Ketha", "Lorin"]
    count = random.randint(2, 5)
    chosen = random.sample(RARE_ITEM_TEMPLATES, min(count, len(RARE_ITEM_TEMPLATES)))
    inv = []
    for tpl in chosen:
        it = _build_item_from_tpl(tpl)
        # give each merchant slightly different markup
        markup = random.uniform(0.85, 1.5)
        it.price = max(1, int(it.price * markup))
        inv.append(it)
    name = f"{random.choice(names)} the Peddler"
    return NPC(name, "merchant", dialog=["Finest goods from far away.", "Rare wares, take a look."], inventory=inv)

def _merchant_shopping_session(player, merchant):
    print(f"\nYou meet {merchant.name}. They offer rare items:")
    while True:
        if not merchant.inventory:
            print(f"{merchant.name} has nothing left to sell.")
            break
        for i, it in enumerate(merchant.inventory, 1):
            print(f"{i}. {it.name} ({it.type}) - {it.price} gold")
        print("b) Back")
        choice = input("> ").strip().lower()
        if choice == "b" or not choice:
            break
        try:
            idx = int(choice) - 1
            item = merchant.inventory[idx]
        except Exception:
            print("Invalid selection.")
            continue
        if player.gold < item.price:
            print("You cannot afford that.")
            continue
        player.gold -= item.price
        player.inventory.append(Item(item.name, item.type, item.power, item.heal, item.price))
        print(f"Purchased {item.name}.")
        # remove one copy from merchant stock
        merchant.inventory.pop(idx)
        # merchants may leave after a few sales
        if random.random() < 0.3:
            print(f"{merchant.name} thanks you and moves on.")
            break

def _new_random_world_event(player):
    r = random.random()
    # rare traveling merchant (very uncommon)
    if r < 0.04:
        merchant = _generate_traveling_merchant()
        _merchant_shopping_session(player, merchant)
        return f"You encountered {merchant.name} and browsed their wares."
    # trap / loss (slightly reduced chance to leave room for merchants)
    if r < 0.09:
        loss = min(player.gold, random.randint(5, 20))
        player.gold -= loss
        return f"You fell into a shallow pit and lost {loss} gold while recovering."
    # meet town NPC (chance to meet merchants/trainers/questgivers)
    if r < 0.18:
        npc = random.choice(TOWN_NPCS)
        print(f"\nYou encounter {npc.name} ({npc.role}).")
        if getattr(npc, "dialog", None):
            print(f"{npc.name} says: \"{npc.talk()}\"")
        while True:
            print("Options: (t)alk  (l)eave", end="")
            if npc.role == "merchant":
                print("  (b)uy/trade", end="")
            if npc.role == "trainer":
                print("  (r)eceive training", end="")
            if npc.role == "questgiver":
                print("  (q)uest", end="")
            print()
            choice = input("> ").strip().lower()
            if choice in ("l", "leave", "b"):  # leave or back
                return f"You leave {npc.name}."
            if choice in ("t", "talk"):
                print(f"{npc.name} says: \"{npc.talk()}\"")
            elif choice in ("b", "buy", "trade") and npc.role == "merchant":
                _merchant_shopping_session(player, npc)
            elif choice in ("r", "train", "training") and npc.role == "trainer":
                print(f"{npc.name} offers you some training. You gain +1 ATK!")
                player.atk += 1
            elif choice in ("q", "quest") and npc.role == "questgiver":
                # Offer a quest
                possible = [t for t in QUEST_TEMPLATES if getattr(t, "type", "").lower() in ("gather", "fetch", "item")]
                offered = None
                if possible:
                    tpl = random.choice(possible)
                    if not any(a.title == tpl.title for a in player.active_quests):
                        offered = Quest(str(uuid.uuid4()), tpl.title, tpl.description, tpl.type, tpl.target_name, tpl.target_count, tpl.level_req, tpl.reward_gold, tpl.reward_exp, [Item.from_dict(it.to_dict()) for it in tpl.reward_items])
                if offered is None:
                    if not any(a.title == "Gather Herbs" for a in player.active_quests):
                        offered = Quest(str(uuid.uuid4()), "Gather Herbs", "Collect 3 Herbs for the herbalist.", "gather", "Herb", 3, level_req=1, reward_gold=10, reward_exp=15, reward_items=[])
                if offered:
                    if player.level >= offered.level_req:
                        offered.accepted = True
                        player.active_quests.append(offered)
                        print(f"Quest accepted: {offered.title}")
                    else:
                        print(f"{npc.name} says: 'This quest is too advanced for you.'")
                else:
                    print(f"{npc.name} has no quests for you right now.")
            else:
                print("Invalid option.")
    # foraging / find items (expanded pool and a small chance to find rare gear)
    if r < 0.33:
        # small chance to find a crafted/rare item
        if random.random() < 0.06:
            tpl = random.choice(RARE_ITEM_TEMPLATES)
            found = _build_item_from_tpl(tpl)
            player.inventory.append(found)
            return f"You stumbled upon a rare item while foraging: {found.name}."
        else:
            ingredients = ["Herb", "Leather", "Wood", "Thread", "Bottle", "Iron Ingot"] + EXTRA_FINDABLES
            found = random.choice(ingredients)
            player.inventory.append(Item(found, "quest", price=0))
            return f"You foraged and found: {found}."
    return None

def _expand_world_after_load(delay=0.05):
    global _world_expanded
    if _world_expanded:
        return
    # Wait until key globals exist (defined later in file)
    if "SHOP_ITEMS" not in globals() or "TOWN_NPCS" not in globals():
        threading.Timer(delay, lambda: _expand_world_after_load(min(delay * 2, 1.0))).start()
        return
    # seed a few rare items into the town marketplace so players can occasionally buy them
    additions = random.sample(RARE_ITEM_TEMPLATES, k=min(3, len(RARE_ITEM_TEMPLATES)))
    for tpl in additions:
        SHOP_ITEMS.append(_build_item_from_tpl(tpl))
    # add an occasional wandering merchant to NPC pool so world events can pick them up
    TOWN_NPCS.append(NPC("Wandering Trader", "merchant", dialog=["I've traveled far and wide. Take a look at these!"]))
    # replace the original random_world_event with the enhanced one
    globals()["random_world_event"] = _new_random_world_event
    _world_expanded = True

# schedule augmentation shortly after module load so it runs once all later globals (Item, SHOP_ITEMS, TOWN_NPCS) exist
threading.Timer(0.05, _expand_world_after_load).start()
#!/usr/bin/env python3
# rpg.py - simple text-based medieval RPG with basic quest system

SAVE_FILE = os.path.join(os.path.dirname(__file__), "rpg_save.json")
# Additional features: difficulty modes, skills, NPCs, crafting and random events

# Difficulty modifies enemy strength / loot
class Difficulty(Enum):
    EASY = 0.8
    NORMAL = 1.0
    HARD = 1.25

GAME_DIFFICULTY = Difficulty.NORMAL

def set_difficulty(name: str):
    global GAME_DIFFICULTY
    try:
        GAME_DIFFICULTY = Difficulty[name.upper()]
    except Exception:
        GAME_DIFFICULTY = Difficulty.NORMAL

# Simple skill system
class Skill:
    def __init__(self, id_, name, description, mp_cost=0, power=0, heal=0):
        self.id = id_
        self.name = name
        self.description = description
        self.mp_cost = mp_cost
        self.power = power
        self.heal = heal

    def to_dict(self):
        return {"id": self.id, "name": self.name, "description": self.description,
                "mp_cost": self.mp_cost, "power": self.power, "heal": self.heal}

    @staticmethod
    def from_dict(d):
        return Skill(d["id"], d["name"], d["description"], d.get("mp_cost",0), d.get("power",0), d.get("heal",0))

# Basic NPC (merchant / trainer) data holder
class NPC:
    def __init__(self, name, role, dialog=None, inventory=None):
        self.name = name
        self.role = role  # "merchant", "trainer", "questgiver"
        self.dialog = dialog or []
        self.inventory = inventory or []

    def talk(self):
        if not self.dialog:
            return f"{self.name} has nothing to say."
        return self.dialog[random.randint(0, len(self.dialog)-1)]

# Crafting recipes (ingredient names -> result item name and simple attributes)
CRAFTING_RECIPES = {
    # "result": ( {"ingredient_name": count, ...}, {"type": "weapon"/"armor"/"consumable", "power": X, "heal": Y, "price": P} )
    "Sharpened Iron Sword": ({"Iron Ingot": 2, "Wood": 1}, {"type": "weapon", "power": 9, "heal": 0, "price": 60}),
    "Sturdy Leather Armor": ({"Leather": 3, "Thread": 2}, {"type": "armor", "power": 5, "heal": 0, "price": 45}),
    "Healing Salve": ({"Herb": 2, "Bottle": 1}, {"type": "consumable", "power": 0, "heal": 30, "price": 25}),
}

def craft_item(player, recipe_name):
    """Attempt to craft an item by name. Ingredients are matched by item name strings in inventory."""
    if recipe_name not in CRAFTING_RECIPES:
        return "Unknown recipe."
    ingredients, attrs = CRAFTING_RECIPES[recipe_name]
    inv_counts = {}
    for it in player.inventory:
        inv_counts[it.name] = inv_counts.get(it.name, 0) + 1
    # check ingredients
    for req_name, req_cnt in ingredients.items():
        if inv_counts.get(req_name, 0) < req_cnt:
            return f"Missing ingredients: {req_name} x{req_cnt} required."
    # remove ingredients (first matching occurrences)
    to_remove = dict(ingredients)
    new_inventory = []
    for it in player.inventory:
        if to_remove.get(it.name,0) > 0:
            to_remove[it.name] -= 1
            continue
        new_inventory.append(it)
    player.inventory = new_inventory
    # create crafted Item (Item class exists later so we construct by name and attributes at runtime)
    crafted = Item(recipe_name, attrs["type"], power=attrs.get("power",0), heal=attrs.get("heal",0), price=attrs.get("price",0))
    player.inventory.append(crafted)
    return f"Crafted {crafted.name}."

# Small pool of NPCs that can appear during exploration or in town
TOWN_NPCS = [
    NPC("Lysa the Merchant", "merchant", dialog=["Fresh goods!", "Best prices in town."], inventory=[]),
    NPC("Gorim the Trainer", "trainer", dialog=["Train hard and you'll survive.", "Learned a new stance today."]),
    NPC("Elda the Herbalist", "questgiver", dialog=["I could use some herbs..."]),
]

# Random world events that may occur during exploration
def random_world_event(player):
    """
    Returns a descriptive string of the event (or None), and ensures NPC encounters always
    produce at least a dialogue line or an interactive flow where appropriate.
    """
    r = random.random()
    if r < 0.05:
        # trap event - lose some gold or HP
        loss = min(player.gold, random.randint(5, 20))
        player.gold -= loss
        return f"You fell into a shallow pit and lost {loss} gold while recovering."

    # higher and more robust chance to meet NPCs and ensure interaction happens
    if r < 0.18:
        if not TOWN_NPCS:
            return None
        npc = random.choice(TOWN_NPCS)
        # Always show a line of dialog if available
        if getattr(npc, "dialog", None):
            print(f"\n{npc.name} says: \"{npc.talk()}\"")

        # Merchant: try merchant-specific session, then fallback to general shop
        if npc.role == "merchant":
            try:
                if "_merchant_shopping_session" in globals() and getattr(npc, "inventory", None):
                    _merchant_shopping_session(player, npc)
                else:
                    if "shop" in globals():
                        shop(player)
            except Exception:
                pass
            return f"You encounter {npc.name}, a traveling merchant."

        # Trainer: at minimum they speak; possible future extension to train player
        if npc.role == "trainer":
            return f"You meet {npc.name}, a seasoned trainer. Perhaps you can learn something."

        # Questgiver: attempt to offer a gather-type quest, prefer templates
        if npc.role == "questgiver":
            try:
                possible = [t for t in QUEST_TEMPLATES if getattr(t, "type", "").lower() in ("gather", "fetch", "item")]
                offered = None
                if possible:
                    tpl = random.choice(possible)
                    if not any(a.title == tpl.title for a in player.active_quests):
                        offered = Quest(str(uuid.uuid4()), tpl.title, tpl.description, tpl.type, tpl.target_name, tpl.target_count, tpl.level_req, tpl.reward_gold, tpl.reward_exp, [Item.from_dict(it.to_dict()) for it in tpl.reward_items])
                if offered is None:
                    if not any(a.title == "Gather Herbs" for a in player.active_quests):
                        offered = Quest(str(uuid.uuid4()), "Gather Herbs", "Collect 3 Herbs for the herbalist.", "gather", "Herb", 3, level_req=1, reward_gold=10, reward_exp=15, reward_items=[])
                if offered:
                    if player.level >= offered.level_req:
                        offered.accepted = True
                        player.active_quests.append(offered)
                        return f"You meet {npc.name}. They ask for help: {offered.title}. Quest accepted."
                    else:
                        return f"You meet {npc.name} who needs help but the task is too advanced for you."
            except Exception:
                pass
            return f"You meet {npc.name} who seems to need assistance."

        return f"You meet {npc.name} who seems to need assistance."

    if r < 0.27:
        # find ingredients
        ingredients = ["Herb", "Leather", "Wood", "Thread", "Bottle", "Iron Ingot"]
        found = random.choice(ingredients)
        try:
            player.inventory.append(Item(found, "quest", price=0))
        except Exception:
            # fallback: append a simple Item-like dict if Item class isn't available
            class SimpleItem:
                def __init__(self, name):
                    self.name = name
                    self.type = "quest"
                    self.power = 0
                    self.heal = 0
                    self.price = 0
            player.inventory.append(SimpleItem(found))
        return f"You foraged and found: {found}."
    return None

# Utility to scale enemy specs by difficulty / player level (used by encounter)
def scale_enemy_spec(spec, player_level):
    base_hp = spec.get("hp", 10) + player_level * 2
    base_atk = spec.get("atk", 2) + player_level // 2
    base_def = spec.get("def", 0)
    multiplier = GAME_DIFFICULTY.value
    return {
        "name": spec.get("name", "Foe"),
        "hp": int(max(1, base_hp * multiplier)),
        "atk": int(max(1, base_atk * multiplier)),
        "def": int(max(0, base_def * multiplier)),
        "exp": int(spec.get("exp", 0) * multiplier),
        "gold": int(spec.get("gold", 0) * multiplier),
    }

# Basic auto-save thread to persist periodically
_AUTO_SAVE_INTERVAL = 60  # seconds
_auto_save_thread = None
_auto_save_running = False

def _auto_save_loop(player_getter, interval=_AUTO_SAVE_INTERVAL):
    global _auto_save_running
    _auto_save_running = True
    while _auto_save_running:
        time.sleep(interval)
        try:
            pl = player_getter()
            if pl:
                save_game(pl)
        except Exception:
            pass

def start_auto_save(player_getter, interval=_AUTO_SAVE_INTERVAL):
    global _auto_save_thread
    if _auto_save_thread and _auto_save_thread.is_alive():
        return
    _auto_save_thread = threading.Thread(target=_auto_save_loop, args=(player_getter, interval), daemon=True)
    _auto_save_thread.start()

def stop_auto_save():
    global _auto_save_running
    _auto_save_running = False

atexit.register(stop_auto_save)

class Item:
    def __init__(self, name, itype, power=0, heal=0, price=0):
        self.name = name
        self.type = itype  # "weapon", "armor", "consumable", "quest"
        self.power = power
        self.heal = heal
        self.price = price

    def to_dict(self):
        return {"name": self.name, "type": self.type, "power": self.power, "heal": self.heal, "price": self.price}

    @staticmethod
    def from_dict(d):
        return Item(d["name"], d["type"], d.get("power", 0), d.get("heal", 0), d.get("price", 0))


class Quest:
    def __init__(self, qid, title, description, qtype, target_name, target_count, level_req=1, reward_gold=0, reward_exp=0, reward_items=None):
        self.id = qid
        self.title = title
        self.description = description
        self.type = qtype  # "kill" (currently supported)
        self.target_name = target_name  # enemy name for kill quests
        self.target_count = target_count
        self.progress = 0
        self.level_req = level_req
        self.reward_gold = reward_gold
        self.reward_exp = reward_exp
        self.reward_items = reward_items or []
        self.completed = False
        self.accepted = False

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "type": self.type,
            "target_name": self.target_name,
            "target_count": self.target_count,
            "progress": self.progress,
            "level_req": self.level_req,
            "reward_gold": self.reward_gold,
            "reward_exp": self.reward_exp,
            "reward_items": [it.to_dict() for it in self.reward_items],
            "completed": self.completed,
            "accepted": self.accepted,
        }

    @staticmethod
    def from_dict(d):
        q = Quest(d["id"], d["title"], d["description"], d["type"], d["target_name"], d["target_count"],
                  d.get("level_req", 1), d.get("reward_gold", 0), d.get("reward_exp", 0),
                  [Item.from_dict(it) for it in d.get("reward_items", [])])
        q.progress = d.get("progress", 0)
        q.completed = d.get("completed", False)
        q.accepted = d.get("accepted", False)
        return q


# --- Chain Quest: The Cursed Crown ---
# Each quest unlocks the next. Final quest triggers boss battle.
CHAIN_QUESTS = [
    {
        "id": "crown_1",
        "title": "Rumors in the Tavern",
        "description": "Investigate the outskirts of the village after hearing whispers of unnatural darkness.",
        "type": "investigate",
        "target_name": "Corrupted Wildlife",
        "target_count": 1,
        "level_req": 2,
        "reward_gold": 20,
        "reward_exp": 30,
        "reward_items": [],
        "story": "You find evidence of corrupted wildlife and a broken crown fragment."
    },
    {
        "id": "crown_2",
        "title": "The Scholar's Warning",
        "description": "Bring the crown fragment to the local scholar.",
        "type": "deliver",
        "target_name": "Crown Fragment",
        "target_count": 1,
        "level_req": 3,
        "reward_gold": 30,
        "reward_exp": 40,
        "reward_items": [],
        "story": "The scholar reveals the fragment belongs to the Crown of Blackthorne, once worn by a tyrant king who made a pact with a demon."
    },
    {
        "id": "crown_3",
        "title": "The Forgotten Catacombs",
        "description": "Explore the catacombs beneath the ruined chapel to recover the second fragment.",
        "type": "recover",
        "target_name": "Second Crown Fragment",
        "target_count": 1,
        "level_req": 4,
        "reward_gold": 40,
        "reward_exp": 60,
        "reward_items": [],
        "story": "You face traps, skeletal guardians, and the Undying Knight."
    },
    {
        "id": "crown_4",
        "title": "The Betrayal",
        "description": "Escort the scholar and fragments to Mount Eldric. Beware of betrayal.",
        "type": "escort",
        "target_name": "Scholar",
        "target_count": 1,
        "level_req": 5,
        "reward_gold": 50,
        "reward_exp": 80,
        "reward_items": [],
        "story": "Ambushed by mercenaries, the scholar is revealed as a cultist. The fragments are stolen."
    },
    {
        "id": "crown_5",
        "title": "The Cult's Ritual",
        "description": "Track the cult to their ritual site in the cursed woods.",
        "type": "defeat",
        "target_name": "Cultists",
        "target_count": 3,
        "level_req": 6,
        "reward_gold": 60,
        "reward_exp": 100,
        "reward_items": [],
        "story": "Fight through cultists and corrupted beasts. Arrive too late—the Crown is reforged."
    },
    {
        "id": "crown_final",
        "title": "The Return of King Blackthorne",
        "description": "Defeat King Blackthorne in the abandoned throne room.",
        "type": "boss",
        "target_name": "King Blackthorne",
        "target_count": 1,
        "level_req": 7,
        "reward_gold": 200,
        "reward_exp": 300,
        "reward_items": [Item("Crown Shard", "quest", price=0)],
        "story": "Defeating him shatters the Crown forever. The kingdom is freed."
    }
]


ENEMIES = [
    {"name": "Rat", "hp": 8, "atk": 2, "def": 0, "exp": 8, "gold": 3},
    {"name": "Bandit", "hp": 14, "atk": 4, "def": 1, "exp": 18, "gold": 8},
    {"name": "Wolf", "hp": 18, "atk": 6, "def": 2, "exp": 28, "gold": 12},
    {"name": "Goblin", "hp": 24, "atk": 8, "def": 3, "exp": 40, "gold": 18},
    {"name": "Orc", "hp": 36, "atk": 10, "def": 4, "exp": 60, "gold": 30},
    {"name": "Corrupted Wildlife", "hp": 20, "atk": 5, "def": 2, "exp": 25, "gold": 15},
]
# Add boss enemy after ENEMIES is defined
ENEMIES.append({
    "name": "King Blackthorne",
    "hp": 120,
    "atk": 18,
    "def": 8,
    "exp": 300,
    "gold": 200
})


# Helper to check and offer next chain quest
def offer_chain_quest(player):
    # Use persistent completed_chain_quests for tracking
    completed_ids = set(player.completed_chain_quests)
    # Mark completed chain quests if they haven't been already
    for q in player.active_quests:
        if q.id.startswith("crown_") and q.completed and q.id not in player.completed_chain_quests:
            player.completed_chain_quests.append(q.id)
            print(f"DEBUG: Added {q.id} to completed_chain_quests")
    
    # Find next quest not yet accepted or completed
    for qdata in CHAIN_QUESTS:
        if qdata["id"] not in completed_ids and not any(a.id == qdata["id"] for a in player.active_quests):
            # Only offer if previous quest is completed (or it's the first)
            idx = CHAIN_QUESTS.index(qdata)
            if idx == 0 or CHAIN_QUESTS[idx-1]["id"] in completed_ids:
                # Create a proper Quest object with the right reward items
                reward_items = []
                if qdata.get("reward_items"):
                    for item_data in qdata["reward_items"]:
                        if isinstance(item_data, Item):
                            reward_items.append(item_data)
                        else:
                            # This might be a dict description of an item
                            reward_items.append(Item(
                                item_data.get("name", "Unknown Item"),
                                item_data.get("type", "quest"),
                                power=item_data.get("power", 0),
                                heal=item_data.get("heal", 0),
                                price=item_data.get("price", 0)
                            ))
                
                quest = Quest(
                    qdata["id"], qdata["title"], qdata["description"], qdata["type"],
                    qdata["target_name"], qdata["target_count"], qdata["level_req"],
                    qdata["reward_gold"], qdata["reward_exp"], reward_items
                )
                quest.accepted = True
                player.active_quests.append(quest)
                print(f"Chain Quest Started: {quest.title}\n{qdata['story']}")
                return True
    return False


SHOP_ITEMS = [
    Item("Iron Sword", "weapon", power=6, price=30),
    Item("Steel Sword", "weapon", power=10, price=70),
    Item("Leather Armor", "armor", power=3, price=25),
    Item("Plate Armor", "armor", power=7, price=80),
    Item("Minor Potion", "consumable", heal=15, price=10),
    Item("Major Potion", "consumable", heal=40, price=40),
]


# Quest templates - simple kill quests
QUEST_TEMPLATES = [
    Quest(str(uuid.uuid4()), "Pest Problem", "Reduce the rat population. Kill 3 Rats.", "kill", "Rat", 3, level_req=1, reward_gold=15, reward_exp=20),
    Quest(str(uuid.uuid4()), "Bandit Trouble", "The road is unsafe. Kill 2 Bandits.", "kill", "Bandit", 2, level_req=2, reward_gold=30, reward_exp=45),
    Quest(str(uuid.uuid4()), "Goblin Menace", "Hunt the goblins troubling the outskirts. Kill 4 Goblins.", "kill", "Goblin", 4, level_req=3, reward_gold=80, reward_exp=120, reward_items=[Item("Minor Potion", "consumable", heal=15, price=10)]),
]


def save_game(player):
    data = player.to_dict()
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print("Game saved.")


def load_game():
    if not os.path.exists(SAVE_FILE):
        return None
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)
    return Player.from_dict(data)


def game_over(player):
    """Handle player death: attempt to save and exit cleanly."""
    print("\nYou have been defeated. Game over.")
    try:
        save_game(player)
    except Exception:
        # if saving fails, ignore and continue to exit
        pass
    print("Exiting.")
    sys.exit(0)


def update_quests_on_kill(player, enemy_name):
    messages = []
    for q in player.active_quests:
        # Accept both 'kill' and 'investigate' types for chain quests
        if q.completed or q.type.lower() not in ("kill", "investigate"):
            continue
        if q.target_name.lower() == enemy_name.lower():
            q.progress += 1
            messages.append(f"Progressed quest '{q.title}': {q.progress}/{q.target_count}")
            if q.progress >= q.target_count:
                q.completed = True
                # For chain quests, mark as completed right away
                if q.id.startswith("crown_") and q.id not in player.completed_chain_quests:
                    player.completed_chain_quests.append(q.id)
                    
                    # Add quest-specific items for chain quests
                    if q.id == "crown_1":
                        crown_fragment = Item("Crown Fragment", "quest", price=0)
                        player.inventory.append(crown_fragment)
                        messages.append(f"You found a Crown Fragment!")
                    elif q.id == "crown_3":
                        second_fragment = Item("Second Crown Fragment", "quest", price=0)
                        player.inventory.append(second_fragment)
                        messages.append(f"You found a Second Crown Fragment!")
                    
                    # Try to offer next chain quest immediately
                    try:
                        offer_chain_quest(player)
                    except Exception as e:
                        print(f"Error offering next chain quest: {e}")
                messages.append(f"Quest '{q.title}' completed! Visit the quest board to claim rewards.")
    return messages


def encounter_enemy(player):
    # Choose enemy scaled to player level
    spec = random.choice(ENEMIES)
    scaled = scale_enemy_spec(spec, player.level)
    enemy = Actor(scaled["name"], scaled["hp"], scaled["atk"], scaled["def"])
    combat(player, enemy, scaled.get("exp", 0), scaled.get("gold", 0))


def combat(player, enemy, exp_reward, gold_reward):
    while player.is_alive() and enemy.is_alive():
        print(f"\nPlayer HP: {player.hp}/{player.max_hp}  |  {enemy.name} HP: {enemy.hp}")
        print("Choose action: (a)ttack  (s)kill(skill)  (d)efend  (u)se item  (f)lee")
        choice = input("> ").strip().lower()
        if choice in ("a", "attack"):
            dmg = max(1, player.attack_value() - enemy.defense)
            dealt = enemy.take_damage(dmg)
            print(f"You strike {enemy.name} for {dealt} damage.")
        elif choice in ("s", "skill"):
            if not player.skills:
                print("You have no skills to use!")
                continue
            for i, skill in enumerate(player.skills, 1):
                print(f"{i}. {skill.name} ({skill.description}) - Power: {skill.power}, Heal: {skill.heal}")
            try:
                sel = int(input("Choose skill # (0 to cancel): ").strip())
                if sel == 0 or sel > len(player.skills):
                    continue
                skill = player.skills[sel-1]
                # Damage/attack skill
                if skill.power > 0:
                    dmg = max(1, player.attack_value() + skill.power - enemy.defense)
                    dealt = enemy.take_damage(dmg)
                    print(f"Using {skill.name}, you deal {dealt} damage to {enemy.name}!")
                # Healing skill
                if skill.heal > 0:
                    heal = min(skill.heal, player.max_hp - player.hp)
                    player.hp += heal
                    print(f"Using {skill.name}, you heal for {heal} HP.")
            except Exception:
                print("Invalid selection.")
        elif choice in ("d", "defend"):
            bonus = player.defense_value() // 2
            enemy_dmg = max(1, enemy.atk - (player.defense_value() + bonus))
            player.hp -= enemy_dmg
            print(f"You defend, reducing damage. {enemy.name} hits for {enemy_dmg}.")
        elif choice in ("u", "use"):
            consumables = [it for it in player.inventory if it.type == "consumable"]
            if not consumables:
                print("You have no usable items!")
                continue
            for i, it in enumerate(consumables, 1):
                print(f"{i}. {it.name} (Heal: {it.heal})")
            try:
                sel = int(input("Choose item # (0 to cancel): ").strip())
                if sel == 0 or sel > len(consumables):
                    continue
                item = consumables[sel-1]
                if item.heal > 0:
                    heal = min(item.heal, player.max_hp - player.hp)
                    player.hp += heal
                    player.inventory.remove(item)
                    print(f"Used {item.name} and healed for {heal} HP.")
                else:
                    print(f"Cannot use {item.name} in combat.")
            except Exception:
                print("Invalid selection.")
        elif choice in ("f", "flee"):
            if random.random() < 0.6:  # 60% chance to succeed
                print(f"You successfully fled from the {enemy.name}!")
                return
            else:
                print(f"Failed to escape! The {enemy.name} blocks your path.")
        else:
            print("Invalid command.")
            continue

        # Enemy's turn (only if still alive)
        if enemy.is_alive():
            enemy_dmg = max(1, enemy.atk - player.defense_value())
            player.hp -= enemy_dmg
            print(f"{enemy.name} attacks for {enemy_dmg} damage.")

        # Check if player died
        if not player.is_alive():
            game_over(player)

    # If we got here, enemy was defeated
    if not enemy.is_alive():
        print(f"\nYou defeated the {enemy.name}!")
        print(f"Gained {exp_reward} EXP and {gold_reward} gold.")
        player.gold += gold_reward
        
        # Check for level up
        if player.gain_exp(exp_reward):
            print(f"Level up! You are now level {player.level}.")
            print(f"Max HP increased to {player.max_hp}.")
            
        # Check for quest progress
        messages = update_quests_on_kill(player, enemy.name)
        for msg in messages:
            print(msg)


def explore(player):
    print("\nExploring...")
    time.sleep(0.5)  # short delay for effect

    # Chance for random world event
    event_result = random_world_event(player)
    if event_result:
        print(event_result)
        return

    # Chance for enemy encounter
    if random.random() < 0.7:  # 70% chance
        print("You encountered an enemy!")
        encounter_enemy(player)
    else:
        print("You found nothing of interest.")


def shop(player):
    while True:
        print("\n--- Shop ---")
        for i, item in enumerate(SHOP_ITEMS, 1):
            print(f"{i}. {item.name} ({item.type}) - {item.price} gold")
        print("b) Back")
        
        choice = input("> ").strip().lower()
        if choice == "b" or not choice:
            return
        
        try:
            idx = int(choice) - 1
            item = SHOP_ITEMS[idx]
        except Exception:
            print("Invalid selection.")
            continue
            
        if player.gold < item.price:
            print("You cannot afford that.")
            continue
            
        player.gold -= item.price
        player.inventory.append(Item(item.name, item.type, item.power, item.heal, item.price))
        print(f"Purchased {item.name}.")


def show_inventory(player):
    while True:
        print("\n--- Inventory ---")
        print(f"Gold: {player.gold}")
        if not player.inventory:
            print("Your inventory is empty.")
            return

        # Group by type for better organization
        weapons = [it for it in player.inventory if it.type == "weapon"]
        armors = [it for it in player.inventory if it.type == "armor"]
        consumables = [it for it in player.inventory if it.type == "consumable"]
        quest_items = [it for it in player.inventory if it.type == "quest"]
        
        print("\nWeapons:")
        for i, it in enumerate(weapons, 1):
            equipped = " (Equipped)" if player.weapon and player.weapon.name == it.name else ""
            print(f"{i}. {it.name} - Power: {it.power}{equipped}")
            
        print("\nArmor:")
        for i, it in enumerate(armors, 1):
            equipped = " (Equipped)" if player.armor and player.armor.name == it.name else ""
            print(f"{i+len(weapons)}. {it.name} - Defense: {it.power}{equipped}")
            
        print("\nConsumables:")
        for i, it in enumerate(consumables, 1):
            print(f"{i+len(weapons)+len(armors)}. {it.name} - Heal: {it.heal}")
            
        print("\nQuest Items:")
        for i, it in enumerate(quest_items, 1):
            print(f"{i+len(weapons)+len(armors)+len(consumables)}. {it.name}")
        
        print("\nOptions: (e)quip <item name>  (u)se <item name>  (b)ack")
        choice = input("> ").strip().lower()
        
        if choice == "b" or not choice:
            return
            
        if choice.startswith("e "):
            item_name = choice[2:].strip()
            result = player.equip(item_name)
            print(result)
        elif choice.startswith("u "):
            item_name = choice[2:].strip()
            # Find the item
            for it in player.inventory:
                if it.name.lower() == item_name.lower():
                    if it.type == "consumable":
                        if player.hp < player.max_hp:
                            heal = min(it.heal, player.max_hp - player.hp)
                            player.hp += heal
                            player.inventory.remove(it)
                            print(f"Used {it.name} and healed for {heal} HP.")
                        else:
                            print("You are already at full health.")
                    else:
                        print(f"Cannot use {it.name} here.")
                    break
            else:
                print("Item not found.")


def show_quests(player):
    if not player.active_quests:
        print("\nYou have no active quests.")
        return
        
    while True:
        print("\n--- Active Quests ---")
        for i, quest in enumerate(player.active_quests, 1):
            status = "Completed" if quest.completed else f"{quest.progress}/{quest.target_count}"
            print(f"{i}. {quest.title} - {status}")
            print(f"   {quest.description}")
            if quest.completed:
                print("   Ready to turn in for rewards.")
        
        print("\nOptions: (d)etails <#>  (t)urn in <#>  (b)ack")
        choice = input("> ").strip().lower()
        
        if choice == "b" or not choice:
            return
            
        if choice.startswith("d "):
            try:
                idx = int(choice[2:].strip()) - 1
                quest = player.active_quests[idx]
                print(f"\nQuest: {quest.title}")
                print(f"Description: {quest.description}")
                print(f"Type: {quest.type.capitalize()}")
                print(f"Target: {quest.target_name} x{quest.target_count}")
                print(f"Progress: {quest.progress}/{quest.target_count}")
                print(f"Required Level: {quest.level_req}")
                print("Rewards:")
                print(f"- {quest.reward_gold} gold")
                print(f"- {quest.reward_exp} experience")
                if quest.reward_items:
                    for item in quest.reward_items:
                        print(f"- {item.name}")
            except (ValueError, IndexError):
                print("Invalid quest number.")
                
        elif choice.startswith("t "):
            try:
                idx = int(choice[2:].strip()) - 1
                quest = player.active_quests[idx]
                if not quest.completed:
                    print("This quest is not yet completed.")
                    continue
                    
                # Give rewards
                player.gold += quest.reward_gold
                leveled = player.gain_exp(quest.reward_exp)
                
                # Add reward items to inventory
                for item in quest.reward_items:
                    player.inventory.append(item)
                
                # Remove quest from active quests
                player.active_quests.remove(quest)
                
                print(f"Quest '{quest.title}' turned in!")
                print(f"Received: {quest.reward_gold} gold, {quest.reward_exp} EXP")
                for item in quest.reward_items:
                    print(f"Received: {item.name}")
                
                if leveled:
                    print(f"Level up! You are now level {player.level}.")
                    print(f"Max HP increased to {player.max_hp}.")
                
                # For chain quests, immediately try to offer the next one
                if quest.id.startswith("crown_") and quest.id not in player.completed_chain_quests:
                    player.completed_chain_quests.append(quest.id)
                    offer_chain_quest(player)
                
            except (ValueError, IndexError):
                print("Invalid quest number.")


def quest_board(player):
    # Simple 1-3 random quests based on level
    available = []
    
    # Try to offer chain quest if none active
    if not any(q.id.startswith("crown_") for q in player.active_quests):
        offer_chain_quest(player)
    
    # Offer 1-3 random template quests
    eligible = [q for q in QUEST_TEMPLATES if q.level_req <= player.level and 
                not any(a.title == q.title for a in player.active_quests)]
    
    if eligible:
        count = min(3, len(eligible))
        available = random.sample(eligible, count)
    
    if not available:
        print("\nNo quests available on the board at your level.")
        return
    
    while True:
        print("\n--- Quest Board ---")
        for i, quest in enumerate(available, 1):
            print(f"{i}. {quest.title} (Level {quest.level_req})")
            print(f"   {quest.description}")
        
        print("\nOptions: (a)ccept <#>  (b)ack")
        choice = input("> ").strip().lower()
        
        if choice == "b" or not choice:
            return
            
        if choice.startswith("a "):
            try:
                idx = int(choice[2:].strip()) - 1
                quest = available[idx]
                
                # Create a copy of the quest with a unique ID
                new_quest = Quest(
                    str(uuid.uuid4()), quest.title, quest.description, quest.type,
                    quest.target_name, quest.target_count, quest.level_req,
                    quest.reward_gold, quest.reward_exp,
                    [Item.from_dict(it.to_dict()) for it in quest.reward_items]
                )
                new_quest.accepted = True
                
                player.active_quests.append(new_quest)
                available.pop(idx)
                
                print(f"Accepted quest: {new_quest.title}")
                
            except (ValueError, IndexError):
                print("Invalid quest number.")


def crafting_menu(player):
    print("\n--- Crafting ---")
    
    # Show available recipes
    print("Available Recipes:")
    recipes = list(CRAFTING_RECIPES.keys())
    
    for i, recipe_name in enumerate(recipes, 1):
        ingredients, attrs = CRAFTING_RECIPES[recipe_name]
        ingredient_list = ", ".join([f"{n} x{c}" for n, c in ingredients.items()])
        print(f"{i}. {recipe_name} - Requires: {ingredient_list}")
    
    print("b) Back")
    choice = input("> ").strip().lower()
    
    if choice == "b" or not choice:
        return
        
    try:
        idx = int(choice) - 1
        recipe_name = recipes[idx]
        result = craft_item(player, recipe_name)
        print(result)
    except (ValueError, IndexError):
        print("Invalid recipe number.")


def town(player):
    while True:
        print("\n--- Town ---")
        print("1. Shop")
        print("2. Inn (Rest and Heal)")
        print("3. Quest Board")
        print("4. Crafting")
        print("b. Back to Main")
        
        choice = input("> ").strip().lower()
        
        if choice == "1":
            shop(player)
        elif choice == "2":
            cost = 5
            if player.gold < cost:
                print(f"You need {cost} gold to rest at the inn.")
            else:
                player.gold -= cost
                player.hp = player.max_hp
                print(f"You rest at the inn and restore your health to {player.hp}.")
        elif choice == "3":
            quest_board(player)
        elif choice == "4":
            crafting_menu(player)
        elif choice == "b":
            return
        else:
            print("Invalid choice.")


def show_status(player):
    print("\n--- Character Status ---")
    print(f"Name: {player.name}")
    print(f"Level: {player.level}")
    print(f"EXP: {player.exp}/{player.level*50}")
    print(f"HP: {player.hp}/{player.max_hp}")
    print(f"Attack: {player.atk}")
    print(f"Defense: {player.defense}")
    print(f"Gold: {player.gold}")
    
    print("\nEquipment:")
    print(f"Weapon: {player.weapon.name if player.weapon else 'None'}")
    if player.weapon:
        print(f"  Power: +{player.weapon.power}")
    print(f"Armor: {player.armor.name if player.armor else 'None'}")
    if player.armor:
        print(f"  Defense: +{player.armor.power}")
    
    print("\nSkills:")
    for skill in player.skills:
        print(f"- {skill.name}: {skill.description}")
    
    print("\nActive Quests:")
    if not player.active_quests:
        print("None")
    else:
        for quest in player.active_quests:
            status = "Completed" if quest.completed else f"{quest.progress}/{quest.target_count}"
            print(f"- {quest.title} ({status})")


def main():
    print("\n=== Welcome to the RPG Game ===\n")
    
    # Try to load a saved game, or start a new one
    player = load_game()
    
    if not player:
        name = input("Enter your character's name: ").strip()
        if not name:
            name = "Adventurer"
        player = Player(name)
        print(f"\nWelcome, {player.name}! Your adventure begins...")
    else:
        print(f"\nWelcome back, {player.name}!")
    
    # Start auto-save in background
    start_auto_save(lambda: player)
    
    # Offer first chain quest to returning players if they don't have one
    if player and not any(q.id.startswith("crown_") for q in player.active_quests):
        offer_chain_quest(player)
    
    while True:
        print("\n=== Main Menu ===")
        print(f"HP: {player.hp}/{player.max_hp}  |  Level: {player.level}  |  Gold: {player.gold}")
        print("1. Explore")
        print("2. Town")
        print("3. Inventory")
        print("4. Quests")
        print("5. Status")
        print("6. Save Game")
        print("7. Quit")
        
        choice = input("\n> ").strip()
        
        if choice == "1":
            explore(player)
        elif choice == "2":
            town(player)
        elif choice == "3":
            show_inventory(player)
        elif choice == "4":
            show_quests(player)
        elif choice == "5":
            show_status(player)
        elif choice == "6":
            save_game(player)
        elif choice == "7":
            save_game(player)
            print("Thanks for playing! Goodbye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
