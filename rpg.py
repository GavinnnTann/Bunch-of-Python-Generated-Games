import random
import json
import os
import sys
import uuid

#!/usr/bin/env python3
# rpg.py - simple text-based medieval RPG with basic quest system

SAVE_FILE = "rpg_save.json"


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


class Actor:
    def __init__(self, name, hp, atk, defense):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.atk = atk
        self.defense = defense

    def is_alive(self):
        return self.hp > 0

    def take_damage(self, amount):
        dmg = max(1, amount)
        self.hp = max(0, self.hp - dmg)
        return dmg


class Player(Actor):
    def __init__(self, name):
        super().__init__(name, hp=30, atk=5, defense=2)
        self.level = 1
        self.exp = 0
        self.gold = 20
        self.inventory = [
            Item("Rusty Sword", "weapon", power=3, price=0),
            Item("Cloth Armor", "armor", power=1, price=0),
            Item("Minor Potion", "consumable", heal=15, price=10),
        ]
        self.weapon = next((i for i in self.inventory if i.type == "weapon"), None)
        self.armor = next((i for i in self.inventory if i.type == "armor"), None)
        self.active_quests = []

    def equip(self, item_name):
        for it in self.inventory:
            if it.name.lower() == item_name.lower():
                if it.type == "weapon":
                    self.weapon = it
                    return f"Equipped {it.name} as weapon."
                if it.type == "armor":
                    self.armor = it
                    return f"Equipped {it.name} as armor."
                return "Cannot equip that item."
        return "Item not found."

    def attack_value(self):
        return self.atk + (self.weapon.power if self.weapon else 0)

    def defense_value(self):
        return self.defense + (self.armor.power if self.armor else 0)

    def gain_exp(self, amount):
        self.exp += amount
        leveled = False
        while self.exp >= self.level * 50:
            self.exp -= self.level * 50
            self.level += 1
            self.max_hp += 5
            self.atk += 1
            self.defense += 1
            self.hp = self.max_hp
            leveled = True
        return leveled

    def to_dict(self):
        return {
            "name": self.name,
            "max_hp": self.max_hp,
            "hp": self.hp,
            "atk": self.atk,
            "defense": self.defense,
            "level": self.level,
            "exp": self.exp,
            "gold": self.gold,
            "inventory": [it.to_dict() for it in self.inventory],
            "weapon": self.weapon.name if self.weapon else None,
            "armor": self.armor.name if self.armor else None,
            "active_quests": [q.to_dict() for q in self.active_quests],
        }

    @staticmethod
    def from_dict(d):
        p = Player(d["name"])
        p.max_hp = d["max_hp"]
        p.hp = d["hp"]
        p.atk = d["atk"]
        p.defense = d["defense"]
        p.level = d["level"]
        p.exp = d["exp"]
        p.gold = d["gold"]
        p.inventory = [Item.from_dict(it) for it in d.get("inventory", [])]
        p.weapon = next((i for i in p.inventory if i.name == d.get("weapon")), None)
        p.armor = next((i for i in p.inventory if i.name == d.get("armor")), None)
        p.active_quests = [Quest.from_dict(q) for q in d.get("active_quests", [])]
        return p


ENEMIES = [
    {"name": "Rat", "hp": 8, "atk": 2, "def": 0, "exp": 8, "gold": 3},
    {"name": "Bandit", "hp": 14, "atk": 4, "def": 1, "exp": 18, "gold": 8},
    {"name": "Wolf", "hp": 18, "atk": 6, "def": 2, "exp": 28, "gold": 12},
    {"name": "Goblin", "hp": 24, "atk": 8, "def": 3, "exp": 40, "gold": 18},
    {"name": "Orc", "hp": 36, "atk": 10, "def": 4, "exp": 60, "gold": 30},
]


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
        json.dump(data, f)
    print("Game saved.")


def load_game():
    if not os.path.exists(SAVE_FILE):
        return None
    with open(SAVE_FILE, "r") as f:
        data = json.load(f)
    return Player.from_dict(data)


def encounter_enemy(player):
    # Choose enemy scaled to player level
    idx = min(len(ENEMIES) - 1, max(0, player.level - 1 + random.randint(-1, 1)))
    spec = ENEMIES[idx]
    enemy = Actor(spec["name"], spec["hp"] + player.level * 2, spec["atk"] + player.level // 2, spec["def"])
    exp_reward = spec["exp"] + player.level * 4
    gold_reward = spec["gold"] + player.level * 2
    print(f"A wild {enemy.name} appears! (HP: {enemy.hp})")
    combat(player, enemy, exp_reward, gold_reward)


def combat(player, enemy, exp_reward, gold_reward):
    while player.is_alive() and enemy.is_alive():
        print(f"\nPlayer HP: {player.hp}/{player.max_hp}  |  {enemy.name} HP: {enemy.hp}")
        print("Choose action: (a)ttack  (d)efend  (u)se item  (f)lee")
        choice = input("> ").strip().lower()
        if choice in ("a", "attack"):
            dmg = max(1, player.attack_value() - enemy.defense)
            dealt = enemy.take_damage(dmg)
            print(f"You strike {enemy.name} for {dealt} damage.")
        elif choice in ("d", "defend"):
            print("You brace for the incoming attack. Damage reduced this turn.")
        elif choice in ("u", "use", "use item"):
            used = use_item_combat(player)
            if not used:
                print("No usable items or canceled.")
                continue
        elif choice in ("f", "flee"):
            if random.random() < 0.5:
                print("You fled successfully.")
                return
            else:
                print("Failed to flee!")
        else:
            print("Unknown action.")
            continue

        # Enemy turn if still alive
        if enemy.is_alive():
            if choice in ("d", "defend"):
                enemy_atk = max(1, enemy.atk - player.defense_value() - 2)
            else:
                enemy_atk = max(1, enemy.atk - player.defense_value())
            damage = player.take_damage(enemy_atk)
            print(f"{enemy.name} hits you for {damage} damage.")

    if player.is_alive():
        print(f"You defeated {enemy.name}!")
        player.gold += gold_reward
        leveled = player.gain_exp(exp_reward)
        print(f"Found {gold_reward} gold and gained {exp_reward} EXP.")
        # update quests (kill type)
        updated = update_quests_on_kill(player, enemy.name)
        if updated:
            for msg in updated:
                print(msg)
        if leveled:
            print(f"You leveled up! You are now level {player.level}.")
        # small loot chance
        if random.random() < 0.25:
            loot = random.choice(SHOP_ITEMS)
            player.inventory.append(loot)
            print(f"You looted a {loot.name}.")
    else:
        print("You have been defeated...")
        game_over(player)


def update_quests_on_kill(player, enemy_name):
    messages = []
    for q in player.active_quests:
        if q.type == "kill" and not q.completed and q.target_name == enemy_name:
            q.progress += 1
            messages.append(f"Quest update: {q.title} ({q.progress}/{q.target_count})")
            if q.progress >= q.target_count:
                q.completed = True
                messages.append(f"Quest completed: {q.title}! Return to the quest board to claim your reward.")
    return messages


def use_item_combat(player):
    consumables = [it for it in player.inventory if it.type == "consumable"]
    if not consumables:
        return False
    print("Consumables:")
    for i, it in enumerate(consumables, 1):
        print(f"{i}. {it.name} heal {it.heal}")
    print("Choose item number or press Enter to cancel:")
    choice = input("> ").strip()
    if not choice:
        return False
    try:
        idx = int(choice) - 1
        it = consumables[idx]
    except Exception:
        print("Invalid choice.")
        return False
    player.hp = min(player.max_hp, player.hp + it.heal)
    player.inventory.remove(it)
    print(f"You used {it.name}, healed {it.heal} HP.")
    return True


def shop(player):
    print("\n--- Marketplace ---")
    for i, it in enumerate(SHOP_ITEMS, 1):
        print(f"{i}. {it.name} ({it.type}) - {it.price} gold")
    print("b) Back")
    choice = input("> ").strip().lower()
    if choice == "b":
        return
    try:
        idx = int(choice) - 1
        it = SHOP_ITEMS[idx]
    except Exception:
        print("Invalid choice.")
        return
    if player.gold < it.price:
        print("You cannot afford that.")
        return
    player.gold -= it.price
    player.inventory.append(Item(it.name, it.type, it.power, it.heal, it.price))
    print(f"Purchased {it.name}.")


def rest(player):
    cost = max(1, player.level * 2)
    print(f"Rest at the inn for {cost} gold? (y/n)")
    if input("> ").strip().lower().startswith("y"):
        if player.gold >= cost:
            player.gold -= cost
            player.hp = player.max_hp
            print("You feel refreshed.")
        else:
            print("Not enough gold.")


def view_stats(player):
    print(f"\n{player.name} - Level {player.level}  EXP {player.exp}/{player.level*50}")
    print(f"HP: {player.hp}/{player.max_hp}  ATK: {player.attack_value()}  DEF: {player.defense_value()}  Gold: {player.gold}")
    print(f"Weapon: {player.weapon.name if player.weapon else 'None'}  Armor: {player.armor.name if player.armor else 'None'}")
    if player.active_quests:
        print("\nActive Quests:")
        for q in player.active_quests:
            status = "Completed" if q.completed else f"In progress ({q.progress}/{q.target_count})"
            print(f"- {q.title}: {status}")


def inventory_menu(player):
    print("\nInventory:")
    for i, it in enumerate(player.inventory, 1):
        if it.type == "consumable":
            desc = f"heal {it.heal}"
        else:
            desc = f"power {it.power}"
        print(f"{i}. {it.name} ({it.type}) - {desc}")
    print("Options: (e)quip <num>  (u)se <num>  (d)rop <num>  (b)ack")
    cmd = input("> ").strip().lower().split()
    if not cmd:
        return
    action = cmd[0]
    if action in ("b", "back"):
        return
    if len(cmd) < 2:
        print("Specify item number.")
        return
    try:
        idx = int(cmd[1]) - 1
        it = player.inventory[idx]
    except Exception:
        print("Invalid item number.")
        return
    if action in ("e", "equip"):
        print(player.equip(it.name))
    elif action in ("u", "use"):
        if it.type == "consumable":
            player.hp = min(player.max_hp, player.hp + it.heal)
            player.inventory.pop(idx)
            print(f"Used {it.name}, healed {it.heal} HP.")
        else:
            print("Cannot use that now.")
    elif action in ("d", "drop"):
        player.inventory.pop(idx)
        print(f"Dropped {it.name}.")


def explore(player):
    r = random.random()
    if r < 0.6:
        encounter_enemy(player)
    elif r < 0.8:
        # treasure
        gold = random.randint(5, 25) + player.level * 2
        player.gold += gold
        print(f"You found a hidden stash with {gold} gold!")
    else:
        # find merchant
        print("You meet a traveling merchant.")
        shop(player)


def game_over(player):
    print("Game over. You can restart or load a save.")
    if os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)
    sys.exit(0)


def quest_board(player):
    print("\n--- Quest Board ---")
    print("1) View available quests  2) View active quests  3) Claim completed quest rewards  b) Back")
    choice = input("> ").strip().lower()
    if choice == "b":
        return
    if choice == "1":
        list_available = [t for t in QUEST_TEMPLATES if t.level_req <= player.level and not any(a.title == t.title for a in player.active_quests)]
        if not list_available:
            print("No quests available right now.")
            return
        for i, t in enumerate(list_available, 1):
            print(f"{i}. {t.title} (Lvl {t.level_req}) - {t.description} Reward: {t.reward_gold}g, {t.reward_exp}xp")
        print("Choose quest number to accept or press Enter to cancel:")
        sel = input("> ").strip()
        if not sel:
            return
        try:
            idx = int(sel) - 1
            template = list_available[idx]
        except Exception:
            print("Invalid choice.")
            return
        # make a copy with new id
        newq = Quest(str(uuid.uuid4()), template.title, template.description, template.type, template.target_name, template.target_count, template.level_req, template.reward_gold, template.reward_exp, template.reward_items)
        newq.accepted = True
        player.active_quests.append(newq)
        print(f"Accepted quest: {newq.title}")
    elif choice == "2":
        if not player.active_quests:
            print("You have no active quests.")
            return
        for i, q in enumerate(player.active_quests, 1):
            status = "Completed" if q.completed else f"{q.progress}/{q.target_count}"
            print(f"{i}. {q.title} - {status}")
    elif choice == "3":
        completed = [q for q in player.active_quests if q.completed]
        if not completed:
            print("No completed quests to claim.")
            return
        for q in completed:
            player.gold += q.reward_gold
            leveled = player.gain_exp(q.reward_exp)
            for it in q.reward_items:
                player.inventory.append(it)
            print(f"Claimed {q.title}: +{q.reward_gold} gold, +{q.reward_exp} EXP.")
            if leveled:
                print(f"You leveled up! You are now level {player.level}.")
            player.active_quests.remove(q)
    else:
        print("Invalid selection.")


def main():
    print("Welcome to the Medieval Text RPG")
    player = load_game()
    if player:
        print(f"Loaded saved game for {player.name}.")
    else:
        name = input("Enter your character name: ").strip() or "Adventurer"
        player = Player(name)
        print(f"Good luck, {player.name}.")

    while True:
        if not player.is_alive():
            game_over(player)
        print("\n--- Main Menu ---")
        print("1) Explore  2) Rest  3) Town (Shop/Inn/Quests)  4) Inventory  5) Stats  6) Save  7) Quit")
        choice = input("> ").strip()
        if choice == "1":
            explore(player)
        elif choice == "2":
            rest(player)
        elif choice == "3":
            print("\n--- Town ---")
            print("1) Marketplace  2) Inn  3) Quest Board  b) Back")
            t = input("> ").strip().lower()
            if t == "1":
                shop(player)
            elif t == "2":
                rest(player)
            elif t == "3":
                quest_board(player)
            else:
                continue
        elif choice == "4":
            inventory_menu(player)
        elif choice == "5":
            view_stats(player)
        elif choice == "6":
            save_game(player)
        elif choice == "7":
            print("Quit and save? (y/n)")
            if input("> ").strip().lower().startswith("y"):
                save_game(player)
            print("Farewell.")
            break
        else:
            print("Invalid selection.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting game.")
        sys.exit(0)