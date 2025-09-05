# player.py
from enum import Enum
import uuid
from .item import Item

class Difficulty(Enum):
    EASY = 0.8
    NORMAL = 1.0
    HARD = 1.25

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
        self.skills = [
            Skill("slash", "Slash", "A basic slashing attack.", mp_cost=0, power=3, heal=0),
            Skill("heal", "Cauterize", "Small self-heal to patch wounds.", mp_cost=0, power=0, heal=10),
        ]

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
            "skills": [s.to_dict() for s in self.skills],
        }

    @staticmethod
    def from_dict(d):
        p = Player(d["name"])
        p.max_hp = d.get("max_hp", p.max_hp)
        p.hp = max(1, d.get("hp", p.hp))
        p.atk = d.get("atk", p.atk)
        p.defense = d.get("defense", p.defense)
        p.level = d.get("level", p.level)
        p.exp = d.get("exp", p.exp)
        p.gold = d.get("gold", p.gold)
        p.inventory = [Item.from_dict(it) for it in d.get("inventory", [])]
        weapon_name = d.get("weapon")
        armor_name = d.get("armor")
        p.weapon = next((i for i in p.inventory if i.name == weapon_name), None)
        if weapon_name and not p.weapon:
            # If not found, create and add the item
            p.weapon = Item(weapon_name, "weapon")
            p.inventory.append(p.weapon)
        p.armor = next((i for i in p.inventory if i.name == armor_name), None)
        if armor_name and not p.armor:
            p.armor = Item(armor_name, "armor")
            p.inventory.append(p.armor)
        p.active_quests = [q.from_dict(q) for q in d.get("active_quests", [])]
        p.skills = [Skill.from_dict(s) for s in d.get("skills", [])] if d.get("skills") is not None else p.skills
        return p
