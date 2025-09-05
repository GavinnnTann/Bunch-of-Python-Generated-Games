# npc.py
import random
class NPC:
    def __init__(self, name, role, dialog=None, inventory=None):
        self.name = name
        self.role = role
        self.dialog = dialog or []
        self.inventory = inventory or []

    def talk(self):
        if not self.dialog:
            return f"{self.name} has nothing to say."
        return self.dialog[random.randint(0, len(self.dialog)-1)]
