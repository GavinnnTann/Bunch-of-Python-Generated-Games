# item.py
class Item:
    def __init__(self, name, itype, power=0, heal=0, price=0):
        self.name = name
        self.type = itype
        self.power = power
        self.heal = heal
        self.price = price

    def to_dict(self):
        return {"name": self.name, "type": self.type, "power": self.power, "heal": self.heal, "price": self.price}

    @staticmethod
    def from_dict(d):
        return Item(d["name"], d["type"], d.get("power", 0), d.get("heal", 0), d.get("price", 0))
