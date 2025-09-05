# quest.py
import uuid
from .item import Item
import uuid

class Quest:
    def __init__(self, qid, title, description, qtype, target_name, target_count, level_req=1, reward_gold=0, reward_exp=0, reward_items=None):
        self.id = qid
        self.title = title
        self.description = description
        self.type = qtype
        self.target_name = target_name
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
