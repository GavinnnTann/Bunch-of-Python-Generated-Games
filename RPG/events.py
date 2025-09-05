
import random
import uuid
from RPG.item import Item
from RPG.npc import NPC
from RPG.quest import Quest

def random_world_event(player, TOWN_NPCS, QUEST_TEMPLATES, SHOP_ITEMS, _merchant_shopping_session=None):
	r = random.random()
	# rare traveling merchant (very uncommon)
	if r < 0.04:
		merchant = _generate_traveling_merchant()
		if _merchant_shopping_session:
			_merchant_shopping_session(player, merchant)
		return f"You encountered {merchant.name} and browsed their wares."
	# trap / loss (slightly reduced chance to leave room for merchants)
	if r < 0.09:
		loss = min(player.gold, random.randint(5, 20))
		player.gold -= loss
		return f"You fell into a shallow pit and lost {loss} gold while recovering."
	# meet town NPC (chance to meet merchants/trainers/questgivers)
	if r < 0.18:
		# Expanded NPC pool
		expanded_npcs = TOWN_NPCS + [
			NPC("Mira the Collector", "merchant", dialog=["I trade in rare curiosities. Interested in a swap?"], inventory=[Item("Crystal Shard", "quest", price=50), Item("Ancient Coin", "quest", price=80)]),
			NPC("Tharn the Blacksmith", "merchant", dialog=["My wares are forged for heroes.", "Trade me something rare for a discount!"], inventory=[Item("Reinforced Mail", "armor", power=9, price=135)]),
			NPC("Seren the Sage", "trainer", dialog=["Wisdom is earned, not given.", "Complete my trial for true power."]),
			NPC("Ryn the Adventurer", "questgiver", dialog=["I need help with a dangerous task.", "Are you brave enough?"])
		]
		npc = random.choice(expanded_npcs)
		print(f"\nYou encounter {npc.name} ({npc.role}).")
		if getattr(npc, "dialog", None):
			print(f"{npc.name} says: \"{npc.talk()}\"")
		while True:
			print("Options: (t)alk  (l)eave", end="")
			if npc.role == "merchant":
				print("  (b)uy/trade  (s)pecial wares", end="")
			if npc.role == "trainer":
				print("  (r)eceive training", end="")
			if npc.role == "questgiver":
				print("  (q)uest", end="")
			print()
			choice = input("> ").strip().lower()
			if choice in ("l", "leave"):  # leave
				return f"You leave {npc.name}."
			if choice in ("t", "talk"):
				print(f"{npc.name} says: \"{npc.talk()}\"")
			elif choice in ("b", "buy", "trade") and npc.role == "merchant":
				if _merchant_shopping_session and getattr(npc, "inventory", None):
					_merchant_shopping_session(player, npc)
				else:
					print("This merchant has nothing to trade.")
			elif choice in ("s", "special") and npc.role == "merchant":
				# Offer rare items for sale
				rare_items = [
					Item("Enchanted Short Sword", "weapon", power=14, price=180),
					Item("Elven Bow", "weapon", power=12, price=150),
					Item("Mystic Robe", "armor", power=6, price=140),
					Item("Greater Potion", "consumable", heal=80, price=120),
					Item("Golden Amulet", "quest", price=0),
					Item("Runed Dagger", "weapon", power=9, price=95),
					Item("Reinforced Mail", "armor", power=9, price=135)
				]
				print("Special wares:")
				for i, it in enumerate(rare_items, 1):
					print(f"{i}. {it.name} ({it.type}) - {it.price} gold")
				sel = input("Choose item number to buy or b) back: ").strip().lower()
				if sel == "b" or not sel:
					continue
				try:
					idx = int(sel) - 1
					it = rare_items[idx]
					if player.gold < it.price:
						print("You cannot afford that.")
					else:
						player.gold -= it.price
						player.inventory.append(it)
						print(f"Purchased {it.name}.")
				except Exception:
					print("Invalid choice.")
			elif choice in ("r", "train", "training") and npc.role == "trainer":
				# Require quest completion for stat boost
				required_quest = next((q for q in player.active_quests if q.title == "Trial of Wisdom" and q.completed), None)
				if required_quest:
					print(f"{npc.name} recognizes your achievement. You gain +2 ATK!")
					player.atk += 2
				else:
					print(f"{npc.name} says: 'Complete the Trial of Wisdom quest first.'")
			elif choice in ("q", "quest") and npc.role == "questgiver":
				# Offer a quest
				possible = [t for t in QUEST_TEMPLATES if getattr(t, "type", "").lower() in ("gather", "fetch", "item")]
				offered = None
				if possible:
					tpl = random.choice(possible)
					if not any(a.title == tpl.title for a in player.active_quests):
						offered = Quest(str(uuid.uuid4()), tpl.title, tpl.description, tpl.type, tpl.target_name, tpl.target_count, tpl.level_req, tpl.reward_gold, tpl.reward_exp, [Item.from_dict(it.to_dict()) for it in tpl.reward_items])
				# Add a special quest for trainers
				if npc.name == "Seren the Sage" and not any(q.title == "Trial of Wisdom" for q in player.active_quests):
					offered = Quest(str(uuid.uuid4()), "Trial of Wisdom", "Prove your wisdom to Seren the Sage.", "gather", "Crystal Shard", 1, level_req=1, reward_gold=50, reward_exp=50, reward_items=[])
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
	if r < 0.33:
		ingredients = ["Herb", "Leather", "Wood", "Thread", "Bottle", "Iron Ingot", "Silver Ring", "Ancient Coin", "Crystal Shard", "Elixir Residue", "Enchanted Bark", "Iron Ingot"]
		if random.random() < 0.06:
			rare_items = [
				{"name": "Enchanted Short Sword", "type": "weapon", "power": 14, "heal": 0, "price": 180},
				{"name": "Elven Bow", "type": "weapon", "power": 12, "heal": 0, "price": 150},
				{"name": "Mystic Robe", "type": "armor", "power": 6, "heal": 0, "price": 140},
				{"name": "Greater Potion", "type": "consumable", "power": 0, "heal": 80, "price": 120},
				{"name": "Golden Amulet", "type": "quest", "power": 0, "heal": 0, "price": 0},
				{"name": "Runed Dagger", "type": "weapon", "power": 9, "heal": 0, "price": 95},
				{"name": "Reinforced Mail", "type": "armor", "power": 9, "heal": 0, "price": 135},
			]
			tpl = random.choice(rare_items)
			found = Item(tpl["name"], tpl["type"], power=tpl.get("power", 0), heal=tpl.get("heal", 0), price=tpl.get("price", 0))
			player.inventory.append(found)
			return f"You stumbled upon a rare item while foraging: {found.name}."
		else:
			found = random.choice(ingredients)
			player.inventory.append(Item(found, "quest", price=0))
			return f"You foraged and found: {found}."
	return None

def _generate_traveling_merchant():
	names = ["Maris", "Tob", "Selra", "Borin", "Ketha", "Lorin"]
	rare_items = [
		{"name": "Enchanted Short Sword", "type": "weapon", "power": 14, "heal": 0, "price": 180},
		{"name": "Elven Bow", "type": "weapon", "power": 12, "heal": 0, "price": 150},
		{"name": "Mystic Robe", "type": "armor", "power": 6, "heal": 0, "price": 140},
		{"name": "Greater Potion", "type": "consumable", "power": 0, "heal": 80, "price": 120},
		{"name": "Golden Amulet", "type": "quest", "power": 0, "heal": 0, "price": 0},
		{"name": "Runed Dagger", "type": "weapon", "power": 9, "heal": 0, "price": 95},
		{"name": "Reinforced Mail", "type": "armor", "power": 9, "heal": 0, "price": 135},
	]
	count = random.randint(2, 5)
	chosen = random.sample(rare_items, min(count, len(rare_items)))
	inv = []
	for tpl in chosen:
		it = Item(tpl["name"], tpl["type"], power=tpl.get("power", 0), heal=tpl.get("heal", 0), price=tpl.get("price", 0))
		markup = random.uniform(0.85, 1.5)
		it.price = max(1, int(it.price * markup))
		inv.append(it)
	name = f"{random.choice(names)} the Peddler"
	return NPC(name, "merchant", dialog=["Finest goods from far away.", "Rare wares, take a look."], inventory=inv)
