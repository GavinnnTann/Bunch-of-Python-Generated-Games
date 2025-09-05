# --- Sell menu for shop ---
def sell_menu(player):
    print("\n--- Sell Items ---")
    sellable = [it for it in player.inventory if getattr(it, "price", 0) > 0]
    if not sellable:
        print("You have nothing to sell.")
        return
    for i, it in enumerate(sellable, 1):
        print(f"{i}. {it.name} ({it.type}) - {it.price} gold")
    print("Enter item numbers to sell separated by spaces (e.g. '1 2 3'), or b) back:")
    sel = input("> ").strip().lower()
    if sel == "b" or not sel:
        return
    try:
        idxs = [int(s)-1 for s in sel.split() if s.isdigit()]
        to_sell = [sellable[i] for i in idxs if 0 <= i < len(sellable)]
    except Exception:
        print("Invalid selection.")
        return
    total = sum(it.price for it in to_sell)
    # Remove items from actual inventory (remove first matching object per item)
    for it in to_sell:
        try:
            player.inventory.remove(it)
        except ValueError:
            # already removed (duplicate items may be the same object); try to find by name as fallback
            for inv_it in player.inventory:
                if inv_it.name == it.name and getattr(inv_it, "price", 0) > 0:
                    player.inventory.remove(inv_it)
                    break
    player.gold += total
    print(f"Sold items for {total} gold. You now have {player.gold} gold.")
    return
# shop.py
from .item import Item
SHOP_ITEMS = [
    Item("Iron Sword", "weapon", power=6, price=30),
    Item("Steel Sword", "weapon", power=10, price=70),
    Item("Leather Armor", "armor", power=3, price=25),
    Item("Plate Armor", "armor", power=7, price=80),
    Item("Minor Potion", "consumable", heal=15, price=10),
    Item("Major Potion", "consumable", heal=40, price=40),
]
