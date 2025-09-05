import random
import atexit
import os
import sys

START_BANK = 10000
CPU_BET_VARIANCE = 0.3  # CPUs will match player's bet ±30%
MIN_BET = 100

INSTRUCTIONS = (
    "Welcome to Blackjack!\n\n"
    "Overview:\n"
    "- You play against 4 computer players.\n"
    "- Each player starts with $10000. Minimum bet is $100.\n\n"
    "Betting:\n"
    "- Enter a whole number to bet, 0 to fold, or 'a' to go all-in.\n\n"
    "Gameplay:\n"
    "- On your turn enter 'h' to hit or 's' to stand.\n"
    "- A natural Blackjack (initial 2-card 21) is paid with the usual bonus.\n\n"
    "Goal:\n"
    "- Play hands until a player goes bankrupt; the first bankrupt player ends the game.\n\n"
    "Press Enter to begin..."
)

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["A"] + [str(n) for n in range(2, 11)] + ["J", "Q", "K"]


def make_deck():
    return [(r, s) for s in SUITS for r in RANKS]


def card_str(card):
    return f"{card[0]}{card[1]}"


def hand_str(hand):
    return ", ".join(card_str(c) for c in hand)


def hand_value(hand):
    # Aces count as 11 unless that makes you bust; then some aces count as 1.
    total = 0
    aces = 0
    for rank, _ in hand:
        if rank == "A":
            total += 11
            aces += 1
        elif rank in ("J", "Q", "K"):
            total += 10
        else:
            total += int(rank)
    # Downgrade aces from 11 to 1 as needed
    while total > 21 and aces:
        total -= 10
        aces -= 1
    return total


def is_blackjack(hand):
    return len(hand) == 2 and hand_value(hand) == 21


def deal_initial(players, deck):
    for _ in range(2):
        for p in players:
            p["hand"].append(deck.pop())


def user_turn(player, deck):
    print("\n--- Your turn ---")
    while True:
        val = hand_value(player["hand"])
        print(f"Your hand: {hand_str(player['hand'])}  => {val}")
        if val > 21:
            print("You busted!")
            break
        if is_blackjack(player["hand"]):
            print("Blackjack!")
            break
        choice = input("Hit or stand? (h/s): ").strip().lower()
        if choice == "h":
            card = deck.pop()
            player["hand"].append(card)
            print(f"You drew {card_str(card)}")
            continue
        elif choice == "s":
            print("You stand.")
            break
        else:
            print("Type 'h' to hit or 's' to stand.")


def computer_turn(player, deck):
    # Simple strategy: hit until value >= 17
    # Slight randomness: on soft 17 (17 with an ace counted as 11), some CPUs may hit
    print(f"\n--- {player['name']}'s turn ---")
    while True:
        val = hand_value(player["hand"])
        print(f"{player['name']}: {hand_str(player['hand'])} => {val}")
        if val > 21:
            print(f"{player['name']} busted.")
            break
        if is_blackjack(player["hand"]):
            print(f"{player['name']} has Blackjack.")
            break

        # Determine soft 17: contains an ace counted as 11
        contains_ace = any(r == "A" for r, _ in player["hand"])
        soft = False
        if contains_ace:
            soft = (val == 17)

        # CPU decision
        if val < 17:
            action = "hit"
        elif val > 17:
            action = "stand"
        else:  # val == 17
            # on soft 17, sometimes hit (30% chance), otherwise stand
            if soft and random.random() < 0.3:
                action = "hit"
            else:
                action = "stand"

        if action == "hit":
            card = deck.pop()
            player["hand"].append(card)
            print(f"{player['name']} draws {card_str(card)}")
            continue
        else:
            print(f"{player['name']} stands.")
            break


def evaluate(players):
    results = []
    for p in players:
        val = hand_value(p["hand"])
        busted = val > 21
        results.append({"name": p["name"], "value": val, "busted": busted})
    # Find highest non-bust value
    non_busts = [r for r in results if not r["busted"]]
    if not non_busts:
        return [], results
    max_val = max(r["value"] for r in non_busts)
    winners = [r["name"] for r in non_busts if r["value"] == max_val]
    return winners, results


def print_round_summary(players, results):
    print("\n--- Round Summary ---")
    for p, r in zip(players, results):
        status = "BUST" if r["busted"] else str(r["value"])
        print(f"{p['name']}: {hand_str(p['hand'])} => {status}")


def create_players_with_bank():
    players = [{"name": "You", "hand": [], "bank": START_BANK}]
    for i in range(1, 5):
        players.append({"name": f"CPU {i}", "hand": [], "bank": START_BANK})
    return players


def take_bets(players):
    bets = {}
    for p in players:
        # skip players already bankrupt
        if p["bank"] <= 0:
            bets[p["name"]] = 0
            continue
        if p["name"] == "You":
            while True:
                try:
                    print(f"\nYour bank: ${p['bank']}")
                    amt = input(f"Enter bet (min ${MIN_BET}, 0 to fold, 'a' for all-in): ").strip().lower()
                    if amt == "a":
                        bet = p["bank"]
                    else:
                        bet = int(amt)
                except Exception:
                    print("Enter a whole number, 0 to fold or 'a' to go all-in.")
                    continue

                if bet == 0:
                    print("You fold this hand.")
                    bets[p["name"]] = 0
                    break
                # allow all-in even if under MIN_BET
                if amt == "a":
                    print(f"You go all-in for ${bet}.")
                    bets[p["name"]] = bet
                    break
                if bet < MIN_BET:
                    print(f"Minimum bet is ${MIN_BET}.")
                elif bet > p["bank"]:
                    print("You cannot bet more than your bank.")
                else:
                    bets[p["name"]] = bet
                    break
        else:
            # CPU betting matches the player's bet ±CPU_BET_VARIANCE (30%)
            if p["bank"] < MIN_BET:
                bets[p["name"]] = 0
            else:
                player_bet = bets.get("You", 0)
                if player_bet > 0:
                    low = max(MIN_BET, int(round(player_bet * (1.0 - CPU_BET_VARIANCE))))
                    high = min(p["bank"], int(round(player_bet * (1.0 + CPU_BET_VARIANCE))))
                    # Ensure there's a valid range to pick from
                    if high < MIN_BET:
                        # CPU can't meet even the minimum after matching; they fold
                        bets[p["name"]] = 0
                    else:
                        if low > high:
                            low = max(MIN_BET, min(low, high))
                        bet = random.randint(low, high)
                        bets[p["name"]] = bet
                else:
                    # If player folded or bet 0, pick a conservative random bet
                    maxb = min(1000, p["bank"])
                    bet = random.randint(MIN_BET, maxb)
                    bets[p["name"]] = bet
                print(f"{p['name']} bets ${bets[p['name']]} (bank ${p['bank']})")
    return bets


def settle_bets(players, bets, winners, results):
    # Deduct all bets first
    for p in players:
        p["bank"] -= bets.get(p["name"], 0)

    pot = sum(bets.values())
    if pot == 0:
        print("\nNo bets this hand.")
        return

    if not winners:
        # everyone busted: house takes pot (pot removed already)
        print("\nAll busted. House takes the pot.")
        return

    # If single winner and they have blackjack (and it was initial blackjack), pay 1.5x
    # Detect blackjack winners (initial 2-card 21)
    blackjack_pays = {}
    if len(winners) == 1:
        winner_name = winners[0]
        # find player
        p = next(x for x in players if x["name"] == winner_name)
        if is_blackjack(p["hand"]):
            blackjack_pays[winner_name] = 1.5

    # Split pot equally among winners, applying blackjack bonus if applicable
    splits = {}
    base_share = pot / len(winners)
    for w in winners:
        multiplier = blackjack_pays.get(w, 1.0)
        splits[w] = base_share * multiplier

    # If any blackjack multipliers exist, their extra must come from the pot;
    # we won't create additional money, so normalize: cap total paid to pot and
    # if overpay would occur, re-normalize proportionally.
    total_payout = sum(splits.values())
    if total_payout > pot:
        factor = pot / total_payout
        for w in splits:
            splits[w] *= factor

    # Apply payouts
    for p in players:
        if p["name"] in splits:
            amt = int(round(splits[p["name"]]))
            p["bank"] += amt
            print(f"{p['name']} wins ${amt} this hand.")


def play_until_bankrupt():
    players = create_players_with_bank()
    round_no = 1
    while True:
        print("\n" + "=" * 40)
        print(f"Round {round_no}")
        # show banks
        for p in players:
            print(f"{p['name']}: ${p['bank']}")
        # remove and report bankrupt players; end only if the user is bankrupt
        bankrupt = [p for p in players if p["bank"] <= 0]
        if bankrupt:
            user_bankrupt = any(p["name"] == "You" for p in bankrupt)
            if user_bankrupt:
                for b in bankrupt:
                    print(f"\n{b['name']} is bankrupt. Game over.")
                break
            else:
                for b in bankrupt:
                    print(f"\n{b['name']} is bankrupt and eliminated from the game.")
                    players.remove(b)

        # clear hands
        for p in players:
            p["hand"] = []

        bets = take_bets(players)
        # if nobody bet, skip hand
        if all(v == 0 for v in bets.values()):
            print("No one bet. Ending game.")
            break

        # play a hand using existing helpers defined above in the file
        deck = make_deck()
        random.shuffle(deck)
        deal_initial(players, deck)

        # user automatic blackjack check still handled inside user_turn
        user = next(p for p in players if p["name"] == "You")
        if is_blackjack(user["hand"]):
            print("You have Blackjack on the deal!")
        user_turn(user, deck)
        for p in players:
            if p["name"] != "You":
                computer_turn(p, deck)

        winners, results = evaluate(players)
        print_round_summary(players, results)
        if not winners:
            print("\nAll players busted. No winners this round.")
        elif len(winners) == 1:
            if winners[0] == "You":
                print("\nYou win the hand!")
            else:
                print(f"\n{winners[0]} wins the hand.")
        else:
            if "You" in winners:
                print("\nYou tied for the win with:", ", ".join(w for w in winners if w != "You"))
            else:
                print("\nTie between computers:", ", ".join(winners))

        settle_bets(players, bets, winners, results)

        # print updated banks so the user can always see money after the hand
        print("\nBanks after this hand:")
        for p in players:
            print(f"{p['name']}: ${p['bank']}")

        round_no += 1


def show_instructions():
    print("\n" + "=" * 60)
    print(INSTRUCTIONS)
    print("=" * 60 + "\n")
    try:
        input()
    except Exception:
        # non-interactive environments may not support input()
        pass


def main():
    print("Blackjack with banking: first to go bankrupt ends the game.")
    show_instructions()
    play_until_bankrupt()


# Entry point: run the banking game loop (players start with banks and game continues until someone is bankrupt)
if __name__ == "__main__":
    while True:
        main()
        try:
            ans = input("\nGame ended. Press 'r' + Enter to restart, or any other key to exit: ").strip().lower()
        except Exception:
            # Non-interactive environment or EOF; exit the loop
            break
        if ans == "r":
            # Restart by repeating the loop (works in terminals and IDEs where execv may not behave as expected)
            continue
        else:
            break