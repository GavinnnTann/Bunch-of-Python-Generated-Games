import random
import atexit
import os
import sys
import json
from datetime import datetime

START_BANK = 10000
CPU_BET_VARIANCE = 0.3  # CPUs will match player's bet ±30%
MIN_BET = 100
NUM_DECKS = 6  # Number of decks in the shoe

# Game settings
GAME_SETTINGS = {
    "double_down": True,     # Allow doubling the bet after seeing initial cards
    "split_pairs": True,     # Allow splitting matching pairs into two hands
    "insurance": True,       # Allow insurance bets when dealer shows an Ace
    "surrender": True,       # Allow surrendering hand for half the bet back
    "blackjack_payout": 1.5, # Configurable blackjack payout ratio (1.5 = 3:2)
    "num_decks": NUM_DECKS,  # Number of decks in the shoe
    "deck_penetration": 0.25 # Percentage of cards left when reshuffling occurs
}

# Save file location
SAVE_FILE = "blackjack_save.json"

INSTRUCTIONS = (
    "Welcome to Blackjack!\n\n"
    "Overview:\n"
    "- You play against the dealer and 3 computer players.\n"
    "- Each player starts with $10,000. Minimum bet is $100.\n\n"
    "Betting:\n"
    "- Enter a whole number to bet, 0 to fold, or 'a' to go all-in.\n\n"
    "Gameplay:\n"
    "- On your turn, you can:\n"
    "  * 'h' to hit (take another card)\n"
    "  * 's' to stand (end your turn)\n"
    "  * 'd' to double down (double bet, take one card, then stand)\n"
    "  * 'p' to split pairs (if you have two cards of same rank)\n"
    "  * 'u' to surrender (give up hand, lose half bet)\n"
    "- Insurance is offered when dealer shows an Ace\n"
    "- A natural Blackjack (initial 2-card 21) pays 3:2\n"
    "- Dealer must hit on 16 or less and stand on 17 or more\n\n"
    "Card Counting:\n"
    "- The running count is shown for educational purposes\n"
    "- Hi-Lo system: +1 for 2-6, 0 for 7-9, -1 for 10-A\n\n"
    "Goal:\n"
    "- Play hands until a player goes bankrupt; the first bankrupt player ends the game.\n\n"
    "Press Enter to begin..."
)

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["A"] + [str(n) for n in range(2, 11)] + ["J", "Q", "K"]


def make_deck():
    return [(r, s) for s in SUITS for r in RANKS]


def make_shoe(num_decks=NUM_DECKS):
    """Create a shoe with multiple decks"""
    shoe = []
    for _ in range(num_decks):
        shoe.extend(make_deck())
    random.shuffle(shoe)
    return shoe


def update_count(card, count):
    """Update the running count using Hi-Lo system"""
    rank = card[0]
    if rank in ["2", "3", "4", "5", "6"]:
        return count + 1
    elif rank in ["10", "J", "Q", "K", "A"]:
        return count - 1
    return count  # 7, 8, 9 are neutral


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


def is_soft_hand(hand):
    """Check if hand is soft (contains an Ace counted as 11)"""
    value = hand_value(hand)
    # Check if any Ace in the hand could be counted as 11 without busting
    for rank, _ in hand:
        if rank == "A" and value <= 21:
            return True
    return False


def is_pair(hand):
    """Check if hand is a pair (two cards of same rank)"""
    if len(hand) != 2:
        return False
    return hand[0][0] == hand[1][0]


def deal_initial(players, dealer, deck):
    """Deal initial cards to players and dealer"""
    # First card to each player and dealer
    for p in players:
        card = deck.pop()
        p["hand"].append(card)
    
    # Dealer's first card (face up)
    card = deck.pop()
    dealer["hand"].append(card)
    dealer["shown_card"] = card
    
    # Second card to each player
    for p in players:
        card = deck.pop()
        p["hand"].append(card)
    
    # Dealer's second card (face down)
    dealer["hand"].append(deck.pop())


def user_turn(player, dealer, deck, bets, insurance_bets):
    print("\n--- Your turn ---")
    
    # Check for blackjack
    if is_blackjack(player["hand"]):
        print(f"Your hand: {hand_str(player['hand'])}  => 21")
        print("Blackjack!")
        return
    
    # Check for insurance if dealer's shown card is an Ace
    if GAME_SETTINGS["insurance"] and dealer["shown_card"][0] == "A" and "You" not in insurance_bets:
        print(f"Dealer shows: {card_str(dealer['shown_card'])}")
        while True:
            ins_choice = input(f"Dealer has an Ace. Would you like insurance? (y/n): ").strip().lower()
            if ins_choice == 'y':
                insurance_amount = int(min(bets["You"] / 2, player["bank"]))
                insurance_bets["You"] = insurance_amount
                player["bank"] -= insurance_amount
                print(f"You place an insurance bet of ${insurance_amount:,}")
                break
            elif ins_choice == 'n':
                print("No insurance taken.")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    # Player's turn
    original_hand = player["hand"].copy()
    hands = [player["hand"]]  # For potential split hands
    current_hand_index = 0
    
    while current_hand_index < len(hands):
        current_hand = hands[current_hand_index]
        hand_complete = False
        
        # Special case for split Aces - only get one card per hand
        split_aces = False
        
        while not hand_complete:
            val = hand_value(current_hand)
            
            # Format display for multiple hands
            hand_label = "" if len(hands) == 1 else f" (Hand {current_hand_index + 1}/{len(hands)})"
            print(f"Your hand{hand_label}: {hand_str(current_hand)}  => {val}")
            
            if val > 21:
                print("You busted!")
                break
                
            # If this is a split Ace hand and has 2 cards, auto-stand
            if split_aces and len(current_hand) == 2:
                print("Auto-standing on split Ace.")
                break
                
            # Available actions
            actions = ['h', 's']  # Hit and Stand always available
            
            # Double down option - only on first action with 2 cards
            can_double = GAME_SETTINGS["double_down"] and len(current_hand) == 2 and player["bank"] >= bets["You"]
            if can_double:
                actions.append('d')
                
            # Split option - only on first action with a pair
            can_split = (GAME_SETTINGS["split_pairs"] and len(current_hand) == 2 and 
                        current_hand[0][0] == current_hand[1][0] and 
                        player["bank"] >= bets["You"] and
                        len(hands) < 4)  # Limit to 4 split hands
            if can_split:
                actions.append('p')
                
            # Surrender option - only on first action with original hand
            can_surrender = (GAME_SETTINGS["surrender"] and len(current_hand) == 2 and 
                           current_hand == original_hand and len(hands) == 1)
            if can_surrender:
                actions.append('u')
                
            # Show available actions
            action_text = "Hit (h), Stand (s)"
            if can_double:
                action_text += ", Double (d)"
            if can_split:
                action_text += ", Split (p)"
            if can_surrender:
                action_text += ", Surrender (u)"
                
            choice = input(f"{action_text}: ").strip().lower()
            
            if choice == "h":  # Hit
                card = deck.pop()
                current_hand.append(card)
                print(f"You drew {card_str(card)}")
                
            elif choice == "s":  # Stand
                print("You stand.")
                break
                
            elif choice == "d" and can_double:  # Double down
                print(f"Doubling down! Your bet increases from ${bets['You']:,} to ${bets['You']*2:,}")
                player["bank"] -= bets["You"]  # Deduct the additional bet
                bets["You"] *= 2
                
                # Take exactly one more card and then stand
                card = deck.pop()
                current_hand.append(card)
                print(f"You drew {card_str(card)}")
                print(f"Final hand: {hand_str(current_hand)}  => {hand_value(current_hand)}")
                break
                
            elif choice == "p" and can_split:  # Split
                print("Splitting pair!")
                
                # Create new hand with the second card
                new_hand = [current_hand.pop()]
                
                # Add a card to each hand
                card1 = deck.pop()
                current_hand.append(card1)
                print(f"First hand receives: {card_str(card1)}")
                
                card2 = deck.pop()
                new_hand.append(card2)
                print(f"Second hand receives: {card_str(card2)}")
                
                # Add the new hand to the list of hands
                hands.append(new_hand)
                
                # Double the bet (for the new hand)
                player["bank"] -= bets["You"]
                
                # Check if splitting Aces
                if current_hand[0][0] == 'A':
                    split_aces = True
                    
            elif choice == "u" and can_surrender:  # Surrender
                print("You surrender, losing half your bet.")
                bets["You"] /= 2  # Lose half the bet
                player["surrender"] = True
                return
                
            else:
                print(f"Invalid choice. Please enter one of: {', '.join(actions)}")
                
        # Move to next hand
        current_hand_index += 1


def computer_turn(player, dealer, deck, bets, insurance_bets):
    """Enhanced CPU strategy based on basic blackjack strategy chart"""
    print(f"\n--- {player['name']}'s turn ---")
    
    # Check for blackjack first
    if is_blackjack(player["hand"]):
        print(f"{player['name']}: {hand_str(player['hand'])} => 21")
        print(f"{player['name']} has Blackjack!")
        return
    
    # Basic strategy for insurance
    if GAME_SETTINGS["insurance"] and dealer["shown_card"][0] == "A" and player["name"] not in insurance_bets:
        # Take insurance about 30% of the time (not optimal but adds variation)
        if random.random() < 0.3:
            insurance_amount = int(min(bets[player["name"]] / 2, player["bank"]))
            insurance_bets[player["name"]] = insurance_amount
            player["bank"] -= insurance_amount
            print(f"{player['name']} takes insurance for ${insurance_amount:,}")
    
    # Computer plays using basic strategy
    dealer_up_card = dealer["shown_card"][0]
    if dealer_up_card in ["J", "Q", "K"]:
        dealer_up_card = "10"  # Treat face cards as 10
    
    # Basic strategy decision tree
    while True:
        player_hand = player["hand"]
        player_value = hand_value(player_hand)
        
        print(f"{player['name']}: {hand_str(player_hand)} => {player_value}")
        
        if player_value > 21:
            print(f"{player['name']} busted.")
            break
            
        # Define strategy based on player's hand and dealer's upcard
        
        # Always stand on 17 or higher
        if player_value >= 17:
            print(f"{player['name']} stands.")
            break
            
        # Always hit on 8 or less
        if player_value <= 8:
            action = "hit"
            
        # For 9-16, use basic strategy
        elif player_value == 9:
            # Double on 9 against dealer 3-6, otherwise hit
            if dealer_up_card in ["3", "4", "5", "6"] and len(player_hand) == 2 and player["bank"] >= bets[player["name"]]:
                action = "double"
            else:
                action = "hit"
                
        elif player_value == 10:
            # Double on 10 against dealer 2-9, otherwise hit
            if dealer_up_card in ["2", "3", "4", "5", "6", "7", "8", "9"] and len(player_hand) == 2 and player["bank"] >= bets[player["name"]]:
                action = "double"
            else:
                action = "hit"
                
        elif player_value == 11:
            # Double on 11 against all dealer cards, otherwise hit
            if len(player_hand) == 2 and player["bank"] >= bets[player["name"]]:
                action = "double"
            else:
                action = "hit"
                
        elif player_value == 12:
            # Stand against dealer 4-6, otherwise hit
            if dealer_up_card in ["4", "5", "6"]:
                action = "stand"
            else:
                action = "hit"
                
        elif 13 <= player_value <= 16:
            # Stand against dealer 2-6, otherwise hit
            if dealer_up_card in ["2", "3", "4", "5", "6"]:
                action = "stand"
            else:
                action = "hit"
        
        # Execute the chosen action
        if action == "hit":
            card = deck.pop()
            player_hand.append(card)
            print(f"{player['name']} draws {card_str(card)}")
            continue
            
        elif action == "stand":
            print(f"{player['name']} stands.")
            break
            
        elif action == "double":
            print(f"{player['name']} doubles down.")
            player["bank"] -= bets[player["name"]]
            bets[player["name"]] *= 2
            
            # Take one more card and stand
            card = deck.pop()
            player_hand.append(card)
            print(f"{player['name']} draws {card_str(card)}")
            print(f"Final hand: {hand_str(player_hand)} => {hand_value(player_hand)}")
            break


def dealer_turn(dealer, deck):
    """Dealer follows house rules: hit on 16 or less, stand on 17 or more"""
    print("\n--- Dealer's turn ---")
    
    # Reveal the face-down card
    print(f"Dealer reveals: {hand_str(dealer['hand'])}")
    
    # Dealer must hit until 17 or higher
    while True:
        dealer_value = hand_value(dealer["hand"])
        print(f"Dealer has: {hand_str(dealer['hand'])} => {dealer_value}")
        
        if dealer_value > 21:
            print("Dealer busts!")
            dealer["busted"] = True
            break
            
        if dealer_value >= 17:
            print("Dealer stands.")
            break
            
        # Dealer must hit
        card = deck.pop()
        dealer["hand"].append(card)
        print(f"Dealer draws {card_str(card)}")


def evaluate(players, dealer):
    """Evaluate hands against the dealer's hand"""
    dealer_value = hand_value(dealer["hand"])
    dealer_blackjack = is_blackjack(dealer["hand"])
    dealer_busted = dealer_value > 21
    
    results = []
    
    for p in players:
        hands = p.get("split_hands", [p["hand"]])
        hand_results = []
        
        for hand in hands:
            player_value = hand_value(hand)
            player_blackjack = is_blackjack(hand)
            player_busted = player_value > 21
            player_surrendered = p.get("surrender", False)
            
            # Determine outcome
            if player_surrendered:
                outcome = "surrender"
            elif player_busted:
                outcome = "loss"
            elif dealer_busted:
                outcome = "win"
            elif player_blackjack and not dealer_blackjack:
                outcome = "blackjack"
            elif dealer_blackjack and not player_blackjack:
                outcome = "loss"
            elif player_value > dealer_value:
                outcome = "win"
            elif dealer_value > player_value:
                outcome = "loss"
            else:
                outcome = "push"  # Tie
                
            hand_results.append({
                "hand": hand,
                "value": player_value,
                "outcome": outcome
            })
            
        results.append({
            "name": p["name"],
            "hand_results": hand_results
        })
        
    return results


def print_round_summary(players, dealer, results):
    print("\n--- Round Summary ---")
    
    # Print dealer's hand
    dealer_value = hand_value(dealer["hand"])
    dealer_status = "BUST" if dealer_value > 21 else str(dealer_value)
    print(f"Dealer: {hand_str(dealer['hand'])} => {dealer_status}")
    
    # Print each player's result
    for player_result in results:
        player_name = player_result["name"]
        hand_results = player_result["hand_results"]
        
        if len(hand_results) == 1:
            # Single hand
            hr = hand_results[0]
            status = "SURRENDER" if hr["outcome"] == "surrender" else "BUST" if hr["outcome"] == "loss" and hr["value"] > 21 else hr["value"]
            print(f"{player_name}: {hand_str(hr['hand'])} => {status} ({hr['outcome'].upper()})")
        else:
            # Multiple hands from splitting
            print(f"{player_name}:")
            for i, hr in enumerate(hand_results, 1):
                status = "BUST" if hr["outcome"] == "loss" and hr["value"] > 21 else hr["value"]
                print(f"  Hand {i}: {hand_str(hr['hand'])} => {status} ({hr['outcome'].upper()})")


def create_players_with_bank():
    players = [{"name": "You", "hand": [], "bank": START_BANK}]
    for i in range(1, 4):  # Now 3 CPU players + dealer
        players.append({"name": f"CPU {i}", "hand": [], "bank": START_BANK})
    return players


def create_dealer():
    return {"name": "Dealer", "hand": [], "shown_card": None}


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
                    print(f"\nYour bank: ${p['bank']:,}")
                    amt = input(f"Enter bet (min ${MIN_BET:,}, 0 to fold, 'a' for all-in): ").strip().lower()
                    if amt == "a":
                        bet = int(p["bank"])
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
                    print(f"You go all-in for ${bet:,}.")
                    bets[p["name"]] = bet
                    break
                if bet < MIN_BET:
                    print(f"Minimum bet is ${MIN_BET:,}.")
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
                    # Ensure we're using integers for randint
                    low = max(MIN_BET, int(round(player_bet * (1.0 - CPU_BET_VARIANCE))))
                    high = min(int(p["bank"]), int(round(player_bet * (1.0 + CPU_BET_VARIANCE))))
                    # Ensure there's a valid range to pick from
                    if high < MIN_BET:
                        # CPU can't meet even the minimum after matching; they fold
                        bets[p["name"]] = 0
                    else:
                        if low > high:
                            low = max(MIN_BET, min(low, high))
                        # Ensure both low and high are integers
                        low = int(low)
                        high = int(high)
                        bet = random.randint(low, high)
                        bets[p["name"]] = bet
                else:
                    # If player folded or bet 0, pick a conservative random bet
                    maxb = min(1000, int(p["bank"]))
                    bet = random.randint(MIN_BET, maxb)
                    bets[p["name"]] = bet
                print(f"{p['name']} bets ${bets[p['name']]:,} (bank ${p['bank']:,})")
    return bets


def settle_bets(players, bets, insurance_bets, results, dealer):
    """Settle all bets including main bets, insurance, and split hands"""
    dealer_blackjack = is_blackjack(dealer["hand"])
    
    # First, handle insurance bets if dealer has blackjack
    if dealer_blackjack and insurance_bets:
        print("\n--- Insurance Payouts ---")
        for player_name, bet_amount in insurance_bets.items():
            player = next(p for p in players if p["name"] == player_name)
            # Insurance pays 2:1
            payout = int(bet_amount * 2)
            player["bank"] += payout
            print(f"{player_name} wins ${payout:,} on insurance bet.")
    
    # Now handle main bets
    print("\n--- Main Bet Settlements ---")
    for player_result in results:
        player_name = player_result["name"]
        player = next(p for p in players if p["name"] == player_name)
        
        # If player folded or is already bankrupt
        if player_name not in bets or bets[player_name] == 0:
            continue
            
        # Handle each hand result
        for hand_result in player_result["hand_results"]:
            outcome = hand_result["outcome"]
            original_bet = bets[player_name]
            
            if outcome == "blackjack":
                # Blackjack pays 3:2
                payout = int(original_bet * (1 + GAME_SETTINGS["blackjack_payout"]))
                player["bank"] += payout
                print(f"{player_name} wins ${payout:,} with blackjack.")
                
            elif outcome == "win":
                # Regular win pays 1:1
                payout = int(original_bet * 2)
                player["bank"] += payout
                print(f"{player_name} wins ${payout:,}.")
                
            elif outcome == "push":
                # Push returns the original bet
                player["bank"] += original_bet
                print(f"{player_name} pushes and gets ${original_bet:,} back.")
                
            elif outcome == "surrender":
                # Surrender returns half the bet
                refund = int(original_bet / 2)
                player["bank"] += refund
                print(f"{player_name} surrendered and gets ${refund:,} back.")
                
            elif outcome == "loss":
                # Loss - bet is already deducted
                print(f"{player_name} loses ${original_bet:,}.")
                
    # Reset player states for next hand
    for p in players:
        p.pop("surrender", None)
        p.pop("split_hands", None)


def save_game_state(players, round_no, running_count):
    """Save the current game state to a file"""
    try:
        # We don't save hands, just player stats
        state = {
            "players": [{
                "name": p["name"],
                "bank": p["bank"]
            } for p in players],
            "round": round_no,
            "count": running_count,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(SAVE_FILE, "w") as f:
            json.dump(state, f, indent=2)
            
        print(f"\nGame saved to {SAVE_FILE}")
        return True
    except Exception as e:
        print(f"Error saving game: {e}")
        return False


def load_game_state():
    """Load a previously saved game state"""
    try:
        if not os.path.exists(SAVE_FILE):
            print("No saved game found.")
            return None
            
        with open(SAVE_FILE, "r") as f:
            state = json.load(f)
            
        print(f"\nLoaded game from {SAVE_FILE}")
        print(f"Saved on: {state['timestamp']}")
        print(f"Round: {state['round']}")
        
        for p in state["players"]:
            print(f"{p['name']}: ${p['bank']:,}")
            
        return state
    except Exception as e:
        print(f"Error loading game: {e}")
        return None


def apply_loaded_state(loaded_state):
    """Apply a loaded state to a new game"""
    if not loaded_state:
        return create_players_with_bank(), 1, 0
        
    # Create new players with loaded banks
    players = []
    for p_data in loaded_state["players"]:
        players.append({
            "name": p_data["name"],
            "hand": [],
            "bank": int(p_data["bank"])
        })
        
    round_no = int(loaded_state["round"])
    running_count = int(loaded_state.get("count", 0))  # Default to 0 if not found
    
    return players, round_no, running_count


def play_until_bankrupt():
    """Main game loop that runs until a player goes bankrupt"""
    # Check for saved game
    load_option = input("Would you like to load a saved game? (y/n): ").strip().lower()
    if load_option == 'y':
        loaded_state = load_game_state()
        players, round_no, running_count = apply_loaded_state(loaded_state)
    else:
        players = create_players_with_bank()
        round_no = 1
        running_count = 0
    
    # Create a dealer
    dealer = create_dealer()
    
    # Create a shoe (multiple decks)
    shoe = make_shoe(GAME_SETTINGS["num_decks"])
    cards_used = 0
    
    while True:
        print("\n" + "=" * 40)
        print(f"Round {round_no}")
        
        # Show card counting info
        true_count = running_count / max(1, (GAME_SETTINGS["num_decks"] - (cards_used / 52)))
        print(f"Running Count: {running_count} | True Count: {true_count:.2f}")
        
        # Show banks
        for p in players:
            print(f"{p['name']}: ${p['bank']:,}")
            
        # Save game option every 5 rounds
        if round_no % 5 == 0:
            save_option = input("Would you like to save your game? (y/n): ").strip().lower()
            if save_option == 'y':
                save_game_state(players, round_no, running_count)
        
        # Remove and report bankrupt players; end only if the user is bankrupt
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
        
        # Clear hands
        for p in players:
            p["hand"] = []
        dealer["hand"] = []
        dealer["shown_card"] = None
        dealer["busted"] = False
        
        # Take bets
        bets = take_bets(players)
        
        # If nobody bet, skip hand
        if all(v == 0 for v in bets.values()):
            print("No one bet. Ending game.")
            break
        
        # Check if shoe needs reshuffling (reached penetration threshold)
        cards_remaining = len(shoe)
        total_cards = GAME_SETTINGS["num_decks"] * 52
        if cards_remaining <= total_cards * GAME_SETTINGS["deck_penetration"]:
            print("\nReshuffling the shoe...")
            shoe = make_shoe(GAME_SETTINGS["num_decks"])
            cards_used = 0
            running_count = 0
        
        # Insurance bets
        insurance_bets = {}
        
        # Deal initial cards
        deal_initial(players, dealer, shoe)
        
        # Update card count for visible cards
        for p in players:
            for card in p["hand"]:
                running_count = update_count(card, running_count)
        # Update count for dealer's up card
        running_count = update_count(dealer["shown_card"], running_count)
        
        # Display dealer's up card
        print(f"\nDealer shows: {card_str(dealer['shown_card'])}")
        
        # Check for dealer blackjack
        if dealer["shown_card"][0] in ["A", "10", "J", "Q", "K"]:
            print("Checking for dealer blackjack...")
            if is_blackjack(dealer["hand"]):
                print(f"Dealer has blackjack! {hand_str(dealer['hand'])}")
                # Skip player turns if dealer has blackjack
                dealer_has_blackjack = True
            else:
                dealer_has_blackjack = False
                print("Dealer does not have blackjack. Play continues.")
        else:
            dealer_has_blackjack = False
        
        # Player turns (skip if dealer has blackjack)
        if not dealer_has_blackjack:
            # User turn
            user = next(p for p in players if p["name"] == "You")
            user_turn(user, dealer, shoe, bets, insurance_bets)
            
            # CPU turns
            for p in players:
                if p["name"] != "You":
                    computer_turn(p, dealer, shoe, bets, insurance_bets)
            
            # Dealer's turn
            dealer_turn(dealer, shoe)
            
            # Update count for dealer's face-down card and any hit cards
            for card in dealer["hand"][1:]:  # Skip the first card which was already counted
                running_count = update_count(card, running_count)
        
        # Evaluate results
        results = evaluate(players, dealer)
        
        # Print round summary
        print_round_summary(players, dealer, results)
        
        # Settle all bets
        settle_bets(players, bets, insurance_bets, results, dealer)
        
        # Update cards used count
        cards_used += total_cards - len(shoe)
        
        # Print updated banks
        print("\nBanks after this hand:")
        for p in players:
            print(f"{p['name']}: ${p['bank']:,}")
        
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