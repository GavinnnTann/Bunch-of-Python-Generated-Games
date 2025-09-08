# Python Projects Collection

A collection of Python projects including games, utilities, and experiments. This repository showcases various programming concepts implemented in Python, from text-based RPG games to mathematical experiments and rendering.

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [RPG Game](#rpg-game)
  - [Battleship](#battleship)
  - [Blackjack](#blackjack)
  - [3D Rendering](#3d-rendering)
  - [Other Projects](#other-projects)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains a diverse collection of Python projects, ranging from simple experiments to more complex game implementations. The main focus is on text-based games and graphical experiments, demonstrating various programming techniques and algorithms.

## Projects

### RPG Game

A comprehensive text-based Role-Playing Game with rich features and mechanics.

**Features:**
- Character progression with level-up system
- Inventory management and equipment
- Combat system with various enemies
- Quest system with chain quests that tell a story
- Town with shops, inn, and quest board
- Crafting system for creating items
- Trading system with special merchants
- Save/load functionality to persist progress

**How to Run:**
```bash
python -m RPG.rpg
```

**Game Structure:**
- `rpg.py`: Main game engine and loop
- `player.py`: Player character management
- `item.py`: Item system implementation
- `enemy.py`: Enemy definitions and combat
- `quest.py`: Quest system implementation
- `npc.py`: Non-player character system
- `crafting.py`: Crafting mechanics
- `shop.py`: Shopping functionality
- `save_load.py`: Game persistence

### Battleship

A classic Battleship game implementation with a text-based interface.

**Features:**
- Player vs Computer gameplay
- Grid-based ship placement
- Turn-based targeting system
- Win/loss detection

**How to Run:**
```bash
python Battleship.py
```

### Blackjack

Implementation of the popular card game Blackjack (21).

**Features:**
- Player vs Dealer gameplay
- Card deck management
- Betting system
- Game rules implementation

**How to Run:**
```bash
python blackjack.py
```

### 3D Rendering

A simple 3D rendering engine implemented in Python.

**Features:**
- Basic 3D wireframe rendering
- Camera controls
- Primitive shape generation

**How to Run:**
```bash
python "3D render.py"
```

### Other Projects

- **4D.py**: Experiments with 4D geometry visualization
- **fps.py**: Simple first-person shooter mechanics
- **primesprial.py**: Prime number generation and analysis
- **random number.py**: Random number generation experiments
- **excel filter.py**: Utility for filtering Excel data
- **Various Jupyter Notebooks**: Data experiments and visualizations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/GavinnnTann/Bunch-of-Python-Generated-Games.git
```

2. Navigate to the project directory:
```bash
cd Bunch-of-Python-Generated-Games
```

3. (Optional) Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

Note: Some projects require specific packages. The requirements.txt file includes all necessary dependencies for running the various projects in this repository.

## Usage

Each project can be run individually by executing its Python file. For more complex projects like the RPG game, follow the specific instructions in the project's section.

## Contributing

Contributions are welcome! If you'd like to improve any of the projects or add new ones:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).
