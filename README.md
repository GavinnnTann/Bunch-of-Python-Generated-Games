# Python Games and Utilities Collection

A comprehensive collection of Python games, utilities, and visualization tools. This repository showcases various programming concepts implemented in Python, from classic arcade games and text-based RPGs to 3D rendering and data manipulation tools.

## Table of Contents

- [Overview](#overview)
- [Games](#games)
  - [Snake Game](#snake-game)
  - [RPG Game](#rpg-game)
  - [Tetris](#tetris)
  - [Battleship](#battleship)
  - [Blackjack](#blackjack)
  - [Minesweeper](#minesweeper)
  - [Race Car](#race-car)
  - [Bouncing Ball](#bouncing-ball)
  - [Forest Fire](#forest-fire)
- [Visualization](#visualization)
  - [3D Rendering](#3d-rendering)
  - [Kaleidoscope](#kaleidoscope)
- [Utilities](#utilities)
  - [Excel Filter](#excel-filter)
  - [Random Number Generator](#random-number-generator)
  - [4D Number Generator](#4d-number-generator)
  - [Aim Trainer](#aim-trainer)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains a diverse collection of Python projects, ranging from classic games to visualization tools and utilities. The main focus is on interactive applications that demonstrate various algorithms, game mechanics, and programming techniques. Many of the games include AI components or physics simulations, making them educational as well as entertaining.

## Games

### Snake Game

A modern implementation of the classic Snake game with multiple play modes.

**Files:** `Snake game.py`

**Features:**
- Three play modes:
  - Manual: Control the snake with arrow keys
  - A* Algorithm: Watch the snake navigate using the A* pathfinding algorithm
  - Dijkstra Algorithm: Watch the snake navigate using Dijkstra's pathfinding algorithm
- Visual comparison between different pathfinding approaches
- Obstacle avoidance and path planning
- Progressive difficulty with increasing obstacles
- Score tracking and game over handling

**How to Run:**
```bash
python Snake game.py
```

### RPG Game

A comprehensive text-based Role-Playing Game with rich features and mechanics.

**Directory:** `RPG/`
**Main File:** `rpg.py`

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

### Tetris

Classic Tetris game implementation with standard gameplay mechanics.

**File:** `tetris.py`

**Features:**
- Falling block mechanics with rotation
- Line clearing and scoring system
- Increasing difficulty as you progress
- Game over detection and scoring

**How to Run:**
```bash
python tetris.py
```

### Battleship

A classic Battleship game implementation with a text-based interface.

**File:** `Battleship.py`

**Features:**
- Player vs Computer gameplay
- 10x10 grid with standard Battleship ships
- Manual or random ship placement
- Coordinate-based targeting system
- Win/loss detection

**How to Run:**
```bash
python Battleship.py
```

### Blackjack

Implementation of the popular card game Blackjack (21).

**File:** `blackjack.py`

**Features:**
- Play against 4 computer opponents
- $10,000 starting bank with betting system
- Standard Blackjack rules with hit/stand options
- Game ends when a player goes bankrupt

**How to Run:**
```bash
python blackjack.py
```

### Minesweeper

Implementation of the classic Minesweeper puzzle game.

**File:** `Minesweeper.py`

**Features:**
- Mine detection with numerical hints
- Flag placement for suspected mines
- Multiple difficulty levels
- Win/loss detection

**How to Run:**
```bash
python Minesweeper.py
```

### Race Car

Simple racing game with car physics.

**File:** `Race car.py`

**Features:**
- Control a car around a track
- Collision detection and physics
- Timing and lap tracking
- Obstacle avoidance

**How to Run:**
```bash
python "Race car.py"
```

### Bouncing Ball

Physics simulation of bouncing balls with realistic behavior.

**File:** `Bouncing Ball.py`

**Features:**
- Gravity and friction effects
- Ball collision physics
- Interactive environment
- Variable bounce parameters

**How to Run:**
```bash
python "Bouncing Ball.py"
```

### Forest Fire

Strategy game where you need to contain a forest fire by placing water barriers.

**Directory:** `Forest Fire/`
**Main File:** `Forest Fire.py`

**Features:**
- Challenging strategy gameplay
- Multiple difficulty levels
- Local highscore tracking with player names
- Visual animations for win/loss scenarios
- Custom grid size option

**How to Run:**
```bash
python "Forest Fire/Forest Fire.py"
```

## Visualization

### 3D Rendering

A 3D model viewer and renderer with advanced capabilities.

**File:** `3D render.py`

**Features:**
- Load and view 3D models
- Rotate, zoom, and manipulate 3D objects
- Uses PyVista and CADQuery for rendering
- Interactive visualization tools

**How to Run:**
```bash
python "3D render.py"
```

### Kaleidoscope

Visual kaleidoscope pattern generator with dynamic effects.

**File:** `Kaleidoscope.py`

**Features:**
- Dynamic, colorful pattern generation
- Interactive controls
- Mesmerizing visual effects
- Adjustable parameters

**How to Run:**
```bash
python Kaleidoscope.py
```

## Utilities

### Excel Filter

Tool for filtering and manipulating Excel data.

**File:** `excel filter.py`

**Features:**
- Load and filter Excel spreadsheets
- Apply custom filtering criteria
- Export filtered results
- Data analysis capabilities

**How to Run:**
```bash
python "excel filter.py"
```

### Random Number Generator

Advanced random number generation utility with visualization.

**File:** `random number.py`

**Features:**
- Various distribution options
- Visualization of random number patterns
- Statistical analysis tools
- Entropy measurement

**How to Run:**
```bash
python "random number.py"
```

### 4D Number Generator

A specialized random number generator that produces 4-digit numbers using multiple entropy sources.

**File:** `4D.py`

**Features:**
- Generates 4-digit random numbers
- Uses multiple entropy sources (System Time, OS Urandom, SHA256 Hash, Process ID, Python Random)
- User input as additional entropy source
- Animated digit rolling effect
- Configurable entropy sources

**How to Run:**
```bash
python 4D.py
```

### Aim Trainer

First-person shooting style aim training game.

**File:** `aim trainer.py`

**Features:**
- Target practice for improving mouse accuracy
- Moving targets with variable speed
- Score tracking system
- Crosshair aiming mechanics
- Timed challenges

**How to Run:**
```bash
python "aim trainer.py"
```

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
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

4. Install required dependencies:
```bash
pip install -r requirements.txt
```

Note: Some projects require specific packages. Individual game files may list their specific dependencies.

## Usage

Each project can be run individually by executing its Python file as shown in the "How to Run" instructions for each game. Most games feature interactive controls and on-screen instructions.

### Game Controls

- **Snake Game**: Arrow keys for movement
- **Tetris**: Arrow keys for movement and rotation
- **RPG Game**: Text-based commands (follow in-game instructions)
- **Battleship**: Text-based coordinate inputs (e.g., "A5", "J10")
- **Blackjack**: Text commands for hit, stand, and betting
- **Aim Trainer**: Mouse movement for aim, left-click to shoot

## Contributing

Contributions are welcome! If you'd like to improve any of the projects or add new ones:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please maintain the existing code style and add appropriate documentation for new features.

## License

This project is open source and available under the [MIT License](LICENSE).
