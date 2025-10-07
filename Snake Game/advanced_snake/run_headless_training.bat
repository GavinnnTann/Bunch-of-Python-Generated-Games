@echo off
echo Snake Game - Headless DQN Training

echo Options:
echo 1. Terminal-only training (fastest)
echo 2. Training with real-time graphs
echo 3. Exit

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" (
    echo Starting headless training (terminal-only)...
    python headless_training.py --episodes 2000 --save-interval 50
) else if "%choice%"=="2" (
    echo Starting headless training with real-time graphs...
    python headless_training.py --episodes 2000 --save-interval 50 --show-graphs
) else if "%choice%"=="3" (
    echo Exiting...
    exit
) else (
    echo Invalid choice!
    pause
    exit
)

echo Training complete!
pause