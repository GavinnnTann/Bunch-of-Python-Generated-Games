import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import os

# Sample sentences for the game
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a great programming language.",
    "Practice makes perfect.",
    "Artificial intelligence is the future.",
    "Type fast and accurately to win the game."
]

class TypeRacerGame:
    def __init__(self, root):
        self.root = root
        self.root.title("TypeRacer - Word Processor Edition")
        self.root.geometry("800x400")
        self.root.resizable(False, False)

        # Styling to mimic Office Word/Excel
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TButton", font=("Calibri", 12))
        self.style.configure("TLabel", font=("Calibri", 14))
        self.style.configure("TEntry", font=("Calibri", 14))

        self.start_time = None
        self.current_sentence = random.choice(SENTENCES)

        self.create_widgets()

    def create_widgets(self):
        # Title bar
        title_bar = tk.Frame(self.root, bg="#2b579a", height=30)
        title_bar.pack(fill="x")
        title_label = tk.Label(title_bar, text="TypeRacer - Word Processor Edition", bg="#2b579a", fg="white", font=("Calibri", 12, "bold"))
        title_label.pack(side="left", padx=10)

        # Sentence display in a text box at the corner
        sentence_frame = tk.Frame(self.root)
        sentence_frame.place(x=10, y=50)  # Position the text box in the corner
        self.sentence_text = tk.Text(sentence_frame, width=40, height=5, wrap="word", font=("Calibri", 12))
        self.sentence_text.insert("1.0", self.current_sentence)
        self.sentence_text.config(state="disabled")  # Make it read-only
        self.sentence_text.pack()

        # Typing area
        typing_frame = tk.Frame(self.root, pady=20)
        typing_frame.pack(pady=(100, 0))  # Adjust padding to move it below the text box
        self.typing_entry = ttk.Entry(typing_frame, width=80)
        self.typing_entry.pack()
        self.typing_entry.bind("<FocusIn>", self.start_timer)
        self.typing_entry.bind("<Return>", self.check_result)

        # Buttons
        button_frame = tk.Frame(self.root, pady=20)
        button_frame.pack()
        self.submit_button = ttk.Button(button_frame, text="Submit", command=self.check_result)
        self.submit_button.pack(side="left", padx=10)
        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_game)
        self.reset_button.pack(side="left", padx=10)

        def generate_sentences_from_workspace(self):
            # Example: Generate sentences from Python files in the workspace
            sentences = []
            workspace_dir = os.path.dirname(__file__)
            for root, _, files in os.walk(workspace_dir):
                for file in files:
                    if file.endswith(".py"):
                        with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith("#"):  # Skip comments and empty lines
                                    sentences.append(line)
            return sentences if sentences else SENTENCES

        def reset_game(self):
            self.start_time = None
            self.current_sentence = random.choice(self.generate_sentences_from_workspace())
            self.sentence_text.config(state="normal")
            self.sentence_text.delete("1.0", tk.END)
            self.sentence_text.insert("1.0", self.current_sentence)
            self.sentence_text.config(state="disabled")
            self.typing_entry.delete(0, tk.END)
