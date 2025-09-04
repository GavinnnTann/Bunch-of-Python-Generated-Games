import tkinter as tk
from tkinter import messagebox
import random
import time
import os

def get_entropy(use_time, use_env, use_user):
    entropy = random.SystemRandom().randint(0, 1 << 30)
    if use_time:
        entropy ^= int(time.time() * 1000)
    if use_env:
        entropy ^= sum(ord(c) for c in os.environ.get('USERNAME', ''))
    if use_user:
        entropy ^= sum(ord(c) for c in user_input.get())
    return entropy

def generate_numbers():
    use_time = var_time.get()
    use_env = var_env.get()
    use_user = var_user.get()
    seed = get_entropy(use_time, use_env, use_user)
    rng = random.Random(seed)
    numbers = rng.sample(range(1, 50), 7)
    result_var.set("Numbers: " + ", ".join(map(str, numbers)))

root = tk.Tk()
root.title("Random Number Generator")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

tk.Label(frame, text="Increase entropy using:").grid(row=0, column=0, sticky="w")

var_time = tk.BooleanVar()
var_env = tk.BooleanVar()
var_user = tk.BooleanVar()

tk.Checkbutton(frame, text="Current Time", variable=var_time).grid(row=1, column=0, sticky="w")
tk.Checkbutton(frame, text="Environment Variable", variable=var_env).grid(row=2, column=0, sticky="w")
tk.Checkbutton(frame, text="User Input", variable=var_user).grid(row=3, column=0, sticky="w")

tk.Label(frame, text="User Input:").grid(row=4, column=0, sticky="w")
user_input = tk.Entry(frame)
user_input.grid(row=5, column=0, sticky="we")

generate_btn = tk.Button(frame, text="Generate Numbers", command=generate_numbers)
generate_btn.grid(row=6, column=0, pady=10)

result_var = tk.StringVar()
result_label = tk.Label(frame, textvariable=result_var, font=("Arial", 12))
result_label.grid(row=7, column=0, pady=5)

root.mainloop()