import tkinter as tk
import random
import time
import threading
import os
import hashlib

# Entropy sources
def entropy_time():
    return int(time.time() * 1000) % 10000

def entropy_os_urandom():
    return int.from_bytes(os.urandom(2), 'big') % 10000

def entropy_hash():
    s = str(random.random()) + str(time.time())
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 10000

def entropy_pid():
    return os.getpid() % 10000

def entropy_random():
    return random.randint(0, 9999)

ENTROPY_FUNCS = [
    ("System Time", entropy_time),
    ("OS Urandom", entropy_os_urandom),
    ("SHA256 Hash", entropy_hash),
    ("Process ID", entropy_pid),
    ("Python Random", entropy_random)
]

class FourDRandomApp:
    def __init__(self, root):
        self.root = root
        self.root.title("4D Random Number Generator")
        self.entropy_vars = []
        self.user_entropy = tk.StringVar()
        self.digits_labels = []
        self.setup_ui()

    def setup_ui(self):
        tk.Label(self.root, text="Increase Entropy (tick to use):").pack()
        for name, _ in ENTROPY_FUNCS:
            var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(self.root, text=name, variable=var)
            cb.pack(anchor='w')
            self.entropy_vars.append(var)

        tk.Label(self.root, text="User Input for Extra Entropy:").pack()
        tk.Entry(self.root, textvariable=self.user_entropy).pack(fill='x')

        self.digits_frame = tk.Frame(self.root)
        self.digits_frame.pack(pady=10)
        for _ in range(4):
            lbl = tk.Label(self.digits_frame, text="0", font=("Consolas", 32), width=2)
            lbl.pack(side='left', padx=5)
            self.digits_labels.append(lbl)

        tk.Button(self.root, text="Generate", command=self.start_generate).pack(pady=10)

    def start_generate(self):
        threading.Thread(target=self.animate_digits, daemon=True).start()

    def get_entropy_seed(self):
        seed = 0
        for var, (_, func) in zip(self.entropy_vars, ENTROPY_FUNCS):
            if var.get():
                seed ^= func()
        user_input = self.user_entropy.get()
        if user_input:
            seed ^= int(hashlib.sha256(user_input.encode()).hexdigest(), 16) % 10000
        return seed

    def animate_digits(self):
        seed = self.get_entropy_seed()
        random.seed(seed)
        digits = [random.randint(0, 9) for _ in range(4)]
        for i in range(4):
            for roll in range(10):
                self.digits_labels[i].config(text=str(random.randint(0, 9)))
                time.sleep(0.07)
            self.digits_labels[i].config(text=str(digits[i]))
            time.sleep(0.2)

if __name__ == "__main__":
    root = tk.Tk()
    app = FourDRandomApp(root)
    root.mainloop()