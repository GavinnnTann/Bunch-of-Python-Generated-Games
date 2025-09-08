import tkinter as tk
import math

class Kaleidoscope:
    def __init__(self, master, width=400, height=400, sectors=8):
        self.master = master
        self.width = width
        self.height = height
        self.sectors = sectors
        self.radius = min(width, height) // 2 - 10
        self.center = (width // 2, height // 2)
        self.canvas = tk.Canvas(master, width=width, height=height, bg='black')
        self.canvas.pack(side=tk.LEFT, padx=10)
        
        # Create instruction sidebar
        self.sidebar = tk.Frame(master)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Add instructions
        tk.Label(self.sidebar, text="KALEIDOSCOPE", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(self.sidebar, text="Instructions:", font=("Arial", 14, "bold")).pack(anchor=tk.W, pady=(10, 5))
        tk.Label(self.sidebar, text="• Click and drag to draw", font=("Arial", 14)).pack(anchor=tk.W)
        tk.Label(self.sidebar, text="• Press 'R' to clear canvas", font=("Arial", 14)).pack(anchor=tk.W)
        
        # Add reset button
        tk.Button(self.sidebar, text="Clear Canvas", command=self.clear_canvas).pack(pady=20)
        
        self.last = None
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset_last_point)
        
        # Bind the 'r' key to the clear canvas function
        master.bind("r", lambda e: self.clear_canvas())
        master.bind("R", lambda e: self.clear_canvas())

    def draw(self, event):
        if self.last is None:
            self.last = (event.x, event.y)
            return
        x0, y0 = self.last
        x1, y1 = event.x, event.y
        for i in range(self.sectors):
            angle = 2 * math.pi * i / self.sectors
            dx0 = x0 - self.center[0]
            dy0 = y0 - self.center[1]
            dx1 = x1 - self.center[0]
            dy1 = y1 - self.center[1]
            r0 = math.hypot(dx0, dy0)
            r1 = math.hypot(dx1, dy1)
            theta0 = math.atan2(dy0, dx0) + angle
            theta1 = math.atan2(dy1, dx1) + angle
            px0 = self.center[0] + r0 * math.cos(theta0)
            py0 = self.center[1] + r0 * math.sin(theta0)
            px1 = self.center[0] + r1 * math.cos(theta1)
            py1 = self.center[1] + r1 * math.sin(theta1)
            color = "#%02x%02x%02x" % (int(128+127*math.cos(angle)), int(128+127*math.sin(angle)), 200)
            self.canvas.create_line(px0, py0, px1, py1, fill=color, width=2)
        self.last = (x1, y1)

    def reset_last_point(self, event):
        self.last = None
        
    def clear_canvas(self):
        # Clear all drawings from the canvas
        self.canvas.delete("all")
        # Display a small message that fades out
        message = self.canvas.create_text(
            self.width // 2, self.height // 2, 
            text="Canvas Cleared!", 
            fill="white", 
            font=("Arial", 14)
        )
        # Use after to fade out the message
        self.master.after(1000, lambda: self.canvas.delete(message))

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Kaleidoscope")
    # Set window size to accommodate sidebar
    kaleido = Kaleidoscope(root, width=500, height=400)
    root.mainloop()