import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import threading
import time
import pyvista as pv
from pyvistaqt import BackgroundPlotter
import cadquery as cq

class StepViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("3D STEP File Viewer")
        self.root.geometry("900x700")
        self.rotation_speed = tk.DoubleVar(value=1.0)
        self.rotation_direction = tk.StringVar(value="Clockwise")
        self.plotter = None
        self.mesh = None
        self.is_rotating = False
        self.rotation_thread = None

        self.setup_gui()

    def setup_gui(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(control_frame, text="Upload STEP File", command=self.upload_step).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, text="Rotation Speed:").pack(side=tk.LEFT)
        tk.Scale(control_frame, variable=self.rotation_speed, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, length=150).pack(side=tk.LEFT)
        tk.Label(control_frame, text="Direction:").pack(side=tk.LEFT)
        ttk.Combobox(control_frame, textvariable=self.rotation_direction, values=["Clockwise", "Counterclockwise"], width=15).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="Toggle Rotation", command=self.toggle_rotation).pack(side=tk.LEFT, padx=5)

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Visual aids
        aid_frame = tk.Frame(self.root)
        aid_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        tk.Label(aid_frame, text="Visual Aids: Axes, Grid, Bounding Box").pack(side=tk.LEFT)

    def upload_step(self):
        file_path = filedialog.askopenfilename(filetypes=[("STEP files", "*.step *.stp")])
        if not file_path:
            return
        self.load_step_file(file_path)

    def load_step_file(self, file_path):
        # Load STEP file using cadquery
        try:
            shape = cq.importers.importStep(file_path)
            vertices, faces = shape.val().tessellate()
            vertices = np.array(vertices)
            faces = np.array(faces)
            # PyVista expects faces in a specific format: [n, i0, i1, i2, ...]
            # For triangles, prepend 3 to each face
            faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int64).flatten()
            pv_mesh = pv.PolyData(vertices, faces_pv)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load STEP file: {e}")
            return

        self.mesh = pv_mesh
        self.show_mesh()
    def show_mesh(self):
        if self.plotter:
            self.plotter.close()
        self.plotter = BackgroundPlotter(show=False, title="3D Render", window_size=[800, 600])
        self.plotter.add_mesh(self.mesh, color="lightblue", show_edges=True)
        self.plotter.show_axes()
        self.plotter.show_grid()
        self.plotter.add_bounding_box(color="red")
        self.plotter.set_background("white")
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
        self.plotter.show()
        self.is_rotating = False

    def reset_view(self):
        if self.plotter:
            self.plotter.camera_position = 'xy'
            self.plotter.reset_camera()

    def toggle_rotation(self):
        if not self.plotter or not self.mesh:
            return
        self.is_rotating = not self.is_rotating
        if self.is_rotating:
            self.rotation_thread = threading.Thread(target=self.rotate_mesh, daemon=True)
            self.rotation_thread.start()

    def rotate_mesh(self):
        while self.is_rotating:
            speed = self.rotation_speed.get()
            direction = 1 if self.rotation_direction.get() == "Clockwise" else -1
            self.plotter.camera.azimuth += direction * speed
            self.plotter.update()
            time.sleep(0.05)

if __name__ == "__main__":
    root = tk.Tk()
    app = StepViewerApp(root)
    root.mainloop()