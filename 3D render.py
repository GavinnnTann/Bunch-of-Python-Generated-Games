import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import threading
import time
import traceback
import pyvista as pv
import sys
import cadquery as cq

# Try to import PyVistaQt BackgroundPlotter
try:
    from pyvistaqt import BackgroundPlotter
    HAS_QT = True
except ImportError as e:
    HAS_QT = False
    print("PyVistaQt import error:", e)

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

        self.canvas_frame = tk.Frame(self.root, bg="white")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initial instructions
        instruction_label = tk.Label(
            self.canvas_frame, 
            text="Click 'Upload STEP File' to open a 3D model\n\nThe 3D viewer will open in a separate window",
            font=("Arial", 14),
            bg="white",
            pady=20
        )
        instruction_label.pack(expand=True)

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
            # Import STEP file
            shape = cq.importers.importStep(file_path)
            
            # For complex models, you might need to adjust the tesselation parameters
            tess_quality = 0.1  # Lower value for higher quality (more triangles)
            vertices, faces = shape.val().tessellate(tess_quality, tess_quality)
            
            # Convert vertices and faces to proper list format before numpy conversion
            vertices = [list(v) for v in vertices]
            vertices = np.array(vertices)
            faces = np.array(faces)
            
            # Convert faces to a list of lists for easier processing
            faces_list = [list(f) for f in faces]
            
            # PyVista expects faces in a specific format: [n, i0, i1, i2, ...]
            # For triangles, prepend 3 to each face
            faces_with_count = []
            for face in faces_list:
                faces_with_count.extend([len(face)] + face)
            
            # Create the PyVista mesh
            pv_mesh = pv.PolyData(np.array(vertices), np.array(faces_with_count))
            
            # Log successful import
            print(f"Successfully imported STEP file: {file_path}")
            print(f"Model contains {len(vertices)} vertices and {len(faces)} faces")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load STEP file: {e}")
            traceback_msg = traceback.format_exc()
            print(f"Error details:\n{traceback_msg}")
            return

        self.mesh = pv_mesh
        self.show_mesh()
    def show_mesh(self):
        if not HAS_QT:
            messagebox.showerror("Missing Dependency", 
                                "Qt bindings not found. Please install PyQt5 or PySide2 with:\n"
                                "pip install PyQt5\n or \npip install PySide2")
            print("\nPlease install a Qt binding with one of these commands:")
            print("pip install PyQt5")
            print("pip install PySide2")
            return
            
        # Close existing plotter if open
        if self.plotter:
            try:
                self.plotter.close()
            except:
                pass
            
        try:
            # Create a separate window for the 3D viewer
            print("Creating plotter...")
            self.plotter = BackgroundPlotter(title="3D STEP Model Viewer", window_size=(800, 600))
            print("Adding mesh...")
            self.plotter.add_mesh(self.mesh, color="lightblue", show_edges=True)
            print("Setting up view...")
            self.plotter.show_axes()
            self.plotter.show_grid()
            self.plotter.add_bounding_box(color="red")
            self.plotter.set_background("white")
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            
            print("Plotter created and should be visible now")
            self.is_rotating = False
            
            # Add message in the original Tkinter window
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
                
            success_label = tk.Label(self.canvas_frame, 
                                    text="3D model viewer opened in separate window", 
                                    font=("Arial", 12), fg="green", bg="white")
            success_label.pack(expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to render mesh: {str(e)}")
            print(f"Rendering error: {e}")
            traceback_msg = traceback.format_exc()
            print(f"Error details:\n{traceback_msg}")

    def reset_view(self):
        if self.plotter:
            self.plotter.camera_position = 'xy'
            self.plotter.reset_camera()

    def toggle_rotation(self):
        if not hasattr(self, 'mesh') or self.mesh is None:
            messagebox.showinfo("No Model", "Please load a model first")
            return
            
        self.is_rotating = not self.is_rotating
        if self.is_rotating:
            self.rotation_thread = threading.Thread(target=self.rotate_mesh, daemon=True)
            self.rotation_thread.start()

    def rotate_mesh(self):
        while self.is_rotating and hasattr(self, 'plotter') and self.plotter:
            try:
                speed = self.rotation_speed.get()
                direction = 1 if self.rotation_direction.get() == "Clockwise" else -1
                self.plotter.camera.azimuth += direction * speed
                self.plotter.render()
                time.sleep(0.05)
            except Exception as e:
                print(f"Rotation error: {e}")
                self.is_rotating = False
                break

if __name__ == "__main__":
    try:
        print("Starting 3D STEP File Viewer...")
        print("Dependencies check:")
        print(f"- PyVista version: {pv.__version__}")
        print(f"- Qt bindings available: {HAS_QT}")
        
        root = tk.Tk()
        root.geometry("900x700")
        root.minsize(800, 600)
        
        app = StepViewerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        traceback_msg = traceback.format_exc()
        print(f"Error details:\n{traceback_msg}")
        
        # Show error in GUI if possible
        try:
            tk.messagebox.showerror("Application Error", 
                                  f"The application encountered an error:\n{str(e)}\n\n"
                                  "Please check the console for more details.")
        except:
            print("Could not display error message in GUI.")