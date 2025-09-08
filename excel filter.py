#!/usr/bin/env python3
"""
Excel Filter - Tkinter Application
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
import traceback
import threading
import queue
import re
from typing import Optional, Callable, Any, Dict, List, Tuple


class ExcelFilterApp:
    """Main application controller for the Excel Filter GUI."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the application with the root window."""
        self.root = root
        self.root.title("Excel Filter — Tkinter")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        
        # Data storage
        self.original_df: Optional[pd.DataFrame] = None
        self.working_df: Optional[pd.DataFrame] = None
        self.filter_stack: List[Dict[str, Any]] = []
        self.last_directory: str = os.path.expanduser("~")
        
        # UI update queue for thread safety
        self.ui_queue = queue.Queue()
        
        # Initialize UI
        self._setup_ui()
        self._setup_bindings()
        self._update_ui_state()
        
        # Start UI queue processor
        self._process_ui_queue()
        
        self.log("Application started")
    
    def _setup_ui(self):
        """Set up the main user interface layout."""
        # Configure root grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Main frame
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=2)
        main_frame.grid_columnconfigure(1, weight=1)

        # Top button frame
        self._create_button_frame(main_frame)

        # Main content area (row=1)
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=2)  # Data preview expands horizontally
        content_frame.grid_columnconfigure(1, weight=0)  # Controls fixed width

        # Left pane - Data preview
        self._create_data_preview_pane(content_frame)

        # Right pane - Controls
        self._create_controls_pane(content_frame)

        # Bottom pane - Terminal/Log (row=2)
        self._create_terminal_pane(main_frame)

    def _create_data_preview_pane(self, parent):
        """Create the left pane with data preview table."""
        preview_frame = ttk.LabelFrame(parent, text="Data Preview", padding="5")
        preview_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # Count labels
        self.count_label = ttk.Label(preview_frame, text="Rows (original): 0 | Rows (current): 0 | Columns: 0")
        self.count_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        # Treeview for data display in a canvas for horizontal scrolling
        tree_canvas = tk.Canvas(preview_frame)
        tree_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_canvas.grid_rowconfigure(0, weight=1)
        tree_canvas.grid_columnconfigure(0, weight=1)

        tree_frame = ttk.Frame(tree_canvas)
        tree_window = tree_canvas.create_window((0, 0), window=tree_frame, anchor='nw')

        self.data_tree = ttk.Treeview(tree_frame)
        self.data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)

        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        h_scrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
        self.data_tree.configure(xscrollcommand=h_scrollbar.set)

        # Make treeview expand with frame
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        # Bind resizing to canvas scroll region
        def on_frame_configure(event):
            tree_canvas.configure(scrollregion=tree_canvas.bbox("all"))
        tree_frame.bind("<Configure>", on_frame_configure)

    
    def _create_button_frame(self, parent):
        """Create the top button frame with main action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.upload_btn = ttk.Button(button_frame, text="Upload", command=self.upload_file)
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.filter_btn = ttk.Button(button_frame, text="Filter", command=self.apply_filter)
        self.filter_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_filters)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(button_frame, text="Export CSV", command=self.export_csv)
        self.export_btn.pack(side=tk.LEFT, padx=5)

        self.export_excel_btn = ttk.Button(button_frame, text="Export to Excel Sheet", command=self.export_excel_sheet)
        self.export_excel_btn.pack(side=tk.LEFT, padx=5)
        
        self.calculate_btn = ttk.Button(button_frame, text="Calculate Stats", command=self.calculate_statistics)
        self.calculate_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label on the right
        self.status_label = ttk.Label(button_frame, text="No data loaded")
        self.status_label.pack(side=tk.RIGHT)
    
    def _create_data_preview_pane(self, parent):
        """Create the left pane with data preview table."""
        preview_frame = ttk.LabelFrame(parent, text="Data Preview", padding="5")
        preview_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Count labels
        self.count_label = ttk.Label(preview_frame, text="Rows (original): 0 | Rows (current): 0 | Columns: 0")
        self.count_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # Treeview for data display
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        self.data_tree = ttk.Treeview(tree_frame)
        self.data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.data_tree.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.data_tree.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.data_tree.configure(xscrollcommand=h_scrollbar.set)
    
    def _create_controls_pane(self, parent):
        """Create the right pane with filtering controls."""
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        controls_frame.grid_rowconfigure(3, weight=1)  # Make scripting panel expandable
        controls_frame.grid_columnconfigure(0, weight=1)
        
        # Basic filtering section
        self._create_basic_filter_section(controls_frame)
        
        # Separator
        ttk.Separator(controls_frame, orient=tk.HORIZONTAL).grid(row=2, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Advanced scripting section
        self._create_scripting_section(controls_frame)
        
        # Filter stack section
        self._create_filter_stack_section(controls_frame)
    
    def _create_basic_filter_section(self, parent):
        """Create the basic filter controls section."""
        basic_frame = ttk.LabelFrame(parent, text="Basic Filter", padding="5")
        basic_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        basic_frame.grid_columnconfigure(1, weight=1)
        
        # Column selection
        ttk.Label(basic_frame, text="Column:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(basic_frame, textvariable=self.column_var, state="readonly")
        self.column_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        self.column_combo.bind('<<ComboboxSelected>>', self._on_column_selected)
        
        # Column type display
        self.dtype_label = ttk.Label(basic_frame, text="Type: -", font=("TkDefaultFont", 8))
        self.dtype_label.grid(row=1, column=1, sticky=tk.W, padx=(5, 0))
        
        # Operator selection
        ttk.Label(basic_frame, text="Operator:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.operator_var = tk.StringVar(value="==")
        self.operator_combo = ttk.Combobox(basic_frame, textvariable=self.operator_var, state="readonly")
        self.operator_combo['values'] = ["==", "!=", "contains", "startswith", "endswith", 
                                        ">", ">=", "<", "<=", "between", "isnull", "notnull", "regex"]
        self.operator_combo.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        self.operator_combo.bind('<<ComboboxSelected>>', self._on_operator_selected)
        
        # Value entry
        ttk.Label(basic_frame, text="Value:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.value_var = tk.StringVar()
        self.value_entry = ttk.Entry(basic_frame, textvariable=self.value_var)
        self.value_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=(5, 0), pady=2)
        
        # Options frame
        options_frame = ttk.Frame(basic_frame)
        options_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.case_insensitive_var = tk.BooleanVar(value=True)
        case_check = ttk.Checkbutton(options_frame, text="Case-insensitive", variable=self.case_insensitive_var)
        case_check.pack(side=tk.LEFT)
        
        self.include_nan_var = tk.BooleanVar(value=False)
        nan_check = ttk.Checkbutton(options_frame, text="Include NaN", variable=self.include_nan_var)
        nan_check.pack(side=tk.LEFT, padx=(10, 0))
    
    def _create_scripting_section(self, parent):
        """Create the advanced scripting section."""
        script_frame = ttk.LabelFrame(parent, text="Advanced Filter", padding="5")
        script_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        script_frame.grid_rowconfigure(2, weight=1)
        script_frame.grid_columnconfigure(0, weight=1)
        
        # Script mode selection
        mode_frame = ttk.Frame(script_frame)
        mode_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.script_mode_var = tk.StringVar(value="query")
        query_radio = ttk.Radiobutton(mode_frame, text="pandas query", variable=self.script_mode_var, value="query")
        query_radio.pack(side=tk.LEFT)
        lambda_radio = ttk.Radiobutton(mode_frame, text="Python lambda", variable=self.script_mode_var, value="lambda")
        lambda_radio.pack(side=tk.LEFT, padx=(10, 0))
        
        # Example label
        self.script_example_label = ttk.Label(script_frame, text="Example: (ColA > 5) & ColB.str.contains('abc', case=False)", 
                                            font=("TkDefaultFont", 8), foreground="gray")
        self.script_example_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        # Script text area
        script_text_frame = ttk.Frame(script_frame)
        script_text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        script_text_frame.grid_rowconfigure(0, weight=1)
        script_text_frame.grid_columnconfigure(0, weight=1)
        
        self.script_text = tk.Text(script_text_frame, height=6, wrap=tk.WORD, font=("Courier", 9))
        self.script_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        script_scroll = ttk.Scrollbar(script_text_frame, orient=tk.VERTICAL, command=self.script_text.yview)
        script_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.script_text.configure(yscrollcommand=script_scroll.set)
        
        # Run script button
        self.run_script_btn = ttk.Button(script_frame, text="Run Script", command=self.run_script_filter)
        self.run_script_btn.grid(row=3, column=0, pady=(5, 0))
    
    def _create_filter_stack_section(self, parent):
        """Create the filter stack display section."""
        stack_frame = ttk.LabelFrame(parent, text="Filter Stack", padding="5")
        stack_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        stack_frame.grid_rowconfigure(0, weight=1)
        stack_frame.grid_columnconfigure(0, weight=1)
        
        # Filter stack listbox
        stack_list_frame = ttk.Frame(stack_frame)
        stack_list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stack_list_frame.grid_rowconfigure(0, weight=1)
        stack_list_frame.grid_columnconfigure(0, weight=1)
        
        self.filter_stack_listbox = tk.Listbox(stack_list_frame, height=3, font=("TkDefaultFont", 8))
        self.filter_stack_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        stack_scroll = ttk.Scrollbar(stack_list_frame, orient=tk.VERTICAL, command=self.filter_stack_listbox.yview)
        stack_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.filter_stack_listbox.configure(yscrollcommand=stack_scroll.set)
        
        # Undo button
        self.undo_btn = ttk.Button(stack_frame, text="Undo Last Filter", command=self.undo_last_filter)
        self.undo_btn.grid(row=1, column=0, pady=(5, 0))
    
    def _create_terminal_pane(self, parent):
        """Create the bottom terminal/log pane."""
        terminal_frame = ttk.LabelFrame(parent, text="Activity Log", padding="5")
        terminal_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(5, 0))
        terminal_frame.grid_rowconfigure(0, weight=1)
        terminal_frame.grid_columnconfigure(0, weight=1)
        
        # Log text area
        log_text_frame = ttk.Frame(terminal_frame)
        log_text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_text_frame.grid_rowconfigure(0, weight=1)
        log_text_frame.grid_columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_text_frame, height=8, state=tk.DISABLED, 
                               wrap=tk.WORD, font=("Courier", 9), bg="#f8f8f8")
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scroll = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        # Log control buttons
        log_btn_frame = ttk.Frame(terminal_frame)
        log_btn_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.copy_log_btn = ttk.Button(log_btn_frame, text="Copy Log", command=self.copy_log)
        self.copy_log_btn.pack(side=tk.LEFT)
        
        self.clear_log_btn = ttk.Button(log_btn_frame, text="Clear Log", command=self.clear_log)
        self.clear_log_btn.pack(side=tk.LEFT, padx=(5, 0))
    
    def _setup_bindings(self):
        """Set up event bindings."""
        # Bind script mode change to update example text
        self.script_mode_var.trace('w', self._update_script_example)
        
        # Bind Enter key to filter
        self.value_entry.bind('<Return>', lambda e: self.apply_filter())
        
        # Bind Ctrl+Return to run script
        self.script_text.bind('<Control-Return>', lambda e: self.run_script_filter())
        
        # Set initial operator state
        self._on_operator_selected()
    
    def _update_script_example(self, *args):
        """Update the script example text based on selected mode."""
        if self.script_mode_var.get() == "query":
            example = "Example: (ColA > 5) & ColB.str.contains('abc', case=False)"
        else:
            example = "Example: lambda df: (df['ColA'] > 5) & df['ColB'].str.contains('abc')"
        self.script_example_label.config(text=example)
    
    def _on_column_selected(self, event=None):
        """Handle column selection change."""
        if self.working_df is None:
            return
        
        column = self.column_var.get()
        if column and column in self.working_df.columns:
            dtype = str(self.working_df[column].dtype)
            self.dtype_label.config(text=f"Type: {dtype}")
    
    def _on_operator_selected(self, event=None):
        """Handle operator selection change."""
        operator = self.operator_var.get()
        
        # For isnull/notnull operators, disable value entry (not needed)
        if operator in ["isnull", "notnull"]:
            self.value_entry.config(state="disabled")
        else:
            self.value_entry.config(state="normal")
            
        # For between operator, show hint about comma-separated values
        if operator == "between":
            self.value_var.set("min, max")
            self.value_entry.select_range(0, len(self.value_var.get()))
    
    def _update_ui_state(self):
        """Update the UI state based on current data availability."""
        has_data = self.working_df is not None
        
        self.filter_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.export_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.export_excel_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.run_script_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.undo_btn.config(state=tk.NORMAL if self.filter_stack else tk.DISABLED)
        self.calculate_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        
        if has_data:
            # Update column combobox
            columns = list(self.working_df.columns)
            self.column_combo['values'] = columns
            if not self.column_var.get() and columns:
                self.column_var.set(columns[0])
                self._on_column_selected()
            
            # Update status
            orig_count = len(self.original_df) if self.original_df is not None else 0
            curr_count = len(self.working_df)
            col_count = len(self.working_df.columns)
            
            self.status_label.config(text=f"Data loaded: {curr_count} rows, {col_count} columns")
            self.count_label.config(text=f"Rows (original): {orig_count} | Rows (current): {curr_count} | Columns: {col_count}")
        else:
            self.status_label.config(text="No data loaded")
            self.count_label.config(text="Rows (original): 0 | Rows (current): 0 | Columns: 0")
    
    def _process_ui_queue(self):
        """Process UI updates from background threads."""
        try:
            while True:
                callback = self.ui_queue.get_nowait()
                callback()
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_ui_queue)
    
    def _queue_ui_update(self, callback: Callable):
        """Queue a UI update callback for thread-safe execution."""
        self.ui_queue.put(callback)
    
    def log(self, message: str, level: str = "INFO"):
        """Add a message to the activity log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} — {message}\n"
        
        def update_log():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        
        if threading.current_thread() == threading.main_thread():
            update_log()
        else:
            self._queue_ui_update(update_log)
    
    def upload_file(self):
        """Handle file upload with background loading and header row selection."""
        filetypes = [
            ("All supported", "*.xlsx;*.xls;*.csv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Excel or CSV file",
            filetypes=filetypes,
            initialdir=self.last_directory
        )

        if filename:
            self.last_directory = os.path.dirname(filename)
            
            file_ext = os.path.splitext(filename)[1].lower()
            selected_sheet = None
            
            # For Excel files, first ask which sheet to use
            if file_ext in ['.xlsx', '.xls']:
                try:
                    # Get sheet names to display to user
                    if file_ext == '.xlsx':
                        xl = pd.ExcelFile(filename, engine='openpyxl')
                    else:  # .xls
                        try:
                            xl = pd.ExcelFile(filename, engine='xlrd')
                        except ImportError:
                            xl = pd.ExcelFile(filename, engine='openpyxl')
                    
                    sheet_names = xl.sheet_names
                    
                    # Create a string with sheet names and indices
                    sheet_list = "\n".join([f"{i+1}. {name}" for i, name in enumerate(sheet_names)])
                    sheet_prompt = f"Select sheet to load (1-{len(sheet_names)}):\n{sheet_list}"
                    
                    # Ask user which sheet to use
                    sheet_selection = tk.simpledialog.askinteger(
                        "Sheet Selection",
                        sheet_prompt,
                        initialvalue=1,
                        minvalue=1,
                        maxvalue=len(sheet_names)
                    )
                    
                    if sheet_selection is None:
                        return  # User cancelled
                    
                    selected_sheet = sheet_names[sheet_selection-1]
                    self.log(f"Selected sheet: {selected_sheet}")
                    
                except Exception as e:
                    error_msg = f"Error reading Excel sheets: {str(e)}"
                    self.log(error_msg, "ERROR")
                    messagebox.showerror("Sheet Selection Error", error_msg)
                    return
            
            # Now ask for header row (default is 1)
            header_row = tk.simpledialog.askinteger(
                "Header Row",
                "Enter the row number containing column headers (1 for first row):",
                initialvalue=1,
                minvalue=1
            )
            if header_row is None:
                return  # User cancelled

            self.upload_btn.config(text="Loading...", state=tk.DISABLED)
            self.log(f"Loading file: {os.path.basename(filename)} (header row: {header_row})")
            
            # Pass selected_sheet to the loading thread
            threading.Thread(
                target=self._load_file_thread, 
                args=(filename, header_row, selected_sheet), 
                daemon=True
            ).start()

    def _load_file_thread(self, filename: str, header_row: int, selected_sheet=None):
        """Load file in background thread with header row selection."""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            header_idx = header_row - 1  # pandas uses zero-based index

            if file_ext == '.csv':
                df = pd.read_csv(filename, header=header_idx)
                self.log("File loaded as CSV")
            elif file_ext in ['.xlsx', '.xls']:
                # Load with the selected sheet
                if file_ext == '.xlsx':
                    df = pd.read_excel(filename, engine='openpyxl', header=header_idx, sheet_name=selected_sheet)
                    self.log(f"File loaded as Excel (.xlsx), sheet: {selected_sheet}")
                else:  # .xls
                    try:
                        df = pd.read_excel(filename, engine='xlrd', header=header_idx, sheet_name=selected_sheet)
                        self.log(f"File loaded as Excel (.xls), sheet: {selected_sheet}")
                    except ImportError:
                        df = pd.read_excel(filename, engine='openpyxl', header=header_idx, sheet_name=selected_sheet)
                        self.log(f"File loaded as Excel (.xls) using openpyxl fallback, sheet: {selected_sheet}")
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")

            if df.empty:
                raise ValueError("File is empty or contains no readable data")

            def update_data():
                self.original_df = df.copy()
                self.working_df = df.copy()
                self.filter_stack.clear()
                self._update_filter_stack_display()
                self._refresh_table()
                self._update_ui_state()
                self.upload_btn.config(text="Upload", state=tk.NORMAL)
                self.log(f"File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                self.log(f"Columns detected: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")

            self._queue_ui_update(update_data)

        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.log(error_msg, "ERROR")
            def reset_button():
                self.upload_btn.config(text="Upload", state=tk.NORMAL)
                messagebox.showerror("Load Error", error_msg)
            self._queue_ui_update(reset_button)
    
    def _refresh_table(self):
        """Refresh the data preview table."""
        # Clear existing items
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.working_df is None:
            return
        
        # Configure columns
        columns = list(self.working_df.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'tree headings'
        
        # Set column headings and widths
        self.data_tree.heading('#0', text='Row', anchor=tk.W)
        self.data_tree.column('#0', width=50, minwidth=30)
        
        for col in columns:
            self.data_tree.heading(col, text=col, anchor=tk.W)
            self.data_tree.column(col, width=120, minwidth=80)
        
        # Insert data (limit to first 5000 rows for performance)
        display_df = self.working_df.head(5000)
        
        for idx, (row_idx, row) in enumerate(display_df.iterrows()):
            values = []
            for col in columns:
                val = row[col]
                if pd.isna(val):
                    val = "NaN"
                elif isinstance(val, float):
                    val = f"{val:.6g}"  # Compact float representation
                else:
                    val = str(val)
                
                # Limit cell content length
                if len(val) > 50:
                    val = val[:47] + "..."
                
                values.append(val)
            
            self.data_tree.insert('', 'end', text=str(row_idx), values=values)
        
        if len(self.working_df) > 5000:
            self.log(f"Note: Displaying first 5,000 rows of {len(self.working_df)} total rows")
    
    def apply_filter(self):
        """Apply basic filter based on current selections."""
        if self.working_df is None:
            messagebox.showwarning("No Data", "Please load a file first.")
            return
        
        column = self.column_var.get()
        operator = self.operator_var.get()
        value = self.value_var.get().strip()
        
        if not column:
            messagebox.showwarning("Invalid Filter", "Please select a column.")
            return
        
        if not operator:
            messagebox.showwarning("Invalid Filter", "Please select an operator.")
            return
        
        if operator not in ["isnull", "notnull"] and not value:
            messagebox.showwarning("Invalid Filter", "Please enter a value.")
            return
        
        try:
            start_time = datetime.now()
            original_count = len(self.working_df)
            
            # Apply filter
            mask = self._create_filter_mask(column, operator, value)
            self.working_df = self.working_df[mask]
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            # Update UI
            self._refresh_table()
            self._update_ui_state()
            
            # Log filter application
            filter_desc = f"[{column}] [{operator}] [{value}]" if value else f"[{column}] [{operator}]"
            matched_count = len(self.working_df)
            self.log(f"Filter: {filter_desc} — matched {matched_count} rows (from {original_count}) in {duration:.1f}ms")
            
            # Add to filter stack
            filter_info = {
                'type': 'basic',
                'column': column,
                'operator': operator,
                'value': value,
                'description': filter_desc,
                'rows_before': original_count,
                'rows_after': matched_count
            }
            self.filter_stack.append(filter_info)
            self._update_filter_stack_display()
            
        except Exception as e:
            error_msg = f"Filter error: {str(e)}"
            self.log(error_msg, "ERROR")
            messagebox.showerror("Filter Error", error_msg)
    
    def _create_filter_mask(self, column: str, operator: str, value: str) -> pd.Series:
        """Create a boolean mask for filtering based on column, operator, and value."""
        series = self.working_df[column]
        case_insensitive = self.case_insensitive_var.get()
        include_nan = self.include_nan_var.get()
        
        # Handle null checks first
        if operator == "isnull":
            return series.isnull()
        elif operator == "notnull":
            return series.notnull()
        
        # For other operators, handle NaN values
        if not include_nan:
            non_null_mask = series.notnull()
        else:
            non_null_mask = pd.Series(True, index=series.index)
        
        # Numeric operations
        if operator in [">", ">=", "<", "<=", "==", "!="]:
            try:
                # Try to convert value to numeric
                numeric_value = pd.to_numeric(value, errors='raise')
                if operator == ">":
                    mask = series > numeric_value
                elif operator == ">=":
                    mask = series >= numeric_value
                elif operator == "<":
                    mask = series < numeric_value
                elif operator == "<=":
                    mask = series <= numeric_value
                elif operator == "==":
                    mask = series == numeric_value
                elif operator == "!=":
                    mask = series != numeric_value
                
                return mask & non_null_mask
                
            except (ValueError, TypeError):
                # Fall back to string comparison for non-numeric data
                if operator == "==":
                    if case_insensitive:
                        mask = series.astype(str).str.lower() == value.lower()
                    else:
                        mask = series.astype(str) == value
                elif operator == "!=":
                    if case_insensitive:
                        mask = series.astype(str).str.lower() != value.lower()
                    else:
                        mask = series.astype(str) != value
                else:
                    raise ValueError(f"Cannot apply numeric operator '{operator}' to non-numeric column '{column}' with value '{value}'")
                
                return mask & non_null_mask
        
        # Between operation
        elif operator == "between":
            try:
                values = [v.strip() for v in value.split(',')]
                if len(values) != 2:
                    raise ValueError("Between operator requires exactly two values separated by comma")
                
                try:
                    # Try numeric comparison first
                    min_val = pd.to_numeric(values[0], errors='raise')
                    max_val = pd.to_numeric(values[1], errors='raise')
                    
                    mask = (series >= min_val) & (series <= max_val)
                except (ValueError, TypeError):
                    # Fall back to string comparison for non-numeric data
                    if case_insensitive:
                        min_val = values[0].lower()
                        max_val = values[1].lower()
                        series_str = series.astype(str).str.lower()
                    else:
                        min_val = values[0]
                        max_val = values[1]
                        series_str = series.astype(str)
                    
                    mask = (series_str >= min_val) & (series_str <= max_val)
                
                return mask & non_null_mask
                
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid between values: {e}")
        
        # String operations
        elif operator in ["contains", "startswith", "endswith"]:
            # Convert to string if not already string type
            series = series.astype(str)
            
            if operator == "contains":
                if case_insensitive:
                    mask = series.str.contains(value, case=False, na=False, regex=False)
                else:
                    mask = series.str.contains(value, case=True, na=False, regex=False)
            elif operator == "startswith":
                # Create a case-insensitive version for startswith
                if case_insensitive:
                    # Manual implementation for case-insensitive startswith
                    mask = series.str.slice(0, len(value)).str.lower() == value.lower()
                else:
                    mask = series.str.startswith(value)
            elif operator == "endswith":
                # Create a case-insensitive version for endswith
                if case_insensitive:
                    # Manual implementation for case-insensitive endswith
                    series_len = series.str.len()
                    value_len = len(value)
                    mask = series.str.slice(series_len - value_len).str.lower() == value.lower()
                else:
                    mask = series.str.endswith(value)
            
            return mask & non_null_mask
        
        # Regex operation
        elif operator == "regex":
            if not hasattr(series, 'str'):
                series = series.astype(str)
            
            try:
                if case_insensitive:
                    mask = series.str.contains(value, case=False, na=False, regex=True)
                else:
                    mask = series.str.contains(value, case=True, na=False, regex=True)
                return mask & non_null_mask
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}")
        
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def run_script_filter(self):
        """Apply advanced filter using pandas query or lambda expression."""
        if self.working_df is None:
            messagebox.showwarning("No Data", "Please load a file first.")
            return
        
        script = self.script_text.get(1.0, tk.END).strip()
        if not script:
            messagebox.showwarning("Empty Script", "Please enter a filter expression.")
            return
        
        try:
            start_time = datetime.now()
            original_count = len(self.working_df)
            mode = self.script_mode_var.get()
            
            if mode == "query":
                # Use DataFrame.query()
                filtered_df = self.working_df.query(script)
            else:  # lambda mode
                # Create a restricted namespace for safety
                namespace = {
                    'df': self.working_df,
                    'pd': pd,
                    'np': np,
                    're': re,
                    # Add common functions but exclude dangerous ones
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'any': any,
                    'all': all,
                }
                
                # Execute lambda expression
                filter_func = eval(script, {"__builtins__": {}}, namespace)
                if not callable(filter_func):
                    raise ValueError("Lambda expression must be callable")
                
                mask = filter_func(self.working_df)
                if not isinstance(mask, (pd.Series, np.ndarray)) or mask.dtype != bool:
                    raise ValueError("Lambda function must return a boolean mask")
                
                filtered_df = self.working_df[mask]
            
            self.working_df = filtered_df
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000
            
            # Update UI
            self._refresh_table()
            self._update_ui_state()
            
            # Log script execution
            matched_count = len(self.working_df)
            script_preview = script[:50] + "..." if len(script) > 50 else script
            self.log(f"Script ({mode}): {script_preview} — matched {matched_count} rows (from {original_count}) in {duration:.1f}ms")
            
            # Add to filter stack
            filter_info = {
                'type': 'script',
                'mode': mode,
                'script': script,
                'description': f"Script ({mode}): {script_preview}",
                'rows_before': original_count,
                'rows_after': matched_count
            }
            self.filter_stack.append(filter_info)
            self._update_filter_stack_display()
            
        except Exception as e:
            error_msg = f"Script error: {str(e)}"
            self.log(error_msg, "ERROR")
            messagebox.showerror("Script Error", error_msg)
    
    def clear_filters(self):
        """Clear all filters and reset to original data."""
        if self.original_df is None:
            return
        
        self.working_df = self.original_df.copy()
        self.filter_stack.clear()
        
        # Clear UI inputs
        self.value_var.set("")
        self.script_text.delete(1.0, tk.END)
        
        # Update UI
        self._refresh_table()
        self._update_ui_state()
        self._update_filter_stack_display()
        
        self.log("All filters cleared - reset to original data")
    
    def undo_last_filter(self):
        """Undo the last applied filter."""
        if not self.filter_stack or self.original_df is None:
            return
        
        # Remove last filter from stack
        last_filter = self.filter_stack.pop()
        
        # Reapply all remaining filters from scratch
        self.working_df = self.original_df.copy()
        
        for filter_info in self.filter_stack:
            try:
                if filter_info['type'] == 'basic':
                    mask = self._create_filter_mask(
                        filter_info['column'], 
                        filter_info['operator'], 
                        filter_info['value']
                    )
                    self.working_df = self.working_df[mask]
                elif filter_info['type'] == 'script':
                    if filter_info['mode'] == 'query':
                        self.working_df = self.working_df.query(filter_info['script'])
                    else:  # lambda
                        namespace = {
                            'df': self.working_df,
                            'pd': pd,
                            'np': np,
                            're': re,
                            'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
                            'abs': abs, 'min': min, 'max': max, 'sum': sum, 'any': any, 'all': all,
                        }
                        filter_func = eval(filter_info['script'], {"__builtins__": {}}, namespace)
                        mask = filter_func(self.working_df)
                        self.working_df = self.working_df[mask]
            except Exception as e:
                self.log(f"Error reapplying filter: {e}", "ERROR")
                break
        
        # Update UI
        self._refresh_table()
        self._update_ui_state()
        self._update_filter_stack_display()
        
        self.log(f"Undid filter: {last_filter['description']}")
    
    def _update_filter_stack_display(self):
        """Update the filter stack listbox display."""
        self.filter_stack_listbox.delete(0, tk.END)
        
        for i, filter_info in enumerate(self.filter_stack):
            display_text = f"{i+1}. {filter_info['description']}"
            self.filter_stack_listbox.insert(tk.END, display_text)
        
        # Update undo button state
        self.undo_btn.config(state=tk.NORMAL if self.filter_stack else tk.DISABLED)
    
    def export_csv(self):
        """Export current working DataFrame to CSV."""
        if self.working_df is None:
            messagebox.showwarning("No Data", "No data to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save CSV file",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=self.last_directory
        )
        
        if filename:
            try:
                self.working_df.to_csv(filename, index=False)
                self.last_directory = os.path.dirname(filename)
                
                row_count = len(self.working_df)
                self.log(f"Exported {row_count} rows to: {os.path.basename(filename)}")
                messagebox.showinfo("Export Complete", f"Successfully exported {row_count} rows to CSV file.")
                
            except Exception as e:
                error_msg = f"Export error: {str(e)}"
                self.log(error_msg, "ERROR")
                messagebox.showerror("Export Error", error_msg)

    def export_excel_sheet(self):
        if self.working_df is None:
            messagebox.showwarning("No Data", "No data to export.")
            return
        filename = filedialog.askopenfilename(
            title="Select Excel file to add sheet",
            filetypes=[("Excel files", "*.xlsx")],
            initialdir=self.last_directory
        )
        if not filename:
            return
        sheet_name = simpledialog.askstring("Sheet Name", "Enter name for new sheet:", initialvalue="FilteredData")
        if not sheet_name:
            return
        try:
            with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                self.working_df.to_excel(writer, sheet_name=sheet_name, index=False)
            self.last_directory = os.path.dirname(filename)
            row_count = len(self.working_df)
            self.log(f"Exported {row_count} rows to new sheet '{sheet_name}' in: {os.path.basename(filename)}")
            messagebox.showinfo("Export Complete", f"Successfully exported {row_count} rows to sheet '{sheet_name}' in Excel file.")
        except Exception as e:
            error_msg = f"Export error: {str(e)}"
            self.log(error_msg, "ERROR")
            messagebox.showerror("Export Error", error_msg)
    
    def copy_log(self):
        """Copy the activity log to clipboard."""
        try:
            log_content = self.log_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.log("Activity log copied to clipboard")
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy log: {str(e)}")
    
    def calculate_statistics(self):
        """Calculate statistics on the current filtered dataset with advanced selection."""
        if self.working_df is None:
            messagebox.showwarning("No Data", "No data to calculate statistics on.")
            return
        
        # Identify numeric columns
        numeric_cols = self.working_df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            messagebox.showinfo("No Numeric Columns", "There are no numeric columns in the data to calculate statistics on.")
            return
        
        # Create statistics window
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Advanced Data Calculations")
        stats_window.geometry("900x700")
        stats_window.grab_set()  # Make the window modal
        
        # Create main notebook with tabs
        notebook = ttk.Notebook(stats_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Column-based calculations
        column_tab = ttk.Frame(notebook, padding=10)
        notebook.add(column_tab, text="Column Calculations")
        
        # Tab 2: Row-based calculations
        row_tab = ttk.Frame(notebook, padding=10)
        notebook.add(row_tab, text="Row Calculations")
        
        # Tab 3: Custom Formula
        formula_tab = ttk.Frame(notebook, padding=10)
        notebook.add(formula_tab, text="Custom Formula")
        
        # Tab 4: Results viewer
        results_tab = ttk.Frame(notebook, padding=10)
        notebook.add(results_tab, text="Results")
        
        #
        # COLUMN CALCULATIONS TAB
        #
        col_frame = ttk.LabelFrame(column_tab, text="Select Columns for Calculation", padding=5)
        col_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Column selection with scrollable frame
        col_canvas = tk.Canvas(col_frame, height=150)
        col_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        col_scrollbar = ttk.Scrollbar(col_frame, orient=tk.VERTICAL, command=col_canvas.yview)
        col_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        col_canvas.configure(yscrollcommand=col_scrollbar.set)
        col_canvas.bind('<Configure>', lambda e: col_canvas.configure(scrollregion=col_canvas.bbox("all")))
        
        col_select_frame = ttk.Frame(col_canvas)
        col_canvas.create_window((0, 0), window=col_select_frame, anchor="nw")
        
        # Column checkboxes with data type info
        column_vars = {}
        for i, col in enumerate(self.working_df.columns):
            var = tk.BooleanVar(value=False)
            column_vars[col] = var
            
            # Determine if column is numeric
            is_numeric = col in numeric_cols
            
            # Create frame for each column
            col_item_frame = ttk.Frame(col_select_frame)
            col_item_frame.grid(row=i//3, column=i%3, sticky="w", padx=5, pady=2)
            
            check = ttk.Checkbutton(col_item_frame, text=col, variable=var)
            check.grid(row=0, column=0, sticky="w")
            
            dtype_label = ttk.Label(col_item_frame, 
                                    text=f"({self.working_df[col].dtype})", 
                                    font=("TkDefaultFont", 8),
                                    foreground="blue" if is_numeric else "gray")
            dtype_label.grid(row=0, column=1, padx=(2, 0), sticky="w")
        
        # Column operations frame
        col_op_frame = ttk.LabelFrame(column_tab, text="Column Operations", padding=5)
        col_op_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Basic operations
        basic_op_frame = ttk.Frame(col_op_frame)
        basic_op_frame.pack(fill=tk.X, pady=5)
        
        col_sum_var = tk.BooleanVar(value=True)
        col_mean_var = tk.BooleanVar(value=True)
        col_min_var = tk.BooleanVar(value=True)
        col_max_var = tk.BooleanVar(value=True)
        col_median_var = tk.BooleanVar(value=False)
        col_std_var = tk.BooleanVar(value=False)
        col_count_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(basic_op_frame, text="Sum", variable=col_sum_var).grid(row=0, column=0, padx=5, sticky="w")
        ttk.Checkbutton(basic_op_frame, text="Average", variable=col_mean_var).grid(row=0, column=1, padx=5, sticky="w")
        ttk.Checkbutton(basic_op_frame, text="Minimum", variable=col_min_var).grid(row=0, column=2, padx=5, sticky="w")
        ttk.Checkbutton(basic_op_frame, text="Maximum", variable=col_max_var).grid(row=0, column=3, padx=5, sticky="w")
        ttk.Checkbutton(basic_op_frame, text="Median", variable=col_median_var).grid(row=1, column=0, padx=5, sticky="w")
        ttk.Checkbutton(basic_op_frame, text="Std Dev", variable=col_std_var).grid(row=1, column=1, padx=5, sticky="w")
        ttk.Checkbutton(basic_op_frame, text="Count", variable=col_count_var).grid(row=1, column=2, padx=5, sticky="w")
        
        # Advanced column operations
        adv_col_op_frame = ttk.LabelFrame(column_tab, text="Advanced Column Operations", padding=5)
        adv_col_op_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Column arithmetic
        col_arith_frame = ttk.Frame(adv_col_op_frame)
        col_arith_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(col_arith_frame, text="Column A:").grid(row=0, column=0, padx=5)
        col_a_var = tk.StringVar()
        col_a_combo = ttk.Combobox(col_arith_frame, textvariable=col_a_var, state="readonly", width=20)
        col_a_combo['values'] = numeric_cols
        if numeric_cols:
            col_a_var.set(numeric_cols[0])
        col_a_combo.grid(row=0, column=1, padx=5)
        
        op_var = tk.StringVar(value="+")
        op_combo = ttk.Combobox(col_arith_frame, textvariable=op_var, state="readonly", width=5)
        op_combo['values'] = ["+", "-", "*", "/", "max", "min"]
        op_combo.grid(row=0, column=2, padx=5)
        
        ttk.Label(col_arith_frame, text="Column B:").grid(row=0, column=3, padx=5)
        col_b_var = tk.StringVar()
        col_b_combo = ttk.Combobox(col_arith_frame, textvariable=col_b_var, state="readonly", width=20)
        col_b_combo['values'] = numeric_cols
        if len(numeric_cols) > 1:
            col_b_var.set(numeric_cols[1])
        else:
            col_b_var.set(numeric_cols[0])
        col_b_combo.grid(row=0, column=4, padx=5)
        
        ttk.Label(col_arith_frame, text="Result column name:").grid(row=1, column=0, padx=5, pady=(10, 0))
        result_name_var = tk.StringVar(value="Result")
        result_name_entry = ttk.Entry(col_arith_frame, textvariable=result_name_var, width=20)
        result_name_entry.grid(row=1, column=1, padx=5, pady=(10, 0))
        
        calc_column_btn = ttk.Button(col_arith_frame, text="Calculate & Add Column")
        calc_column_btn.grid(row=1, column=2, columnspan=3, padx=5, pady=(10, 0), sticky="w")
        
        #
        # ROW CALCULATIONS TAB
        #
        row_top_frame = ttk.Frame(row_tab)
        row_top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Row selection methods
        row_select_frame = ttk.LabelFrame(row_top_frame, text="Row Selection Method", padding=5)
        row_select_frame.pack(fill=tk.X)
        
        row_selection_method = tk.StringVar(value="all")
        ttk.Radiobutton(row_select_frame, text="All rows (current filter)", 
                        variable=row_selection_method, value="all").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Radiobutton(row_select_frame, text="First N rows", 
                        variable=row_selection_method, value="first_n").grid(row=0, column=1, sticky="w", padx=5)
        ttk.Radiobutton(row_select_frame, text="By row indices", 
                        variable=row_selection_method, value="indices").grid(row=0, column=2, sticky="w", padx=5)
        ttk.Radiobutton(row_select_frame, text="By condition", 
                        variable=row_selection_method, value="condition").grid(row=0, column=3, sticky="w", padx=5)
        
        # Row selection options frame (will show/hide based on selection method)
        row_options_frame = ttk.Frame(row_tab)
        row_options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # For "first_n" option
        first_n_frame = ttk.Frame(row_options_frame)
        n_rows_var = tk.StringVar(value="10")
        ttk.Label(first_n_frame, text="Number of rows:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(first_n_frame, textvariable=n_rows_var, width=10).pack(side=tk.LEFT, padx=5)
        
        # For "indices" option
        indices_frame = ttk.Frame(row_options_frame)
        indices_var = tk.StringVar(value="0, 1, 2")
        ttk.Label(indices_frame, text="Row indices (comma-separated):").pack(side=tk.LEFT, padx=5)
        ttk.Entry(indices_frame, textvariable=indices_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # For "condition" option
        condition_frame = ttk.Frame(row_options_frame)
        condition_var = tk.StringVar()
        ttk.Label(condition_frame, text="Filter condition:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(condition_frame, textvariable=condition_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Label(condition_frame, text="(e.g., `Column A` > 100)").pack(side=tk.LEFT, padx=5)
        
        # Function to show the right frame based on row selection method
        def update_row_selection_ui(*args):
            # Hide all frames first
            first_n_frame.pack_forget()
            indices_frame.pack_forget()
            condition_frame.pack_forget()
            
            # Show the appropriate frame
            if row_selection_method.get() == "first_n":
                first_n_frame.pack(fill=tk.X, pady=5)
            elif row_selection_method.get() == "indices":
                indices_frame.pack(fill=tk.X, pady=5)
            elif row_selection_method.get() == "condition":
                condition_frame.pack(fill=tk.X, pady=5)
        
        # Track changes to row selection method
        row_selection_method.trace("w", update_row_selection_ui)
        
        # Initialize with the default selection
        update_row_selection_ui()
        
        # Row operations frame
        row_op_frame = ttk.LabelFrame(row_tab, text="Row Operations", padding=5)
        row_op_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Operation on selected rows
        row_op_inner_frame = ttk.Frame(row_op_frame)
        row_op_inner_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(row_op_inner_frame, text="Apply operation to column:").grid(row=0, column=0, padx=5)
        row_col_var = tk.StringVar()
        row_col_combo = ttk.Combobox(row_op_inner_frame, textvariable=row_col_var, state="readonly", width=20)
        row_col_combo['values'] = numeric_cols
        if numeric_cols:
            row_col_var.set(numeric_cols[0])
        row_col_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(row_op_inner_frame, text="Operation:").grid(row=0, column=2, padx=5)
        row_op_var = tk.StringVar(value="sum")
        row_op_combo = ttk.Combobox(row_op_inner_frame, textvariable=row_op_var, state="readonly", width=10)
        row_op_combo['values'] = ["sum", "mean", "min", "max", "median", "std", "count"]
        row_op_combo.grid(row=0, column=3, padx=5)
        
        # Preview button to show selected rows
        preview_btn = ttk.Button(row_op_inner_frame, text="Preview Selected Rows")
        preview_btn.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="w")
        
        # Calculate button for row operations
        calc_row_btn = ttk.Button(row_op_inner_frame, text="Calculate on Selected Rows")
        calc_row_btn.grid(row=1, column=2, columnspan=2, pady=(10, 0), sticky="w")
        
        #
        # CUSTOM FORMULA TAB
        #
        formula_frame = ttk.LabelFrame(formula_tab, text="Custom Formula", padding=5)
        formula_frame.pack(fill=tk.BOTH, expand=True)
        
        # Formula entry with helper text
        ttk.Label(formula_frame, text="Enter a pandas-compatible formula:").pack(anchor="w", pady=(0, 5))
        
        formula_var = tk.StringVar()
        formula_entry = ttk.Entry(formula_frame, textvariable=formula_var, width=60)
        formula_entry.pack(fill=tk.X, pady=(0, 5))
        
        # Helper text
        helper_text = """
Examples:
1. df['Column A'] + df['Column B']  # Add two columns
2. df['Column A'].sum()  # Sum of a column
3. (df['Column A'] > 100).sum()  # Count values over 100
4. df.loc[df['Column A'] > 100, 'Column B'].mean()  # Average of Column B where Column A > 100

Available functions: sum(), mean(), min(), max(), median(), std(), count(), etc.
Use df to refer to the current filtered dataset.
        """
        helper_label = ttk.Label(formula_frame, text=helper_text, justify="left", wraplength=600)
        helper_label.pack(anchor="w", pady=10)
        
        # Available columns
        ttk.Label(formula_frame, text="Available columns:").pack(anchor="w", pady=(10, 5))
        
        columns_text = ", ".join([f"'{col}'" for col in self.working_df.columns])
        columns_label = ttk.Label(formula_frame, text=columns_text, wraplength=600, justify="left")
        columns_label.pack(anchor="w")
        
        # Execute button
        exec_formula_btn = ttk.Button(formula_frame, text="Execute Formula")
        exec_formula_btn.pack(pady=10)
        
        #
        # RESULTS TAB
        #
        results_frame = ttk.Frame(results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results text area
        results_text = tk.Text(results_frame, wrap=tk.WORD, font=("Courier", 10))
        results_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for results
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=results_text.yview)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        results_text.configure(yscrollcommand=results_scroll.set)
        
        # Button frame for results tab
        results_btn_frame = ttk.Frame(results_tab)
        results_btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(results_btn_frame, text="Copy Results", 
                  command=lambda: self.copy_to_clipboard(results_text.get(1.0, tk.END))).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_btn_frame, text="Export Results", 
                  command=lambda: self._export_results(results_text.get(1.0, tk.END))).pack(side=tk.LEFT, padx=5)
        ttk.Button(results_btn_frame, text="Clear Results", 
                  command=lambda: results_text.delete(1.0, tk.END)).pack(side=tk.LEFT, padx=5)
        
        #
        # IMPLEMENTATION OF CALCULATION FUNCTIONS
        #
        
        # Function to perform column calculations
        def do_column_calculations():
            try:
                results_text.delete(1.0, tk.END)
                
                # Get selected columns
                selected_cols = [col for col, var in column_vars.items() if var.get()]
                
                if not selected_cols:
                    results_text.insert(tk.END, "No columns selected. Please select at least one column.")
                    notebook.select(results_tab)
                    return
                
                # Filter to only numeric columns if operations require it
                numeric_selected = [col for col in selected_cols if col in numeric_cols]
                
                if not numeric_selected and any([
                    col_sum_var.get(), col_mean_var.get(), col_min_var.get(), 
                    col_max_var.get(), col_median_var.get(), col_std_var.get()
                ]):
                    results_text.insert(tk.END, "Warning: No numeric columns selected. Some statistics may be invalid.\n\n")
                
                # Create header
                results_text.insert(tk.END, f"Column Statistics on {len(self.working_df)} rows\n")
                results_text.insert(tk.END, "=" * 50 + "\n\n")
                
                # Perform operations on each selected column
                for col in selected_cols:
                    results_text.insert(tk.END, f"Column: {col}\n")
                    results_text.insert(tk.END, "-" * 30 + "\n")
                    
                    series = self.working_df[col]
                    
                    if col_count_var.get():
                        count = len(series)
                        non_null = series.count()
                        results_text.insert(tk.END, f"Count: {count} (Non-null: {non_null})\n")
                    
                    # Only do numeric operations on numeric columns
                    if col in numeric_cols:
                        if col_sum_var.get():
                            total = series.sum()
                            results_text.insert(tk.END, f"Sum: {total:,.4f}\n")
                        
                        if col_mean_var.get():
                            mean = series.mean()
                            results_text.insert(tk.END, f"Mean: {mean:,.4f}\n")
                        
                        if col_min_var.get():
                            min_val = series.min()
                            results_text.insert(tk.END, f"Min: {min_val:,.4f}\n")
                        
                        if col_max_var.get():
                            max_val = series.max()
                            results_text.insert(tk.END, f"Max: {max_val:,.4f}\n")
                        
                        if col_median_var.get():
                            median = series.median()
                            results_text.insert(tk.END, f"Median: {median:,.4f}\n")
                        
                        if col_std_var.get():
                            std = series.std()
                            results_text.insert(tk.END, f"Std Dev: {std:,.4f}\n")
                    
                    results_text.insert(tk.END, "\n")
                
                # Switch to results tab
                notebook.select(results_tab)
                
            except Exception as e:
                results_text.insert(tk.END, f"Error performing column calculations: {str(e)}")
                traceback.print_exc()
                notebook.select(results_tab)
        
        # Function to preview selected rows
        def preview_selected_rows():
            try:
                results_text.delete(1.0, tk.END)
                
                # Get selected rows based on method
                method = row_selection_method.get()
                selected_rows = None
                
                if method == "all":
                    selected_rows = self.working_df
                    description = "All rows in current filter"
                
                elif method == "first_n":
                    try:
                        n = int(n_rows_var.get())
                        selected_rows = self.working_df.head(n)
                        description = f"First {n} rows"
                    except ValueError:
                        results_text.insert(tk.END, "Error: Please enter a valid number for row count.")
                        notebook.select(results_tab)
                        return
                
                elif method == "indices":
                    try:
                        # Parse indices, handling various formats
                        indices_str = indices_var.get().strip()
                        indices = []
                        
                        # Handle comma-separated values, may include ranges like "1-5"
                        for part in indices_str.split(','):
                            part = part.strip()
                            if '-' in part:
                                # It's a range
                                start, end = part.split('-')
                                indices.extend(range(int(start), int(end) + 1))
                            else:
                                # It's a single number
                                if part:  # Skip empty parts
                                    indices.append(int(part))
                        
                        # Filter to valid indices
                        valid_indices = [i for i in indices if 0 <= i < len(self.working_df)]
                        
                        if not valid_indices:
                            results_text.insert(tk.END, "Error: No valid indices provided. Remember that indices are 0-based.")
                            notebook.select(results_tab)
                            return
                        
                        selected_rows = self.working_df.iloc[valid_indices]
                        description = f"Selected rows by indices: {indices_str}"
                    
                    except Exception as e:
                        results_text.insert(tk.END, f"Error parsing row indices: {str(e)}")
                        notebook.select(results_tab)
                        return
                
                elif method == "condition":
                    try:
                        condition = condition_var.get()
                        if not condition:
                            results_text.insert(tk.END, "Error: Please enter a filter condition.")
                            notebook.select(results_tab)
                            return
                        
                        # Apply the condition
                        selected_rows = self.working_df.query(condition)
                        
                        if len(selected_rows) == 0:
                            results_text.insert(tk.END, f"Warning: No rows match the condition: {condition}\n\n")
                        
                        description = f"Rows matching condition: {condition}"
                    
                    except Exception as e:
                        results_text.insert(tk.END, f"Error applying condition: {str(e)}")
                        notebook.select(results_tab)
                        return
                
                # Display preview
                results_text.insert(tk.END, f"Row Selection Preview: {description}\n")
                results_text.insert(tk.END, "=" * 50 + "\n\n")
                
                results_text.insert(tk.END, f"Selected {len(selected_rows)} rows out of {len(self.working_df)} total rows.\n\n")
                
                # Display the first few rows
                max_preview = 10
                preview_rows = selected_rows.head(max_preview)
                
                # Format as a table
                results_text.insert(tk.END, preview_rows.to_string())
                
                if len(selected_rows) > max_preview:
                    results_text.insert(tk.END, f"\n\n... and {len(selected_rows) - max_preview} more rows.")
                
                # Switch to results tab
                notebook.select(results_tab)
                
            except Exception as e:
                results_text.insert(tk.END, f"Error previewing rows: {str(e)}")
                traceback.print_exc()
                notebook.select(results_tab)
        
        # Function to calculate on selected rows
        def calculate_on_rows():
            try:
                results_text.delete(1.0, tk.END)
                
                # Get selected rows based on method
                method = row_selection_method.get()
                selected_rows = None
                
                if method == "all":
                    selected_rows = self.working_df
                    description = "All rows in current filter"
                
                elif method == "first_n":
                    try:
                        n = int(n_rows_var.get())
                        selected_rows = self.working_df.head(n)
                        description = f"First {n} rows"
                    except ValueError:
                        results_text.insert(tk.END, "Error: Please enter a valid number for row count.")
                        notebook.select(results_tab)
                        return
                
                elif method == "indices":
                    try:
                        # Parse indices, handling various formats
                        indices_str = indices_var.get().strip()
                        indices = []
                        
                        # Handle comma-separated values, may include ranges like "1-5"
                        for part in indices_str.split(','):
                            part = part.strip()
                            if '-' in part:
                                # It's a range
                                start, end = part.split('-')
                                indices.extend(range(int(start), int(end) + 1))
                            else:
                                # It's a single number
                                if part:  # Skip empty parts
                                    indices.append(int(part))
                        
                        # Filter to valid indices
                        valid_indices = [i for i in indices if 0 <= i < len(self.working_df)]
                        
                        if not valid_indices:
                            results_text.insert(tk.END, "Error: No valid indices provided. Remember that indices are 0-based.")
                            notebook.select(results_tab)
                            return
                        
                        selected_rows = self.working_df.iloc[valid_indices]
                        description = f"Selected rows by indices: {indices_str}"
                    
                    except Exception as e:
                        results_text.insert(tk.END, f"Error parsing row indices: {str(e)}")
                        notebook.select(results_tab)
                        return
                
                elif method == "condition":
                    try:
                        condition = condition_var.get()
                        if not condition:
                            results_text.insert(tk.END, "Error: Please enter a filter condition.")
                            notebook.select(results_tab)
                            return
                        
                        # Apply the condition
                        selected_rows = self.working_df.query(condition)
                        
                        if len(selected_rows) == 0:
                            results_text.insert(tk.END, f"Warning: No rows match the condition: {condition}\n\n")
                        
                        description = f"Rows matching condition: {condition}"
                    
                    except Exception as e:
                        results_text.insert(tk.END, f"Error applying condition: {str(e)}")
                        notebook.select(results_tab)
                        return
                
                # Get selected column and operation
                column = row_col_var.get()
                operation = row_op_var.get()
                
                # Check if column is numeric for certain operations
                if column not in numeric_cols and operation not in ["count"]:
                    results_text.insert(tk.END, f"Warning: Column '{column}' is not numeric. Some operations may not work as expected.\n\n")
                
                # Perform the operation
                series = selected_rows[column]
                result = None
                
                if operation == "sum":
                    result = series.sum()
                elif operation == "mean":
                    result = series.mean()
                elif operation == "min":
                    result = series.min()
                elif operation == "max":
                    result = series.max()
                elif operation == "median":
                    result = series.median()
                elif operation == "std":
                    result = series.std()
                elif operation == "count":
                    result = series.count()
                
                # Display results
                results_text.insert(tk.END, f"Row Calculation Results\n")
                results_text.insert(tk.END, "=" * 50 + "\n\n")
                
                results_text.insert(tk.END, f"Selection: {description}\n")
                results_text.insert(tk.END, f"Column: {column}\n")
                results_text.insert(tk.END, f"Operation: {operation}\n")
                results_text.insert(tk.END, f"Result: {result:,.4f}\n\n")
                
                results_text.insert(tk.END, f"(Calculated on {len(selected_rows)} rows)")
                
                # Switch to results tab
                notebook.select(results_tab)
                
            except Exception as e:
                results_text.insert(tk.END, f"Error calculating on rows: {str(e)}")
                traceback.print_exc()
                notebook.select(results_tab)
        
        # Function to calculate and add a column
        def calculate_and_add_column():
            try:
                # Get column selections and operation
                col_a = col_a_var.get()
                col_b = col_b_var.get()
                operation = op_var.get()
                result_name = result_name_var.get()
                
                if not result_name:
                    messagebox.showwarning("Invalid Name", "Please enter a name for the result column.")
                    return
                
                # Verify columns are numeric
                if col_a not in numeric_cols or col_b not in numeric_cols:
                    messagebox.showwarning("Non-numeric Columns", 
                                          f"One or both columns are not numeric. Operations may not work as expected.")
                
                # Calculate the new column
                if operation == "+":
                    self.working_df[result_name] = self.working_df[col_a] + self.working_df[col_b]
                elif operation == "-":
                    self.working_df[result_name] = self.working_df[col_a] - self.working_df[col_b]
                elif operation == "*":
                    self.working_df[result_name] = self.working_df[col_a] * self.working_df[col_b]
                elif operation == "/":
                    # Handle division by zero
                    self.working_df[result_name] = self.working_df[col_a] / self.working_df[col_b].replace(0, float('nan'))
                elif operation == "max":
                    self.working_df[result_name] = self.working_df[[col_a, col_b]].max(axis=1)
                elif operation == "min":
                    self.working_df[result_name] = self.working_df[[col_a, col_b]].min(axis=1)
                
                # Update the preview
                self._refresh_table()
                
                # Add new column to numeric columns list if it's not there
                if result_name not in numeric_cols:
                    numeric_cols.append(result_name)
                    
                    # Update dropdowns
                    col_a_combo['values'] = numeric_cols
                    col_b_combo['values'] = numeric_cols
                    row_col_combo['values'] = numeric_cols
                
                # Show result in results tab
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, f"Column Calculation Result\n")
                results_text.insert(tk.END, "=" * 50 + "\n\n")
                
                results_text.insert(tk.END, f"Created new column: {result_name}\n")
                results_text.insert(tk.END, f"Formula: {col_a} {operation} {col_b}\n\n")
                
                # Show preview of the new column
                preview = self.working_df[[col_a, col_b, result_name]].head(10)
                results_text.insert(tk.END, preview.to_string())
                
                if len(self.working_df) > 10:
                    results_text.insert(tk.END, f"\n\n... and {len(self.working_df) - 10} more rows.")
                
                # Switch to results tab
                notebook.select(results_tab)
                
                messagebox.showinfo("Column Added", f"Created new column '{result_name}' with {len(self.working_df)} values.")
                
            except Exception as e:
                results_text.delete(1.0, tk.END)
                results_text.insert(tk.END, f"Error creating column: {str(e)}")
                traceback.print_exc()
                notebook.select(results_tab)
        
        # Function to execute custom formula
        def execute_formula():
            try:
                results_text.delete(1.0, tk.END)
                
                formula = formula_var.get()
                if not formula:
                    results_text.insert(tk.END, "Error: Please enter a formula.")
                    notebook.select(results_tab)
                    return
                
                # Create a namespace for evaluation
                namespace = {
                    'df': self.working_df,
                    'pd': pd,
                    'np': np,
                    # Add common functions but exclude dangerous ones
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'any': any,
                    'all': all,
                }
                
                # Execute the formula
                start_time = datetime.now()
                result = eval(formula, {"__builtins__": {}}, namespace)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds() * 1000
                
                # Display results
                results_text.insert(tk.END, f"Custom Formula Result\n")
                results_text.insert(tk.END, "=" * 50 + "\n\n")
                
                results_text.insert(tk.END, f"Formula: {formula}\n")
                results_text.insert(tk.END, f"Execution time: {duration:.1f}ms\n\n")
                
                # Format the result based on its type
                if isinstance(result, pd.DataFrame):
                    results_text.insert(tk.END, f"Result is a DataFrame with shape {result.shape}:\n\n")
                    preview = result.head(20)
                    results_text.insert(tk.END, preview.to_string())
                    
                    if len(result) > 20:
                        results_text.insert(tk.END, f"\n\n... and {len(result) - 20} more rows.")
                
                elif isinstance(result, pd.Series):
                    results_text.insert(tk.END, f"Result is a Series with length {len(result)}:\n\n")
                    preview = result.head(20)
                    results_text.insert(tk.END, preview.to_string())
                    
                    if len(result) > 20:
                        results_text.insert(tk.END, f"\n\n... and {len(result) - 20} more values.")
                
                else:
                    # For scalar results
                    if isinstance(result, (int, float)):
                        results_text.insert(tk.END, f"Result: {result:,.4f}")
                    else:
                        results_text.insert(tk.END, f"Result: {result}")
                
                # Switch to results tab
                notebook.select(results_tab)
                
            except Exception as e:
                results_text.insert(tk.END, f"Error executing formula: {str(e)}")
                traceback.print_exc()
                notebook.select(results_tab)
        
        # Connect buttons to functions
        ttk.Button(column_tab, text="Calculate Column Statistics", 
                  command=do_column_calculations).pack(anchor="w", pady=10)
        
        preview_btn.config(command=preview_selected_rows)
        calc_row_btn.config(command=calculate_on_rows)
        calc_column_btn.config(command=calculate_and_add_column)
        exec_formula_btn.config(command=execute_formula)
        
        # Bottom button frame for the whole window
        bottom_frame = ttk.Frame(stats_window)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(bottom_frame, text="Close", command=stats_window.destroy).pack(side=tk.RIGHT)
    
    def _export_results(self, results_text):
        """Export calculation results to a file."""
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir=self.last_directory
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(results_text)
                self.last_directory = os.path.dirname(filename)
                messagebox.showinfo("Export Complete", f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results: {str(e)}")
    
    def clear_log(self):
        """Clear the activity log."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Add a fresh start message
        self.log("Activity log cleared")


def main():
    """Main entry point for the application."""
    # Check for required dependencies
    try:
        import pandas as pd
        import openpyxl
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        print("Please install required packages:")
        print("pip install pandas openpyxl")
        sys.exit(1)
    
    # Create and run the application
    root = tk.Tk()
    app = ExcelFilterApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()