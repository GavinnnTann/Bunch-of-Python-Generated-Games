#!/usr/bin/env python3
"""
Excel Filter - Tkinter Application

A comprehensive single-file Python application for uploading, filtering, and exporting Excel/CSV data
with a full-featured Tkinter GUI including data preview, basic/advanced filtering, and activity logging.

Requirements: pandas, openpyxl, tkinter (built-in)
Optional: xlrd (for legacy .xls files)

Author: Claude
Date: 2025-09-02
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import os
import sys
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
    
    def _update_ui_state(self):
        """Update the UI state based on current data availability."""
        has_data = self.working_df is not None
        
        self.filter_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.clear_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.export_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.run_script_btn.config(state=tk.NORMAL if has_data else tk.DISABLED)
        self.undo_btn.config(state=tk.NORMAL if self.filter_stack else tk.DISABLED)
        
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
            # Ask user for header row (default is 1)
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
            threading.Thread(target=self._load_file_thread, args=(filename, header_row), daemon=True).start()

    def _load_file_thread(self, filename: str, header_row: int):
        """Load file in background thread with header row selection."""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            header_idx = header_row - 1  # pandas uses zero-based index

            if file_ext == '.csv':
                df = pd.read_csv(filename, header=header_idx)
                self.log("File loaded as CSV")
            elif file_ext in ['.xlsx', '.xls']:
                if file_ext == '.xlsx':
                    df = pd.read_excel(filename, engine='openpyxl', header=header_idx)
                    self.log("File loaded as Excel (.xlsx)")
                else:  # .xls
                    try:
                        df = pd.read_excel(filename, engine='xlrd', header=header_idx)
                        self.log("File loaded as Excel (.xls)")
                    except ImportError:
                        df = pd.read_excel(filename, engine='openpyxl', header=header_idx)
                        self.log("File loaded as Excel (.xls) using openpyxl fallback")
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
                    if case_insensitive and hasattr(series, 'str'):
                        mask = series.astype(str).str.lower() == value.lower()
                    else:
                        mask = series.astype(str) == value
                elif operator == "!=":
                    if case_insensitive and hasattr(series, 'str'):
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
                
                min_val = pd.to_numeric(values[0], errors='raise')
                max_val = pd.to_numeric(values[1], errors='raise')
                
                mask = (series >= min_val) & (series <= max_val)
                return mask & non_null_mask
                
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid between values: {e}")
        
        # String operations
        elif operator in ["contains", "startswith", "endswith"]:
            if not hasattr(series, 'str'):
                # Convert to string if not already string type
                series = series.astype(str)
            
            if case_insensitive:
                series_str = series.str.lower()
                value_str = value.lower()
            else:
                series_str = series.str
                value_str = value
            
            if operator == "contains":
                mask = series_str.contains(value_str, na=False, regex=False)
            elif operator == "startswith":
                mask = series_str.startswith(value_str, na=False)
            elif operator == "endswith":
                mask = series_str.endswith(value_str, na=False)
            
            return mask & non_null_mask
        
        # Regex operation
        elif operator == "regex":
            if not hasattr(series, 'str'):
                series = series.astype(str)
            
            try:
                flags = re.IGNORECASE if case_insensitive else 0
                mask = series.str.contains(value, na=False, regex=True, flags=flags)
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
    
    def copy_log(self):
        """Copy the activity log to clipboard."""
        try:
            log_content = self.log_text.get(1.0, tk.END)
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.log("Activity log copied to clipboard")
        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy log: {str(e)}")
    
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