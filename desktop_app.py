"""Tkinter desktop GUI for the photo -> printable-mesh pipeline.

This is the entry point PyInstaller packages into anatomy3d-print.exe.
"""
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from anatomy3d.pipeline import run_pipeline


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("anatomy3d-print")
        self.geometry("520x320")

        self.image_path = tk.StringVar()
        self.height_mm = tk.StringVar(value="150")
        self.out_path = tk.StringVar(value="output/figure.stl")
        self.log_queue = queue.Queue()

        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Photo:", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.image_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=self._pick_image).pack(side="left")

        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Height (mm):", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.height_mm, width=10).pack(side="left")

        row = ttk.Frame(self)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Output STL:", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.out_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=self._pick_out_path).pack(side="left")

        self.run_button = ttk.Button(self, text="Generate model", command=self._start_run)
        self.run_button.pack(pady=10)

        self.log = tk.Text(self, height=10, state="disabled")
        self.log.pack(fill="both", expand=True, **pad)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if path:
            self.image_path.set(path)

    def _pick_out_path(self):
        path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL", "*.stl")])
        if path:
            self.out_path.set(path)

    def _append_log(self, text: str):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            self._append_log(self.log_queue.get_nowait())
        self.after(100, self._drain_log_queue)

    def _start_run(self):
        if not self.image_path.get():
            messagebox.showerror("Missing input", "Pick a photo first.")
            return
        self.run_button.configure(state="disabled")
        self.log_queue.put("Running... this can take a minute or two on CPU.")
        threading.Thread(target=self._run_pipeline_thread, daemon=True).start()

    def _run_pipeline_thread(self):
        try:
            height = float(self.height_mm.get() or 150.0)
            run_pipeline(self.image_path.get(), self.out_path.get(), target_height_mm=height)
            self.log_queue.put(f"Done. Wrote {self.out_path.get()}")
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI log
            self.log_queue.put(f"Error: {exc}")
        finally:
            self.run_button.configure(state="normal")


if __name__ == "__main__":
    App().mainloop()
