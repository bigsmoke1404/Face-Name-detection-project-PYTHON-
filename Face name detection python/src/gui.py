"""
gui.py
------
Full CustomTkinter dark-themed GUI for the Facial Recognition System.

Layout
------
  Left sidebar  — navigation buttons + status indicators
  Right content — tabbed views: Live Feed | Users | History | Attendance | Settings
"""

import logging
import queue
import threading
import time
import tkinter as tk
from datetime import datetime, date
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog
from typing import Optional

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

import attendance
import camera as cam_module
import database as db
import face_detector
import face_recognition_engine as fre
import settings
import utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT      = "#00d4ff"
ACCENT_DARK = "#0099cc"
BG_DARK     = "#0d1117"
BG_CARD     = "#161b22"
BG_PANEL    = "#1c2128"
TEXT_PRI    = "#e6edf3"
TEXT_SEC    = "#8b949e"
SUCCESS     = "#3fb950"
WARNING     = "#d29922"
DANGER      = "#f85149"
FONT_FAMILY = "Segoe UI"


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

class SidebarButton(ctk.CTkButton):
    def __init__(self, master, text, icon="", command=None, **kwargs):
        super().__init__(
            master,
            text=f"  {icon}  {text}",
            anchor="w",
            height=42,
            corner_radius=8,
            fg_color="transparent",
            text_color=TEXT_PRI,
            hover_color=BG_PANEL,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            command=command,
            **kwargs,
        )

    def set_active(self, active: bool):
        if active:
            self.configure(fg_color=ACCENT_DARK, text_color="white")
        else:
            self.configure(fg_color="transparent", text_color=TEXT_PRI)


class StatusDot(ctk.CTkLabel):
    """Coloured status indicator dot."""

    def __init__(self, master, **kwargs):
        super().__init__(master, text="●", font=ctk.CTkFont(size=14), **kwargs)
        self.set_state(False)

    def set_state(self, active: bool):
        self.configure(text_color=SUCCESS if active else DANGER)


class CardFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color=BG_CARD, corner_radius=12, **kwargs)


# ---------------------------------------------------------------------------
# Registration Dialog
# ---------------------------------------------------------------------------

class RegistrationDialog(ctk.CTkToplevel):
    """
    Modal dialog for capturing face samples and registering a new person.
    """

    def __init__(self, master, camera: cam_module.Camera, on_complete=None):
        super().__init__(master)
        self.camera      = camera
        self.on_complete = on_complete
        self._samples: list[np.ndarray] = []   # collected encodings
        self._collecting = False
        self._cancelled  = False
        self._target     = settings.get("num_samples", 40)

        self.title("Register New Face")
        self.geometry("560x640")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)
        self.grab_set()
        self.focus_set()

        self._build_ui()
        self._update_preview()

    def _build_ui(self):
        # Title
        ctk.CTkLabel(
            self, text="👤  Register New Face",
            font=ctk.CTkFont(family=FONT_FAMILY, size=20, weight="bold"),
            text_color=ACCENT,
        ).pack(pady=(20, 4))

        ctk.CTkLabel(
            self, text="Enter the person's name, then click 'Start Capture'.\nLook straight at the camera and slowly turn your head.",
            font=ctk.CTkFont(family=FONT_FAMILY, size=12),
            text_color=TEXT_SEC,
            justify="center",
        ).pack(pady=(0, 12))

        # Name entry
        name_frame = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=10)
        name_frame.pack(fill="x", padx=24, pady=4)
        ctk.CTkLabel(name_frame, text="Full Name:", text_color=TEXT_SEC,
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13)).pack(side="left", padx=12, pady=10)
        self.name_var = tk.StringVar()
        self.name_entry = ctk.CTkEntry(
            name_frame, textvariable=self.name_var, width=280,
            placeholder_text="e.g. Sreerag Kumar",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
        )
        self.name_entry.pack(side="left", padx=8, pady=10)

        # Preview canvas
        self.preview_label = ctk.CTkLabel(self, text="", fg_color=BG_PANEL, corner_radius=10)
        self.preview_label.pack(padx=24, pady=8, fill="both", expand=True)

        # Progress
        prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        prog_frame.pack(fill="x", padx=24, pady=4)
        self.progress_label = ctk.CTkLabel(
            prog_frame, text=f"Samples: 0 / {self._target}",
            text_color=TEXT_PRI, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        )
        self.progress_label.pack(anchor="w")
        self.progress_bar = ctk.CTkProgressBar(prog_frame, height=14, corner_radius=7)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", pady=4)

        # Status
        self.status_label = ctk.CTkLabel(
            self, text="⏳  Enter a name and click 'Start Capture'",
            text_color=TEXT_SEC, font=ctk.CTkFont(family=FONT_FAMILY, size=12),
        )
        self.status_label.pack(pady=4)

        # Buttons
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=24, pady=(8, 20))

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶  Start Capture", command=self._start_capture,
            fg_color=ACCENT_DARK, hover_color=ACCENT,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
            height=40, corner_radius=8,
        )
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="✖  Cancel", command=self._cancel,
            fg_color="#3a1c1c", hover_color=DANGER, text_color=TEXT_PRI,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
            height=40, corner_radius=8,
        ).pack(side="left", fill="x", expand=True, padx=(6, 0))

    # ---- Preview ----

    def _update_preview(self):
        if self._cancelled:
            return
        frame = self.camera.get_frame()
        if frame is not None:
            # Detect + draw on preview
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_detector.detect_faces(frame)
            for (top, right, bottom, left) in locations:
                color = (0, 212, 255) if not self._collecting else (80, 200, 80)
                cv2.rectangle(rgb, (left, top), (right, bottom), color, 2)

            pil_img = Image.fromarray(rgb)
            pil_img.thumbnail((512, 300))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            self.preview_label.configure(image=ctk_img, text="")
            self.preview_label._image = ctk_img  # prevent GC

            # Auto-capture
            if self._collecting and len(self._samples) < self._target and locations:
                self._try_capture(frame, rgb, locations)

        if not self._cancelled:
            self.after(80, self._update_preview)

    def _try_capture(self, bgr_frame, rgb_frame, locations):
        """Try to capture one encoding sample from the current frame."""
        if not locations:
            return
        top, right, bottom, left = locations[0]
        face_crop = face_detector.crop_face(bgr_frame, top, right, bottom, left)
        ok, reason = utils.check_image_quality(face_crop)
        if not ok:
            self.status_label.configure(text=f"⚠  {reason}", text_color=WARNING)
            return

        encodings = fre.encode_face(rgb_frame, locations=[locations[0]])
        if not encodings:
            return

        self._samples.append(encodings[0])
        count = len(self._samples)
        self.progress_bar.set(count / self._target)
        self.progress_label.configure(text=f"Samples: {count} / {self._target}")
        self.status_label.configure(
            text=f"✅  Capturing... {count}/{self._target}  — move your head slightly",
            text_color=SUCCESS,
        )

        if count >= self._target:
            self._finish_registration()

    # ---- Actions ----

    def _start_capture(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Name Required", "Please enter a name before capturing.", parent=self)
            return
        if not self.camera.is_running():
            messagebox.showerror("Camera Off", "Please start the camera first.", parent=self)
            return

        # Duplicate check using current frame encoding
        frame = self.camera.get_frame()
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_detector.detect_faces(frame)
            if locs:
                encs = fre.encode_face(rgb, locations=[locs[0]])
                if encs:
                    is_dup, dup_name = fre.is_duplicate(encs[0])
                    if is_dup:
                        messagebox.showwarning(
                            "Duplicate Detected",
                            f"This face appears to already be registered as '{dup_name}'.\n"
                            "Please ensure you are not re-registering an existing person.",
                            parent=self,
                        )
                        return

        self._collecting = True
        self._samples.clear()
        self.start_btn.configure(state="disabled")
        self.name_entry.configure(state="disabled")
        self.status_label.configure(
            text="📸  Capturing... Look straight, then slowly turn your head.",
            text_color=ACCENT,
        )

    def _finish_registration(self):
        self._collecting = False
        name = self.name_var.get().strip()
        self.status_label.configure(text="💾  Saving to database...", text_color=ACCENT)
        self.update()

        try:
            person_id = db.register_person(name, self._samples)
            fre._cache.add(person_id, name, self._samples)
            self.status_label.configure(
                text=f"🎉  '{name}' registered successfully! ({len(self._samples)} samples)",
                text_color=SUCCESS,
            )
            utils.speak(f"Registration complete. Welcome, {name}!")
            self.after(2000, lambda: self._close(person_id, name))
        except Exception as e:
            logger.exception("Registration failed.")
            messagebox.showerror("Error", f"Registration failed:\n{e}", parent=self)
            self.start_btn.configure(state="normal")
            self.name_entry.configure(state="normal")

    def _cancel(self):
        self._cancelled  = True
        self._collecting = False
        self.destroy()

    def _close(self, person_id, name):
        self._cancelled = True
        if self.on_complete:
            self.on_complete(person_id, name)
        self.destroy()


# ---------------------------------------------------------------------------
# Admin Password Dialog
# ---------------------------------------------------------------------------

class AdminGateDialog(ctk.CTkToplevel):
    def __init__(self, master, on_success=None):
        super().__init__(master)
        self.on_success = on_success
        self.title("Admin Verification")
        self.geometry("380x220")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)
        self.grab_set()
        self.focus_set()

        ctk.CTkLabel(
            self, text="🔒  Admin Password Required",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16, weight="bold"),
            text_color=WARNING,
        ).pack(pady=(24, 8))
        ctk.CTkLabel(
            self, text="Enter the admin password to continue.",
            text_color=TEXT_SEC, font=ctk.CTkFont(family=FONT_FAMILY, size=12),
        ).pack(pady=4)

        self.pw_var = tk.StringVar()
        self.pw_entry = ctk.CTkEntry(
            self, textvariable=self.pw_var, show="●", width=260,
            placeholder_text="Password",
            font=ctk.CTkFont(family=FONT_FAMILY, size=14),
        )
        self.pw_entry.pack(pady=12)
        self.pw_entry.bind("<Return>", lambda e: self._verify())

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=32, pady=8)
        ctk.CTkButton(btn_frame, text="✔  Confirm", command=self._verify,
                      fg_color=ACCENT_DARK, hover_color=ACCENT,
                      height=36, corner_radius=8,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=13)).pack(side="left", expand=True, padx=(0,4))
        ctk.CTkButton(btn_frame, text="✖  Cancel", command=self.destroy,
                      fg_color="#3a1c1c", hover_color=DANGER, text_color=TEXT_PRI,
                      height=36, corner_radius=8,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=13)).pack(side="left", expand=True, padx=(4,0))

        self.pw_entry.focus_set()

    def _verify(self):
        pw = self.pw_var.get()
        if settings.verify_admin_password(pw):
            self.destroy()
            if self.on_success:
                self.on_success()
        else:
            self.pw_var.set("")
            messagebox.showerror("Wrong Password", "Incorrect admin password.", parent=self)


# ---------------------------------------------------------------------------
# Persons Table View
# ---------------------------------------------------------------------------

class UsersView(CardFrame):
    def __init__(self, master, camera: cam_module.Camera, **kwargs):
        super().__init__(master, **kwargs)
        self.camera = camera
        self._build()

    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(hdr, text="👥  Registered Users",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
                     text_color=TEXT_PRI).pack(side="left")

        right_hdr = ctk.CTkFrame(hdr, fg_color="transparent")
        right_hdr.pack(side="right")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh())
        ctk.CTkEntry(right_hdr, textvariable=self.search_var, width=200,
                     placeholder_text="🔍 Search name...",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=13)).pack(side="left", padx=(0, 8))

        ctk.CTkButton(right_hdr, text="↻  Refresh", command=self._refresh,
                      width=90, height=32, corner_radius=8,
                      fg_color=ACCENT_DARK, hover_color=ACCENT,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left")

        # Scrollable table
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._col_headers()
        self._refresh()

    def _col_headers(self):
        headers = ["ID", "Name", "Registered", "Count", "Last Seen", "Actions"]
        widths  = [40, 180, 150, 60, 160, 180]
        for col, (h, w) in enumerate(zip(headers, widths)):
            ctk.CTkLabel(
                self.scroll_frame, text=h,
                font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                text_color=ACCENT, width=w, anchor="w",
            ).grid(row=0, column=col, padx=6, pady=4, sticky="w")

    def _refresh(self):
        # Clear rows (keep header row 0)
        for widget in self.scroll_frame.winfo_children():
            info = widget.grid_info()
            if info and int(info["row"]) > 0:
                widget.destroy()

        query = self.search_var.get().strip()
        persons = db.search_persons(query) if query else db.get_all_persons()

        for row_idx, p in enumerate(persons, start=1):
            bg = BG_PANEL if row_idx % 2 == 0 else "transparent"
            values = [
                str(p["id"]),
                p["name"],
                str(p["registration_date"])[:16],
                str(p["recognition_count"]),
                str(p["last_seen"] or "—")[:16],
            ]
            widths = [40, 180, 150, 60, 160]
            for col, (val, w) in enumerate(zip(values, widths)):
                ctk.CTkLabel(
                    self.scroll_frame, text=val,
                    font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                    text_color=TEXT_PRI, width=w, anchor="w",
                ).grid(row=row_idx, column=col, padx=6, pady=3, sticky="w")

            # Actions
            action_frame = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
            action_frame.grid(row=row_idx, column=5, padx=6, pady=3, sticky="w")
            pid = p["id"]
            name = p["name"]
            ctk.CTkButton(action_frame, text="✏ Rename", width=78, height=26,
                          corner_radius=6, fg_color="#1a3a5c", hover_color=ACCENT_DARK,
                          font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                          command=lambda p=pid, n=name: self._rename(p, n)).pack(side="left", padx=2)
            ctk.CTkButton(action_frame, text="🗑 Delete", width=72, height=26,
                          corner_radius=6, fg_color="#3a1c1c", hover_color=DANGER,
                          font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                          command=lambda p=pid, n=name: self._delete(p, n)).pack(side="left", padx=2)

    def _rename(self, person_id: int, current_name: str):
        new_name = simpledialog.askstring(
            "Rename", f"New name for '{current_name}':",
            parent=self.master,
        )
        if new_name and new_name.strip():
            db.rename_person(person_id, new_name.strip())
            fre._cache.rename(person_id, new_name.strip())
            self._refresh()

    def _delete(self, person_id: int, name: str):
        def do_delete():
            if messagebox.askyesno(
                "Confirm Delete",
                f"Permanently delete '{name}' and all their data?\nThis cannot be undone.",
                parent=self,
            ):
                db.delete_person(person_id)
                fre._cache.remove(person_id)
                self._refresh()
                messagebox.showinfo("Deleted", f"'{name}' has been removed.", parent=self)

        AdminGateDialog(self, on_success=do_delete)


# ---------------------------------------------------------------------------
# History View
# ---------------------------------------------------------------------------

class HistoryView(CardFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._build()

    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(hdr, text="📋  Recognition History",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
                     text_color=TEXT_PRI).pack(side="left")

        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right")
        ctk.CTkButton(right, text="📤 Export CSV", command=self._export,
                      width=110, height=32, corner_radius=8,
                      fg_color=ACCENT_DARK, hover_color=ACCENT,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left", padx=(0,8))
        ctk.CTkButton(right, text="↻ Refresh", command=self._refresh,
                      width=90, height=32, corner_radius=8,
                      fg_color=BG_PANEL, hover_color=BG_DARK,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left")

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._headers()
        self._refresh()

    def _headers(self):
        for col, (h, w) in enumerate(zip(
            ["#", "Name", "Timestamp", "Confidence"],
            [40, 200, 200, 100],
        )):
            ctk.CTkLabel(self.scroll, text=h,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=ACCENT, width=w, anchor="w").grid(
                row=0, column=col, padx=6, pady=4, sticky="w")

    def _refresh(self):
        for w in self.scroll.winfo_children():
            info = w.grid_info()
            if info and int(info["row"]) > 0:
                w.destroy()
        records = db.get_recognition_history(limit=300)
        for i, r in enumerate(records, start=1):
            conf_pct = f"{r['confidence']*100:.1f}%"
            conf_color = SUCCESS if r["confidence"] > 0.8 else ACCENT if r["confidence"] > 0.6 else WARNING
            for col, (val, w, color) in enumerate(zip(
                [str(i), r["name"], r["timestamp"], conf_pct],
                [40, 200, 200, 100],
                [TEXT_SEC, TEXT_PRI, TEXT_SEC, conf_color],
            )):
                ctk.CTkLabel(self.scroll, text=val, width=w, anchor="w",
                             text_color=color,
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12)).grid(
                    row=i, column=col, padx=6, pady=2, sticky="w")

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"history_{date.today()}.csv",
        )
        if path:
            data = db.get_recognition_history(limit=10000)
            if utils.export_to_csv(data, path):
                messagebox.showinfo("Exported", f"History exported to:\n{path}")
            else:
                messagebox.showerror("Export Failed", "Could not write the CSV file.")


# ---------------------------------------------------------------------------
# Attendance View
# ---------------------------------------------------------------------------

class AttendanceView(CardFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._build()

    def _build(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(16, 8))
        ctk.CTkLabel(hdr, text="📅  Attendance Log",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
                     text_color=TEXT_PRI).pack(side="left")

        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right")

        self.date_filter = ctk.CTkEntry(right, width=130, placeholder_text="YYYY-MM-DD",
                                        font=ctk.CTkFont(family=FONT_FAMILY, size=13))
        self.date_filter.pack(side="left", padx=(0, 8))
        self.date_filter.insert(0, date.today().isoformat())

        ctk.CTkButton(right, text="🔍 Filter", command=self._refresh,
                      width=80, height=32, corner_radius=8,
                      fg_color=ACCENT_DARK, hover_color=ACCENT,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left", padx=(0, 6))
        ctk.CTkButton(right, text="📤 Export", command=self._export,
                      width=80, height=32, corner_radius=8,
                      fg_color=BG_PANEL, hover_color=BG_DARK,
                      font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left")

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        self._headers()
        self._refresh()

    def _headers(self):
        for col, (h, w) in enumerate(zip(
            ["#", "Name", "Date", "Entry Time"],
            [40, 220, 120, 180],
        )):
            ctk.CTkLabel(self.scroll, text=h,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12, weight="bold"),
                         text_color=ACCENT, width=w, anchor="w").grid(
                row=0, column=col, padx=6, pady=4, sticky="w")

    def _refresh(self):
        for w in self.scroll.winfo_children():
            info = w.grid_info()
            if info and int(info["row"]) > 0:
                w.destroy()
        filter_date = self.date_filter.get().strip() or None
        records = attendance.get_records_by_date(filter_date) if filter_date else attendance.get_all_records()
        for i, r in enumerate(records, start=1):
            for col, (val, w) in enumerate(zip(
                [str(i), r["name"], r["date"], r["entry_time"]],
                [40, 220, 120, 180],
            )):
                ctk.CTkLabel(self.scroll, text=val, width=w, anchor="w",
                             text_color=TEXT_PRI,
                             font=ctk.CTkFont(family=FONT_FAMILY, size=12)).grid(
                    row=i, column=col, padx=6, pady=2, sticky="w")

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"attendance_{date.today()}.csv",
        )
        if path:
            if attendance.export_attendance_csv(path):
                messagebox.showinfo("Exported", f"Attendance exported to:\n{path}")
            else:
                messagebox.showerror("Export Failed", "Could not write the CSV file.")


# ---------------------------------------------------------------------------
# Settings View
# ---------------------------------------------------------------------------

class SettingsView(CardFrame):
    def __init__(self, master, camera: cam_module.Camera, **kwargs):
        super().__init__(master, **kwargs)
        self.camera = camera
        self._build()

    def _build(self):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=16, pady=16)

        ctk.CTkLabel(scroll, text="⚙️  Settings",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=18, weight="bold"),
                     text_color=TEXT_PRI).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 16))

        def row(r, label, widget_factory):
            ctk.CTkLabel(scroll, text=label, text_color=TEXT_SEC,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13),
                         anchor="w").grid(row=r, column=0, sticky="w", padx=8, pady=6)
            w = widget_factory(scroll)
            w.grid(row=r, column=1, sticky="w", padx=8, pady=6)
            return w

        # Camera index
        def cam_widget(p):
            cameras = cam_module.list_cameras()
            opts = [str(i) for i in cameras] if cameras else ["0"]
            v = ctk.StringVar(value=str(settings.get("camera_index", 0)))
            w = ctk.CTkOptionMenu(p, values=opts, variable=v, width=120,
                                  command=lambda val: settings.set("camera_index", int(val)))
            return w
        row(1, "Camera Index:", cam_widget)

        # Resolution
        def res_widget(p):
            opts = ["640x480", "1280x720", "1920x1080"]
            cur = f"{settings.get('resolution_width',1280)}x{settings.get('resolution_height',720)}"
            if cur not in opts:
                cur = opts[1]
            v = ctk.StringVar(value=cur)
            def on_res(val):
                w, h = val.split("x")
                self.camera.set_resolution(int(w), int(h))
            return ctk.CTkOptionMenu(p, values=opts, variable=v, width=160, command=on_res)
        row(2, "Resolution:", res_widget)

        # Recognition model
        def model_widget(p):
            v = ctk.StringVar(value=settings.get("recognition_model", "hog"))
            return ctk.CTkOptionMenu(p, values=["hog", "cnn"], variable=v, width=120,
                                     command=lambda val: settings.set("recognition_model", val))
        row(3, "Recognition Model:", model_widget)

        # Confidence threshold
        self._conf_label = None
        def conf_widget(p):
            frm = ctk.CTkFrame(p, fg_color="transparent")
            val = settings.get("confidence_threshold", 0.50)
            lbl = ctk.CTkLabel(frm, text=f"{val:.2f}", text_color=ACCENT, width=42,
                               font=ctk.CTkFont(family=FONT_FAMILY, size=13))
            lbl.pack(side="right", padx=4)
            def on_slide(v):
                v2 = round(float(v), 2)
                settings.set("confidence_threshold", v2)
                lbl.configure(text=f"{v2:.2f}")
            s = ctk.CTkSlider(frm, from_=0.3, to=0.8, number_of_steps=50,
                              width=200, command=on_slide)
            s.set(val)
            s.pack(side="left")
            return frm
        row(4, "Confidence Threshold:", conf_widget)

        # Samples count
        def samples_widget(p):
            frm = ctk.CTkFrame(p, fg_color="transparent")
            val = settings.get("num_samples", 40)
            lbl = ctk.CTkLabel(frm, text=str(int(val)), text_color=ACCENT, width=42,
                               font=ctk.CTkFont(family=FONT_FAMILY, size=13))
            lbl.pack(side="right", padx=4)
            def on_slide(v):
                settings.set("num_samples", int(v))
                lbl.configure(text=str(int(v)))
            s = ctk.CTkSlider(frm, from_=10, to=80, number_of_steps=70,
                              width=200, command=on_slide)
            s.set(val)
            s.pack(side="left")
            return frm
        row(5, "Registration Samples:", samples_widget)

        # Debounce seconds
        def debounce_widget(p):
            frm = ctk.CTkFrame(p, fg_color="transparent")
            val = settings.get("debounce_seconds", 2.5)
            lbl = ctk.CTkLabel(frm, text=f"{val:.1f}s", text_color=ACCENT, width=50,
                               font=ctk.CTkFont(family=FONT_FAMILY, size=13))
            lbl.pack(side="right", padx=4)
            def on_slide(v):
                settings.set("debounce_seconds", round(float(v), 1))
                lbl.configure(text=f"{round(float(v),1):.1f}s")
            s = ctk.CTkSlider(frm, from_=0.5, to=10.0, number_of_steps=95,
                              width=200, command=on_slide)
            s.set(val)
            s.pack(side="left")
            return frm
        row(6, "Recognition Debounce:", debounce_widget)

        # Toggles
        def toggle_widget(key, default=True):
            def make(p):
                v = ctk.BooleanVar(value=settings.get(key, default))
                sw = ctk.CTkSwitch(p, text="", variable=v,
                                   command=lambda: settings.set(key, v.get()),
                                   onvalue=True, offvalue=False)
                return sw
            return make
        row(7, "Voice Greeting:",   toggle_widget("voice_enabled"))
        row(8, "Attendance Mode:",  toggle_widget("attendance_mode"))
        row(9, "Show FPS Counter:", toggle_widget("fps_display"))

        # Change admin password
        def change_pw_widget(p):
            return ctk.CTkButton(
                p, text="🔑  Change Admin Password", width=220, height=32, corner_radius=8,
                fg_color="#3a2a00", hover_color=WARNING,
                font=ctk.CTkFont(family=FONT_FAMILY, size=12),
                command=self._change_password,
            )
        row(10, "Admin Password:", change_pw_widget)

        # Save button
        ctk.CTkButton(
            scroll, text="💾  Save & Apply", command=lambda: messagebox.showinfo("Saved", "Settings saved."),
            fg_color=ACCENT_DARK, hover_color=ACCENT,
            height=40, corner_radius=8,
            font=ctk.CTkFont(family=FONT_FAMILY, size=14, weight="bold"),
        ).grid(row=11, column=0, columnspan=2, pady=(20, 8), sticky="ew", padx=8)

    def _change_password(self):
        def do_change():
            new_pw = simpledialog.askstring(
                "New Password", "Enter new admin password:", show="*", parent=self
            )
            if new_pw and len(new_pw) >= 4:
                settings.change_admin_password(new_pw)
                messagebox.showinfo("Updated", "Admin password changed successfully.")
            elif new_pw is not None:
                messagebox.showerror("Too Short", "Password must be at least 4 characters.")
        AdminGateDialog(self, on_success=do_change)


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class MainApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("FaceID — AI Facial Recognition System")
        self.geometry("1300x800")
        self.minsize(1100, 700)
        self.configure(fg_color=BG_DARK)

        # Core objects
        self._camera     = cam_module.Camera()
        self._fps        = utils.FPSCounter(window=20)
        self._rec_queue: queue.Queue = queue.Queue(maxsize=4)
        self._rec_thread: Optional[threading.Thread] = None
        self._rec_running = False
        self._last_results: list = []  # [(top,right,bottom,left, name, conf, is_unknown)]
        self._active_view  = "live"

        # Build UI
        self._build_layout()
        self._switch_view("live")

        # Keyboard shortcut
        self.bind("<Escape>", lambda e: self._on_exit())
        self.protocol("WM_DELETE_WINDOW", self._on_exit)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        # ---- Root grid ----
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Sidebar ----
        self._sidebar = ctk.CTkFrame(self, width=220, fg_color=BG_CARD, corner_radius=0)
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.grid_propagate(False)
        self._build_sidebar()

        # ---- Content area ----
        self._content = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        self._content.grid(row=0, column=1, sticky="nsew", padx=0)
        self._content.grid_columnconfigure(0, weight=1)
        self._content.grid_rowconfigure(0, weight=1)

        # ---- Status bar ----
        self._status_bar = ctk.CTkFrame(self, height=28, fg_color=BG_CARD, corner_radius=0)
        self._status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._status_lbl = ctk.CTkLabel(
            self._status_bar, text="Ready",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_SEC, anchor="w",
        )
        self._status_lbl.pack(side="left", padx=12)
        self._stats_lbl = ctk.CTkLabel(
            self._status_bar, text="",
            font=ctk.CTkFont(family=FONT_FAMILY, size=11),
            text_color=TEXT_SEC, anchor="e",
        )
        self._stats_lbl.pack(side="right", padx=12)
        self._update_stats_bar()

    def _build_sidebar(self):
        sb = self._sidebar

        # Logo
        logo = ctk.CTkLabel(
            sb,
            text="FaceID",
            font=ctk.CTkFont(family=FONT_FAMILY, size=24, weight="bold"),
            text_color=ACCENT,
        )
        logo.pack(pady=(24, 2), padx=16, anchor="w")
        ctk.CTkLabel(sb, text="AI Recognition System",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                     text_color=TEXT_SEC).pack(padx=16, anchor="w")

        ctk.CTkFrame(sb, height=1, fg_color=BG_PANEL).pack(fill="x", padx=12, pady=16)

        # Camera status
        cam_row = ctk.CTkFrame(sb, fg_color="transparent")
        cam_row.pack(fill="x", padx=16, pady=(0, 8))
        ctk.CTkLabel(cam_row, text="Camera", text_color=TEXT_SEC,
                     font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left")
        self._cam_dot = StatusDot(cam_row)
        self._cam_dot.pack(side="right")

        # Navigation buttons
        self._nav_btns: dict[str, SidebarButton] = {}
        nav_items = [
            ("live",       "📷",  "Live Feed"),
            ("users",      "👥",  "Registered Users"),
            ("history",    "📋",  "History"),
            ("attendance", "📅",  "Attendance"),
            ("settings",   "⚙️",  "Settings"),
        ]
        for key, icon, label in nav_items:
            btn = SidebarButton(sb, label, icon=icon,
                                command=lambda k=key: self._switch_view(k))
            btn.pack(fill="x", padx=8, pady=2)
            self._nav_btns[key] = btn

        ctk.CTkFrame(sb, height=1, fg_color=BG_PANEL).pack(fill="x", padx=12, pady=16)

        # Camera controls
        ctk.CTkLabel(sb, text="CAMERA CONTROLS",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=10),
                     text_color=TEXT_SEC).pack(anchor="w", padx=16, pady=(0, 4))

        self._start_btn = ctk.CTkButton(
            sb, text="▶  Start Camera", command=self._start_camera,
            fg_color="#0d3a1a", hover_color=SUCCESS, text_color=TEXT_PRI,
            height=38, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        )
        self._start_btn.pack(fill="x", padx=8, pady=3)

        self._stop_btn = ctk.CTkButton(
            sb, text="⏹  Stop Camera", command=self._stop_camera,
            fg_color="#3a1c1c", hover_color=DANGER, text_color=TEXT_PRI,
            height=38, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
            state="disabled",
        )
        self._stop_btn.pack(fill="x", padx=8, pady=3)

        ctk.CTkButton(
            sb, text="👤  Register Face", command=self._open_registration,
            fg_color=ACCENT_DARK, hover_color=ACCENT, text_color="white",
            height=38, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        ).pack(fill="x", padx=8, pady=3)

        ctk.CTkFrame(sb, height=1, fg_color=BG_PANEL).pack(fill="x", padx=12, pady=16)

        # Exit
        ctk.CTkButton(
            sb, text="❌  Exit", command=self._on_exit,
            fg_color="transparent", hover_color="#3a1c1c", text_color=DANGER,
            height=36, corner_radius=8, font=ctk.CTkFont(family=FONT_FAMILY, size=13),
        ).pack(fill="x", padx=8, pady=2, side="bottom")

    # ------------------------------------------------------------------
    # View switching
    # ------------------------------------------------------------------

    def _switch_view(self, key: str):
        # Clear content
        for w in self._content.winfo_children():
            w.destroy()

        # Update nav highlight
        for k, btn in self._nav_btns.items():
            btn.set_active(k == key)
        self._active_view = key

        if key == "live":
            self._build_live_view()
        elif key == "users":
            UsersView(self._content, self._camera).grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        elif key == "history":
            HistoryView(self._content).grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        elif key == "attendance":
            AttendanceView(self._content).grid(row=0, column=0, sticky="nsew", padx=16, pady=16)
        elif key == "settings":
            SettingsView(self._content, self._camera).grid(row=0, column=0, sticky="nsew", padx=16, pady=16)

    # ------------------------------------------------------------------
    # Live view
    # ------------------------------------------------------------------

    def _build_live_view(self):
        live = ctk.CTkFrame(self._content, fg_color="transparent")
        live.grid(row=0, column=0, sticky="nsew")
        live.grid_columnconfigure(0, weight=3)
        live.grid_columnconfigure(1, weight=1)
        live.grid_rowconfigure(0, weight=1)

        # Video canvas
        self._video_label = ctk.CTkLabel(live, text="", fg_color=BG_CARD, corner_radius=12)
        self._video_label.grid(row=0, column=0, sticky="nsew", padx=(16,8), pady=16)

        # Right panel
        right = ctk.CTkFrame(live, fg_color=BG_CARD, corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 16), pady=16)
        right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(right, text="Recognition Status",
                     font=ctk.CTkFont(family=FONT_FAMILY, size=15, weight="bold"),
                     text_color=TEXT_PRI).pack(padx=16, pady=(16, 8), anchor="w")

        self._recog_scroll = ctk.CTkScrollableFrame(right, fg_color="transparent", height=300)
        self._recog_scroll.pack(fill="x", padx=8, pady=4)

        ctk.CTkFrame(right, height=1, fg_color=BG_PANEL).pack(fill="x", padx=12, pady=8)

        # Stats
        self._stats_panel = ctk.CTkFrame(right, fg_color="transparent")
        self._stats_panel.pack(fill="x", padx=16, pady=4)
        self._stat_labels: dict[str, ctk.CTkLabel] = {}
        for key, display in [("faces_detected", "Faces in frame"),
                              ("today_attendance", "Attendance today"),
                              ("total_registered", "Total registered")]:
            row_f = ctk.CTkFrame(self._stats_panel, fg_color="transparent")
            row_f.pack(fill="x", pady=3)
            ctk.CTkLabel(row_f, text=display, text_color=TEXT_SEC,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(side="left")
            lbl = ctk.CTkLabel(row_f, text="0", text_color=ACCENT,
                               font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"))
            lbl.pack(side="right")
            self._stat_labels[key] = lbl

        # Camera status hint
        self._cam_hint = ctk.CTkLabel(
            self._video_label,
            text="📷  Camera is off\n\nClick 'Start Camera' to begin",
            font=ctk.CTkFont(family=FONT_FAMILY, size=16),
            text_color=TEXT_SEC,
        )
        self._cam_hint.place(relx=0.5, rely=0.5, anchor="center")

        # Start the frame update loop
        self._schedule_frame_update()

    def _schedule_frame_update(self):
        """Periodic GUI update (runs on main thread via after())."""
        if self._active_view != "live":
            return

        frame = self._camera.get_frame()
        if frame is not None:
            self._cam_hint.place_forget()
            self._fps.tick()

            # Draw results on frame
            annotated = frame.copy()
            for (top, right, bottom, left, name, conf, is_unknown) in self._last_results:
                utils.draw_face_box(annotated, top, right, bottom, left,
                                    name, conf, unknown=is_unknown)
            if settings.get("fps_display", True):
                utils.draw_fps(annotated, self._fps.fps)

            # Convert to CTkImage
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Scale to fit label
            lw = self._video_label.winfo_width()
            lh = self._video_label.winfo_height()
            if lw > 10 and lh > 10:
                pil.thumbnail((lw, lh), Image.LANCZOS)

            ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
            self._video_label.configure(image=ctk_img, text="")
            self._video_label._image = ctk_img
        else:
            # Show hint if camera not started
            if not self._camera.is_running():
                try:
                    self._cam_hint.place(relx=0.5, rely=0.5, anchor="center")
                except Exception:
                    pass

        self.after(33, self._schedule_frame_update)  # ~30 FPS GUI update

    def _update_recognition_panel(self, results: list):
        """Update the right-panel status cards."""
        if not hasattr(self, "_recog_scroll"):
            return
        for w in self._recog_scroll.winfo_children():
            w.destroy()

        known = [(t, r, b, l, n, c, u) for t, r, b, l, n, c, u in results if not u]
        unknowns = [x for x in results if x[6]]

        for (top, right, bottom, left, name, conf, _) in known:
            card = ctk.CTkFrame(self._recog_scroll, fg_color=BG_PANEL, corner_radius=8)
            card.pack(fill="x", pady=4)
            ctk.CTkLabel(card, text=f"✅  {name}",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=SUCCESS).pack(anchor="w", padx=12, pady=(8, 2))
            ctk.CTkLabel(card, text=f"Confidence: {conf*100:.1f}%",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=11),
                         text_color=TEXT_SEC).pack(anchor="w", padx=12, pady=(0, 8))

        if unknowns:
            card = ctk.CTkFrame(self._recog_scroll, fg_color="#2a1a1a", corner_radius=8)
            card.pack(fill="x", pady=4)
            ctk.CTkLabel(card, text=f"❓  {len(unknowns)} Unknown face(s)",
                         font=ctk.CTkFont(family=FONT_FAMILY, size=13, weight="bold"),
                         text_color=WARNING).pack(anchor="w", padx=12, pady=(8, 2))
            ctk.CTkButton(card, text="Register Now", command=self._open_registration,
                          height=28, corner_radius=6, width=120,
                          fg_color=ACCENT_DARK, hover_color=ACCENT,
                          font=ctk.CTkFont(family=FONT_FAMILY, size=11)).pack(anchor="w", padx=12, pady=(0, 8))

        if not results:
            ctk.CTkLabel(self._recog_scroll, text="No faces detected",
                         text_color=TEXT_SEC,
                         font=ctk.CTkFont(family=FONT_FAMILY, size=12)).pack(pady=20)

        # Update stat numbers
        stats = db.get_stats()
        if "faces_detected" in self._stat_labels:
            self._stat_labels["faces_detected"].configure(text=str(len(results)))
            self._stat_labels["today_attendance"].configure(text=str(stats["today_attendance"]))
            self._stat_labels["total_registered"].configure(text=str(stats["total_persons"]))

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------

    def _start_camera(self):
        ok = self._camera.start()
        if ok:
            self._cam_dot.set_state(True)
            self._start_btn.configure(state="disabled")
            self._stop_btn.configure(state="normal")
            self._set_status("Camera started  ●  Recognition active")
            self._start_recognition_thread()
        else:
            messagebox.showerror(
                "Camera Error",
                f"Could not open camera {settings.get('camera_index', 0)}.\n"
                "Check that your webcam is connected and not in use by another app.",
            )

    def _stop_camera(self):
        self._stop_recognition_thread()
        self._camera.stop()
        self._cam_dot.set_state(False)
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._last_results = []
        self._set_status("Camera stopped")
        if hasattr(self, "_video_label"):
            self._video_label.configure(image=None, text="")

    # ------------------------------------------------------------------
    # Recognition thread
    # ------------------------------------------------------------------

    def _start_recognition_thread(self):
        if self._rec_running:
            return
        self._rec_running = True
        self._rec_thread = threading.Thread(
            target=self._recognition_loop, daemon=True, name="RecognitionThread"
        )
        self._rec_thread.start()
        self._poll_recognition_queue()

    def _stop_recognition_thread(self):
        self._rec_running = False

    def _recognition_loop(self):
        """
        Runs in a background thread.
        Reads frames from camera, runs detection + recognition,
        posts results back to GUI via queue.
        """
        fre.load_known_faces()
        import face_recognition_engine as fre_mod

        while self._rec_running:
            frame = self._camera.get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # Ensure proper uint8 3-channel frame (some webcam drivers return BGRA)
            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = face_detector.detect_faces(frame, scale_factor=0.4)

            results = []
            for loc in locations:
                top, right, bottom, left = loc
                encs = fre.encode_face(rgb, locations=[loc])
                if not encs:
                    results.append((top, right, bottom, left, "Unknown", 0.0, True))
                    continue

                enc = encs[0]
                person_id, name, conf = fre.identify_face(enc)
                is_unknown = person_id is None

                results.append((top, right, bottom, left, name, conf, is_unknown))

                # Debounce check
                if fre_mod.debouncer.should_fire(person_id, loc):
                    if not is_unknown:
                        db.update_recognition_stats(person_id, conf)
                        if settings.get("attendance_mode", True):
                            attendance.record_attendance(person_id, name)
                        if fre_mod.debouncer.should_greet(person_id):
                            utils.speak(f"Welcome back, {name}!")

            # Post to queue (non-blocking, drop if full)
            try:
                self._rec_queue.put_nowait(results)
            except queue.Full:
                pass

            time.sleep(0.05)  # ~20 recognition cycles/sec

    def _poll_recognition_queue(self):
        """Drain the recognition queue and update the GUI. Runs on main thread."""
        try:
            results = self._rec_queue.get_nowait()
            self._last_results = results
            self._update_recognition_panel(results)
        except queue.Empty:
            pass

        if self._rec_running:
            self.after(100, self._poll_recognition_queue)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _open_registration(self):
        if not self._camera.is_running():
            messagebox.showinfo(
                "Camera Required",
                "Please start the camera before registering a face."
            )
            return

        def on_complete(person_id, name):
            self._set_status(f"✅  Registered: {name}")
            self._update_stats_bar()

        RegistrationDialog(self, self._camera, on_complete=on_complete)

    # ------------------------------------------------------------------
    # Status / Stats
    # ------------------------------------------------------------------

    def _set_status(self, text: str):
        if hasattr(self, "_status_lbl"):
            self._status_lbl.configure(text=text)

    def _update_stats_bar(self):
        try:
            stats = db.get_stats()
            self._stats_lbl.configure(
                text=f"👥 {stats['total_persons']} registered  |  "
                     f"📅 {stats['today_attendance']} today  |  "
                     f"🔍 {stats['total_recognitions']} total recognitions"
            )
        except Exception:
            pass
        self.after(10000, self._update_stats_bar)

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def _on_exit(self):
        if messagebox.askyesno("Exit", "Are you sure you want to exit FaceID?"):
            self._stop_camera()
            self.destroy()


# ---------------------------------------------------------------------------
# Entry point for GUI module (called from main.py)
# ---------------------------------------------------------------------------

def run():
    """Initialise and run the application."""
    app = MainApp()
    app.mainloop()
