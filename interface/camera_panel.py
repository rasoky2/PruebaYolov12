"""
Widget para manejo de cámaras y configuración - PyQt6 Version
"""

import cv2
import json
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QGroupBox, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt


class CameraPanel(QWidget):
    camera_changed = pyqtSignal(int, dict)

    def __init__(self, parent=None, camera_info_callback=None):
        super().__init__(parent)
        self.camera_info_callback = camera_info_callback
        
        self.camera_config = None
        self.available_cameras = []
        self.camera_info = {}
        self.camera_id = 0
        
        self.load_camera_config()
        self.detect_cameras()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        group = QGroupBox("Cámara")
        v_layout = QVBoxLayout()
        
        if not self.available_cameras:
            lbl_error = QLabel("No hay cámaras disponibles")
            lbl_error.setStyleSheet("color: #ef4444;")
            v_layout.addWidget(lbl_error)
            group.setLayout(v_layout)
            layout.addWidget(group)
            return

        # Selection
        sel_layout = QHBoxLayout()
        lbl_cam = QLabel("Dispositivo:")
        
        self.combo_cam = QComboBox()
        self.combo_cam.setStyleSheet("color: #000000; background-color: #ffffff;")
        self.update_camera_options()
        # Connect signal AFTER populating to avoid premature triggering or loops
        self.combo_cam.currentIndexChanged.connect(self.change_camera)
        
        sel_layout.addWidget(lbl_cam)
        sel_layout.addWidget(self.combo_cam)
        v_layout.addLayout(sel_layout)
        
        # Info Display
        self.info_widget = QWidget()
        self.info_layout = QVBoxLayout(self.info_widget)
        self.info_layout.setContentsMargins(0, 5, 0, 5)
        self.update_camera_info_display()
        v_layout.addWidget(self.info_widget)

        
        group.setLayout(v_layout)
        layout.addWidget(group)
        layout.addStretch()

    def update_camera_options(self):
        self.combo_cam.clear()
        if not self.available_cameras:
             self.combo_cam.addItem("No cameras found", -1)
             return

        for cam_id in self.available_cameras:
            cam_info = self.camera_info[cam_id]
            fav_marker = " ⭐" if cam_info["is_favorite"] else ""
            self.combo_cam.addItem(f"{cam_id}: {cam_info['name']}{fav_marker}", cam_id)
            
        # Set current
        index = self.combo_cam.findData(self.camera_id)
        if index >= 0:
            self.combo_cam.setCurrentIndex(index)
        else:
            if self.combo_cam.count() > 0:
                self.combo_cam.setCurrentIndex(0)

    def update_camera_info_display(self):
        # Clear previous info
        for i in reversed(range(self.info_layout.count())): 
            self.info_layout.itemAt(i).widget().setParent(None)
            
        if self.camera_id in self.camera_info:
            info = self.camera_info[self.camera_id]
            
            self._add_info_row("Nombre:", info["name"])
            self._add_info_row("Resolución:", info["resolution"])
            self._add_info_row("FPS:", f"{info['fps']:.1f}")
            
            if info["is_favorite"]:
                lbl_fav = QLabel("⭐ FAVORITA")
                lbl_fav.setStyleSheet("color: #eab308; font-weight: bold;")
                self.info_layout.addWidget(lbl_fav)

    def _add_info_row(self, label, value):
        row = QHBoxLayout()
        lbl_l = QLabel(label)
        lbl_l.setStyleSheet("color: #71717a;")
        lbl_v = QLabel(str(value))
        lbl_v.setStyleSheet("font-weight: 500;")
        row.addWidget(lbl_l)
        row.addStretch()
        row.addWidget(lbl_v)
        container = QWidget()
        container.setLayout(row)
        self.info_layout.addWidget(container)

    def change_camera(self, idx=None):
        if idx is None:
            idx = self.combo_cam.currentIndex()
        if idx < 0: return
        
        new_id = self.combo_cam.itemData(idx)
        if new_id is not None and new_id in self.available_cameras:
            # Avoid re-triggering if same camera
            if new_id == self.camera_id:
                return

            self.camera_id = new_id
            self.update_camera_info_display()
            
            # Removed intrussive messagebox
            # QMessageBox.information(self, "Cámara cambiada", f"Cámara cambiada a: {self.camera_info[self.camera_id]['name']}")
            
            if self.camera_info_callback:
                self.camera_info_callback(self.camera_id, self.camera_info[self.camera_id])
            self.camera_changed.emit(self.camera_id, self.camera_info[self.camera_id])
        else:
            # Only warn if it's a real user action failure, but here it's automatic mostly
            pass

    # Reuse detection logic from previous version mostly
    def load_camera_config(self):
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "camera_config.json")
            with open(config_path, encoding="utf-8") as f:
                self.camera_config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.camera_config = None

    # These helpers need to handle imports carefully or be copied
    def get_camera_name(self, cam_id: int) -> str:
        # Simplified internal logic to avoid circular import dependancy if main.py not ready
        if self.camera_config and "cameras" in self.camera_config and str(cam_id) in self.camera_config["cameras"]:
             return self.camera_config["cameras"][str(cam_id)]["name"]
        return f"Dispositivo {cam_id}"

    def get_camera_description(self, cam_id: int) -> str:
        if self.camera_config and "cameras" in self.camera_config and str(cam_id) in self.camera_config["cameras"]:
             return self.camera_config["cameras"][str(cam_id)].get("description", "")
        return ""

    def is_favorite_camera(self, cam_id: int) -> bool:
        if self.camera_config and "cameras" in self.camera_config and str(cam_id) in self.camera_config["cameras"]:
             return self.camera_config["cameras"][str(cam_id)].get("is_favorite", False)
        return False

    def detect_cameras(self):
        self.available_cameras = []
        self.camera_info = {}

        # Scan a range of indices
        for i in range(3): 
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    # Check if we can strictly read a frame (optional but safer)
                    ret, _ = cap.read()
                    if ret:
                         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                         fps = cap.get(cv2.CAP_PROP_FPS)
                         
                         self.camera_info[i] = {
                             "name": self.get_camera_name(i),
                             "description": self.get_camera_description(i),
                             "resolution": f"{width}x{height}",
                             "fps": fps,
                             "width": width,
                             "height": height,
                             "is_favorite": self.is_favorite_camera(i),
                         }
                         self.available_cameras.append(i)
                    cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")

        if self.available_cameras:
            favorite_cam = None
            for cam_id in self.available_cameras:
                if self.camera_info[cam_id]["is_favorite"]:
                    favorite_cam = cam_id
                    break
            self.camera_id = favorite_cam if favorite_cam is not None else self.available_cameras[0]
        else:
             # Fallback manual add if 0 is detected by system but failed read (common in virtual cams)
             pass

    def get_camera_id(self):
        return self.camera_id

    def get_camera_info(self):
        return self.camera_info.get(self.camera_id, {})

