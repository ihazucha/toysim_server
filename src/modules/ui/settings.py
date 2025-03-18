import json
from pathlib import Path
from PySide6.QtCore import QRect
from PySide6.QtGui import QGuiApplication


class WindowSettings:
    """Persistent UI window state"""

    def __init__(self, window, settings_path=None):
        self.window = window
        self.settings_path = (
            settings_path or Path(__file__).parent / "settings/window_settings.json"
        )

    def load(self):
        if not self.settings_path.exists():
            return False

        try:
            with open(self.settings_path, "r") as f:
                pos = json.load(f)

            # Verify window position is on screen
            screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
            if not screen_geometry.contains(
                QRect(
                    pos.get("x", 0),
                    pos.get("y", 0),
                    pos.get("width", self.window.width()),
                    pos.get("height", self.window.height()),
                )
            ):
                pos["x"], pos["y"] = 100, 100

            self.window.move(pos.get("x", 100), pos.get("y", 100))

            # Apply size if specified
            if "width" in pos and "height" in pos:
                self.window.resize(pos["width"], pos["height"])

            # Apply window state
            if pos.get("isMaximized", False):
                self.window.showMaximized()

            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading window settings: {e}")
            return False

    def save(self):
        """Save window position, size and state."""
        try:
            # Create settings directory if it doesn't exist
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)

            # Build settings data
            pos = {
                "x": self.window.x(),
                "y": self.window.y(),
                "width": self.window.width(),
                "height": self.window.height(),
                "isMaximized": self.window.isMaximized(),
            }

            # Write to file
            with open(self.settings_path, "w") as f:
                json.dump(pos, f, indent=4)

            return True
        except IOError as e:
            print(f"Error saving window settings: {e}")
            return False
