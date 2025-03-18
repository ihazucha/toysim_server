from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPixmap, QImage
import pyqtgraph as pg
import zmq
from importlib import reload
import sys

def main():
    while True:
        # Force reload of the render module
        if 'modules.render' in sys.modules:
            import modules.render
            reload(modules.render)
        
        from modules.render import Renderer

        renderer = Renderer()
        try:
            exit_code = renderer.run()
        finally:
            zmq.Context.instance().destroy()
            print("Exiting")