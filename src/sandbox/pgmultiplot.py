# import itertools

# import numpy as np
# from utils import FrameCounter


# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtWidgets


# app = pg.mkQApp()

# class IMURawRemotePlotWidget(pg.RemoteGraphicsView):
#     def __init__(self):
#         super().__init__()
#         self.pg.setConfigOptions(antialias=True)
        
#         self.data = self._proc.transfer(np.array(2500))
        
#         self._plt = self.pg.PlotItem()
#         self._plt.setYRange(50, -50, padding=0)
#         self._plt.getAxis("left").setPen(dict(color='y', width=1, style=QtCore.Qt.PenStyle.DashLine))
#         self._plt._setProxyOptions(deferGetattr=True)  ## speeds up access to rplt.plot
#         self.curve = self._plt.plot()  
#         self.setCentralItem(self._plt)
#         self._plt._setProxyOptions(callSync="off")
#     def update_data(self, data):
#         if data is None:
#             return None
#         x = np.array(2500)
#         x[:] = data
#         self.curve.setData(self.data, clear=True, _callSync='off')  
#         framecnt.update()

# view = IMURawRemotePlotWidget()

# app.aboutToQuit.connect(view.close)

# layout = pg.LayoutWidget()
# label = QtWidgets.QLabel()
# layout.addWidget(label)
# layout.addWidget(view, row=1, col=0, colspan=3)
# layout.resize(800,800)
# layout.show()


# iterations_counter = itertools.count()

# timer = QtCore.QTimer()
# timer.timeout.connect(lambda: view.update_data(get_data()))
# timer.start(0)

# def get_data():
#     if next(iterations_counter) > 500:
#         timer.stop()
#         app.quit()
#         return None

#     data = np.random.normal(size=(10000,50)).sum(axis=1)
#     data += 5 * np.sin(np.linspace(0, 10, data.shape[0]))
#     return data


# framecnt = FrameCounter()
# framecnt.sigFpsUpdate.connect(lambda fps : label.setText(f"Generating {fps:.1f}"))


# if __name__ == '__main__':
#     pg.exec()


import argparse
import itertools

import numpy as np
from utils import FrameCounter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', default=float('inf'), type=float)
args = parser.parse_args()

iterations_counter = itertools.count()

app = pg.mkQApp()

view = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()
view_setData = pg.widgets.RemoteGraphicsView.RemoteGraphicsView()

pg.setConfigOptions(antialias=True)

view.pg.setConfigOptions(antialias=True)
view_setData.pg.setConfigOptions(antialias=True)
view.setWindowTitle('pyqtgraph example: RemoteSpeedTest')
view.setWindowTitle('pyqtgraph example: RemoteSpeedTest (setData)')

app.aboutToQuit.connect(view.close)

label = QtWidgets.QLabel()
rcheck = QtWidgets.QCheckBox('plot remote (plot)')
rcheck_setData = QtWidgets.QCheckBox('plot remote (setData)')
rcheck.setChecked(True)
lcheck = QtWidgets.QCheckBox('plot local')
lplt = pg.PlotWidget()
layout = pg.LayoutWidget()

layout.addWidget(rcheck)
layout.addWidget(lcheck)
layout.addWidget(rcheck_setData)
layout.addWidget(label)
layout.addWidget(view, row=1, col=0, colspan=3)
layout.addWidget(lplt, row=2, col=0, colspan=3)
layout.addWidget(view_setData, row=3, col=0, colspan=3)
layout.resize(800,800)
layout.show()

rplt = view.pg.PlotItem()
rplt._setProxyOptions(deferGetattr=True)
view.setCentralItem(rplt)

rplt_setData = view_setData.pg.PlotItem()
rplt_setData._setProxyOptions(deferGetattr=True)
rplt_setData_dataItem = rplt_setData.plot()
view_setData.setCentralItem(rplt_setData)

def update():
    if next(iterations_counter) > args.iterations:
        timer.stop()
        app.quit()
        return None

    data = np.random.normal(size=(10000,50)).sum(axis=1)
    data += 5 * np.sin(np.linspace(0, 10, data.shape[0]))
    
    if rcheck.isChecked():
        rplt.plot(data, clear=True, _callSync="off")
    if lcheck.isChecked():
        lplt.plot(data, clear=True)
    if rcheck_setData.isChecked():
        processed_data = np.ascontiguousarray(data)  # Ensure data is C-contiguous
        rplt_setData_dataItem.setData(processed_data, _callSync="off")

    framecnt.update()
        
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

framecnt = FrameCounter()
framecnt.sigFpsUpdate.connect(lambda fps : label.setText(f"Generating {fps:.1f}"))

if __name__ == '__main__':
    pg.exec()