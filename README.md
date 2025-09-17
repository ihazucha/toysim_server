# ToySim - Proxy Server & GUI

Exploring simulation, tinkering, data visualization, computer vision, control theory and others, combined in compact package for whoever wishes enjoy.

The idea here is to make it simple to implement, debug and fine-tune different methods of control for a model vehicle in both simulation and physical world.

I want to keep this custom but simple. The obvious software to use for robotics appplications is:
for:
 - Framework:
   - https://docs.ros.org/en/foxy/index.html
 - Data Capture:
   - https://mcap.dev/spec
 - Visualization:
   - https://app.foxglove.dev
   - https://wiki.ros.org/rviz

## Proxy Server

CMD app to which simulation and physical vehicle connect. It exchanges data with the vehicle and implements the self-driving controller.

## GUI

Desktop app for live/playback data visualization and parameters control.

## TODO

Split Proxy and GUI to separate projects, reimplement UI in browser using websockets or use existing (Foxglove studio)

- Time deltas
  - [ ] Between Simulation/Car components
  - [ ] Client to Proxy
  - [ ] Proxy to UI (render)
- [ ] Unify data format using topics
- [ ] Unify data capture using MCAP:

## Attributions/Sources

- pyqtgraph dev version
  - https://pyqtgraph.readthedocs.io/en/latest/getting_started/installation.html
- superqt: https://pyapp-kit.github.io/superqt/
  - (pip install) fonticon-materialdesignicons7