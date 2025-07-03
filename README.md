# ToySim Proxy + UI

I want to explore the world of simulation, physical tinkering, data visualization,
computer vision, control theory, and others, combined in compact package for whoever wishes to have fun with it.

The idea here is to make it simple to implement, debug and fine-tune different
methods of control in both simulated and physical environment, using either real-time
data or recorded play-back to test and iterate upon.

I want to keep this as simple as possible. I make the custom tools to try things out for myself, my way, for comparison. But otherwise there are solutions
for:
 - framework:
   - https://docs.ros.org/en/foxy/index.html
 - data capture:
   - https://mcap.dev/spec
 - visualization:
   - https://app.foxglove.dev
   - https://wiki.ros.org/rviz

## Proxy

Server CMD app to which simulation or physical vehicle connect. It receives state data and can send control inputs based on implemented vehicle controller.

- Server to send/recv to/from simulation/car  
- (Remote) Controller:
  - DualShock (PS5)
  - Self-Driving - various methods of position estimation, detection, path planning, and path tracking

## UI
Qt desktop app for live/playback data visualization and parameter control.

- Visualization of simulation/car data
- Tuning of settings and algorithm parameters
- Data recording/playback

## TODO

- Time deltas
  - [ ] Simulation/Car components
  - [ ] Client to Proxy
  - [ ] Proxy to UI (render)
- [ ] Unify data format using topics
- [ ] Unify data capture using MCAP:
- Recording:
  - [ ] Currently frames based on camera time - reworks using simulation time
  - [ ] Make playback bar closable
- [ ] Choose source (simulation, car, playback)
- [ ] Choose controller type and update controller panel

## Attributions/Sources

- pyqtgraph dev version
  - https://pyqtgraph.readthedocs.io/en/latest/getting_started/installation.html
- superqt: https://pyapp-kit.github.io/superqt/
  - (pip install) fonticon-materialdesignicons7

- Alternative simulation engines:
  - https://github.com/o3de/o3de
  - https://developer.nvidia.com/isaac
  - https://github.com/projectchrono/chrono