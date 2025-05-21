# ToySim UI

Desktop application for the ToySim suite contains:

- Server to send/recv to/from simulation/car  
- (Remote) Controller:
  - DualShock (PS5)
  - Keyboard  
  - Software self-driving
    - choice from multiple methods of detection, path planning, and path tracking
- UI for:
  - visualization of simulation/car data
  - finetuning of settings and algorithm parameters
  - recorder/playback of simulation data

The general idea is to make it easy to implement, debug and fine-tune different
methods of controlling the car in both simulated and physical environment, using either real-time
data or recorded play-back to test and iterate upon.

The intention was to explore the world of simulation, physical tinkering, data visualization,
computer vision, control theory, and others, combined in compact package for whoever wishes to have fun.

---

## TODO: BUGS

- [ ] When TcpConnection_BP has tick time set to 0.5s - the connection will be made but no data will be passed for some reason

## TODO

 
- [ ] System Panel
  - FPS/dt of the simulation/car (components)
  - FPS/dt of the UI
  - dt between UI and Client
    - [ ] current value and history (plot)
- [x] THE SIMULATION DEPTH DATA IS INCORRECT - EACH PIXEL SHOWS DISTANCE FROM THE PIXEL, NOT FROM CAMERA CENTER
  - [ ] Make this to be done inside of the simulation, rather than in the processor
- [ ] Standardize simulation and physical car communication channel
- [ ] Record-playback:
  - Sidebar with recordings:
    - simulation datetime
    - editable optional description
    - Client ID (sim/alamak)
    - first image (if video-feed present)
    - "Open folder" button
    - recordings folder path
    - double-click to play, or play button
    - shows which recording is selected (playing)
  - playback bar
    - pause/play
    - forward/back
    - reset (from)
    - timeline time ((mili)seconds)
    - timeline frames
    - ability to navigate in timeline fast
- menu of available controllers
  - controller panel for each controller

### Attributions/Sources
Icons: - put on git/in the thesis
Uicons by <a href="https://www.flaticon.com/uicons">Flaticon</a>

- pyqtgraph has to be the latest install (from source)
  - https://pyqtgraph.readthedocs.io/en/latest/getting_started/installation.html
- superqt
  - https://pyapp-kit.github.io/superqt/
- pip install fonticon-materialdesignicons7

- Describe engines:
  - https://github.com/o3de/o3de
  - NVIDIA ISAAC
  - https://github.com/projectchrono/chrono