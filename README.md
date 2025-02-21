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

## TODO

- [x] move client/server + data structures code into a separate git project, included in the existing ones
- [ ] Make renderer visualize all available data - especially FPS statistics
- [ ]standardize simulation and physical car communication channel
- [ ] THE SIMULATION DEPTH DATA IS INCORRECT - EACH PIXEL SHOWS DISTANCE FROM THE PIXEL, NOT FROM CAMERA CENTER
  (TCP vs. UDP with multiple channels for image, fast sensor data, ...)
- [ ] finish record-playback feature such that:
  - panel with recordings is available, showing:
    - simulation datetime
    - optional description
    - simulation vs. car
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
  - general info panel
    - dt and FPS of the simulation/car
    - dt and FPS of the UI
    - delay between the client and UI
      - current value and history (plot)
- menu of available controllers
  - controller panel for each controller