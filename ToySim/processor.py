from multiprocessing import Queue, Process, Event
from controller import DualSense



class Processor(Process):
    def __init__(
        self,
        q_image: Queue,
        q_sensor: Queue,
        q_control: Queue,
        q_render: Queue,
    ):
        super().__init__()
        self._q_image = q_image
        self._q_sensor = q_sensor
        self._q_control = q_control
        self._q_render = q_render


    def run(self):
        controller = DualSense(self._q_control)
        while True:
            # Control queue
            control_data_frame = None
            if controller.isAlive():
                control_data_frame = controller.update()
            else:
                control_data_frame = ControlDataFrame(0.0, 0.0)
            self._q_control.put(control_data_frame.tobytes(), block=False)

            image = self._q_image.get()

        
            # Render queue
            self._q_render.put((data_frame, control_data_frame))
