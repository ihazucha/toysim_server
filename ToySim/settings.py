import struct

class ClientTypes:
    SIMULATION = 0
    VEHICLE = 1

class SimulationCameraSettings:
    WIDTH = 640
    HEIGHT = 480
    PIXEL_COUNT = WIDTH * HEIGHT
    
class TypeSizesBytes:
    UINT64 = struct.calcsize("Q") 
    FLOAT = struct.calcsize("f")
    DOUBLE = struct.calcsize("d")
    
class SimulationDataSizesBytes:
    """Sizes of data that is sent from the simulation to the server."""
    class Camera:
        RGB_PIXEL = 3
        DEPTH_PIXEL = 2
        RGB_FRAME = SimulationCameraSettings.PIXEL_COUNT * RGB_PIXEL
        DEPTH_FRAME = SimulationCameraSettings.PIXEL_COUNT * DEPTH_PIXEL
        TOTAL_FRAME = RGB_FRAME + DEPTH_FRAME
        PRE_RENDER_UNIX_TIMESTAMP = TypeSizesBytes.UINT64
        POST_RENDER_UNIX_TIMESTAMP = TypeSizesBytes.UINT64
        UNIX_TIMESTAMPS = PRE_RENDER_UNIX_TIMESTAMP + POST_RENDER_UNIX_TIMESTAMP
        GAME_FRAME_NUMBER = TypeSizesBytes.UINT64
        RENDER_FRAME_NUMBER = TypeSizesBytes.UINT64
        FRAME_NUMBERS = GAME_FRAME_NUMBER + RENDER_FRAME_NUMBER
        TOTAL = RGB_FRAME + DEPTH_FRAME + UNIX_TIMESTAMPS + FRAME_NUMBERS
        
    class Vehicle: 
        POSITION = 3 * TypeSizesBytes.DOUBLE
        ROTATION = 3 * TypeSizesBytes.DOUBLE
        POSE = POSITION + ROTATION
        SPEED = TypeSizesBytes.FLOAT
        STEERING_ANGLE = TypeSizesBytes.FLOAT
        TOTAL = POSE + SPEED + STEERING_ANGLE

    class Scene:
        DELTA_TIME = TypeSizesBytes.FLOAT
        
    TOTAL = Camera.TOTAL + Vehicle.TOTAL + Scene.DELTA_TIME


class VehicleCamera:
    WIDTH = 820
    HEIGHT = 616
    PIXEL_COUNT = WIDTH * HEIGHT

class VehicleDataSizes:
    class Camera:
        RGB_PIXEL = 3
        DEPTH_PIXEL = 2
        RGB_FRAME = VehicleCamera.PIXEL_COUNT * RGB_PIXEL 
        DEPTH_FRAME = VehicleCamera.PIXEL_COUNT * DEPTH_PIXEL 
        TOTAL_FRAME = RGB_FRAME + DEPTH_FRAME
        # ---
        TOTAL = TOTAL_FRAME
    
    class Vehicle:
        MOTORS_POWER = TypeSizesBytes.FLOAT
        SPEED = TypeSizesBytes.FLOAT
        STEERING_ANGLE = TypeSizesBytes.FLOAT
        DELTA_TIME = TypeSizesBytes.FLOAT
        IMU_DATA = 6 * TypeSizesBytes.FLOAT
        ENCODER_DATA = 2 * TypeSizesBytes.FLOAT
        POSE = 6 * TypeSizesBytes.FLOAT
        # ---
        TOTAL = MOTORS_POWER + SPEED + STEERING_ANGLE + DELTA_TIME + IMU_DATA + ENCODER_DATA + POSE

    TOTAL = Camera.TOTAL + Vehicle.TOTAL


class ControllerDataSizesBytes:
    """Sizes of data that is sent from the server to the simulation."""
    class Vehicle:
        SPEED = TypeSizesBytes.FLOAT
        STEERING_ANGLE = TypeSizesBytes.FLOAT
        TOTAL = SPEED + STEERING_ANGLE
    
    TOTAL = Vehicle.TOTAL
    
    
class NetworkSettings:
    class Simulation:
        SERVER_HOST = "localhost"
        SERVER_PORT = 3333
        SERVER_ADDR = (SERVER_HOST, SERVER_PORT)
        RECV_DATA_SIZE_BYTES = SimulationDataSizesBytes.TOTAL
        SEND_DATA_SIZE_BYTES = SimulationDataSizesBytes.Vehicle.TOTAL
    class Vehicle:
        SERVER_HOST = "localhost"
        SERVER_PORT = 3333
        SERVER_ADDR = (SERVER_HOST, SERVER_PORT)
        RECV_DATA_SIZE_BYTES = VehicleDataSizes.TOTAL
        SEND_DATA_SIZE_BYTES = SimulationDataSizesBytes.Vehicle.TOTAL

class ControlLoopSettings:
    CONTROL_FPS = 60
    CONTROL_DTIME = 1 / CONTROL_FPS
    
class RenderLoopSettings:
    RENDER_FPS = 60
    RENDER_DTIME = 1 / RENDER_FPS

# ATM for testing purposes
GENERATOR_FPS = 60
GENERATOR_DTIME = 60 / GENERATOR_FPS
