from datalink.ipc import SPMCQueue, SPMCQueueType

# TODO: make into a singleton

class Messaging:
    def __init__(self):
        self.q_sim = SPMCQueue(name="simulation", type=SPMCQueueType.TCP, port=10001)
        self.q_real = SPMCQueue(name="real", type=SPMCQueueType.TCP, port=10002)
        self.q_processing = SPMCQueue(name="processing", type=SPMCQueueType.TCP, port=10003)
        self.q_control = SPMCQueue(name="control", type=SPMCQueueType.TCP, port=10004)
        self.q_ui = SPMCQueue(name="ui", type=SPMCQueueType.TCP, port=10005)
        
        
messaging = Messaging()
