from datalink.ipc import SPMCQueue, MPMCQueue, AddrType

class Messaging:
    def __init__(self):
        self.q_sim = MPMCQueue(name="q_state", addr_type=AddrType.TCP, ports=(11001, 11002))
        # self.q_real = MPMCQueue(name="real", addr_type=AddrType.TCP, ports=(11001, 11002), q_size=10)
        self.q_real = SPMCQueue(name="real", type=AddrType.TCP, port=10002)
        self.q_sim_processing = SPMCQueue(name="sim_processing", type=AddrType.TCP, port=10030)
        self.q_real_processing = SPMCQueue(name="real_processing", type=AddrType.TCP, port=10040)
        self.q_control = SPMCQueue(name="q_control", type=AddrType.TCP, port=10050)
        self.q_ui = SPMCQueue(name="ui", type=AddrType.TCP, port=10060)

messaging = Messaging()
