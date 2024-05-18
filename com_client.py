import threading
from communication import Communication


class CommunicationClient:
    def __init__(self, server_host, server_port):
        self.communication = Communication('0.0.0.0', 0)  # 客户端绑定到任意可用端口
        self.server_address = (server_host, server_port)
        self.action = None
        self.received_actions = None

    def start(self):
        threading.Thread(target=self.receive_data).start()
        while True:
            if self.action is not None:
                self.communication.send(self.action, self.server_address)
                self.action = None

    def receive_data(self):
        while True:
            data, _ = self.communication.receive()
            if data:
                self.received_actions = data

    def send_action(self, action):
        self.action = action
