import threading
import time
from communication import Communication


class NetworkServer:
    def __init__(self, host='0.0.0.0', port=9999, max_clients=2):
        self.communication = Communication(host, port)
        self.clients = {}
        self.actions = {}
        self.max_clients = max_clients

    def start(self):
        print("Server started, waiting for connections...")
        threading.Thread(target=self.receive_loop).start()
        threading.Thread(target=self.broadcast_loop).start()

    def receive_loop(self):
        while True:
            data, addr = self.communication.receive()
            if addr not in self.clients and len(self.clients) < self.max_clients:
                self.clients[addr] = threading.Thread(
                    target=self.handle_client, args=(addr,))
                self.clients[addr].start()
            self.actions[addr] = data

    def handle_client(self, addr):
        print(f"New connection from {addr}")
        while True:
            if addr in self.actions:
                action = self.actions[addr]
                # Placeholder for any client-specific handling

    def broadcast_loop(self):
        while True:
            if len(self.actions) >= self.max_clients:
                collected_actions = self.collect_actions()
                for client_addr in self.clients:
                    self.communication.send(collected_actions, client_addr)
                self.actions.clear()
            time.sleep(0.05)  # Adjust this to match the desired frame rate

    def collect_actions(self):
        # Convert actions to a suitable format for sending
        return str(self.actions)
