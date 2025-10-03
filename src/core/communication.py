import socket
import threading
import json
import time


class Communication:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.lock = threading.Lock()

    def send(self, data, address):
        with self.lock:
            serialized_data = json.dumps(data)
            self.socket.sendto(serialized_data.encode(), address)

    def receive(self, buffer_size=1024):
        with self.lock:
            data, addr = self.socket.recvfrom(buffer_size)
            deserialized_data = json.loads(data.decode())
            return deserialized_data, addr

    def close(self):
        self.socket.close()


class CommunicationServer():
    def __init__(self, host='0.0.0.0', port=9999, max_clients=2):
        self.communication = Communication(host, port)
        self.clients = {}
        self.actions = {}
        self.max_clients = max_clients
        self.running = True

    def start(self):
        print("Server started, waiting for connections...")
        threading.Thread(target=self.receive_loop, daemon=True).start()
        threading.Thread(target=self.broadcast_loop, daemon=True).start()

    def receive_loop(self):
        while self.running:
            try:
                data, addr = self.communication.receive()
            except OSError:
                break
            if addr not in self.clients and len(self.clients) < self.max_clients:
                self.clients[addr] = threading.Thread(
                    target=self.handle_client, args=(addr,), daemon=True)
                self.clients[addr].start()
            self.actions[addr] = data

    def handle_client(self, addr):
        print(f"New connection from {addr}")
        while self.running:
            if addr in self.actions:
                action = self.actions[addr]
                # Placeholder for any client-specific handling

    def broadcast_loop(self):
        while self.running:
            if len(self.actions) >= self.max_clients:
                collected_actions = self.collect_actions()
                for client_addr in self.clients:
                    self.communication.send(collected_actions, client_addr)
                self.actions.clear()
            time.sleep(0.05)  # Adjust this to match the desired frame rate

    def collect_actions(self):
        # Convert actions to a suitable format for sending
        return self.actions

    def stop(self):
        self.running = False
        try:
            self.communication.close()
        except Exception:
            pass


class CommunicationClient:
    def __init__(self, server_host, server_port):
        self.communication = Communication('0.0.0.0', 0)  # 客户端绑定到任意可用端口
        self.server_address = (server_host, server_port)
        self.action = None
        self.received_actions = {}
        self.running = True

    def start(self):
        threading.Thread(target=self.receive_data, daemon=True).start()
        while self.running:
            if self.action is not None:
                self.communication.send(self.action, self.server_address)
                self.action = None
            time.sleep(0.01)

    def receive_data(self):
        while self.running:
            try:
                data, _ = self.communication.receive()
            except OSError:
                break
            if data:
                self.received_actions.update(data)  # 合并接收到的动作数据

    def send_action(self, action):
        self.action = action

    def stop(self):
        self.running = False
        try:
            self.communication.close()
        except Exception:
            pass
