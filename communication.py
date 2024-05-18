import socket
import threading


class Communication:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host, self.port))
        self.lock = threading.Lock()

    def send(self, data, address):
        with self.lock:
            self.socket.sendto(data.encode(), address)

    def receive(self, buffer_size=1024):
        with self.lock:
            data, addr = self.socket.recvfrom(buffer_size)
            return data.decode(), addr

    def close(self):
        self.socket.close()
