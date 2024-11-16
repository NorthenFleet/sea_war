import socket
import threading
import pickle


class Server:
    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('127.0.0.1', 5555))
        self.server.listen(2)
        print("Waiting for a connection, Server Started")

        self.players = {}
        self.num_players = 0

    def threaded_client(self, conn, player):
        conn.send(str.encode(str(player)))
        reply = ""
        while True:
            try:
                data = pickle.loads(conn.recv(2048))
                reply = data.decode("utf-8")
                if not data:
                    print("Disconnected")
                    break
                else:
                    print("Received: ", data)
                    print("Sending : ", reply)

                conn.sendall(pickle.dumps(reply))
            except:
                break

        print("Lost connection")
        conn.close()

    def run(self):
        while True:
            conn, addr = self.server.accept()
            print("Connected to:", addr)

            self.num_players += 1
            p = 0
            thread = threading.Thread(
                target=self.threaded_client, args=(conn, p))
            thread.start()


if __name__ == "__main__":
    server = Server()
    server.run()
