import zmq
import json

def zmq_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ (REQUEST) socket for sending requests
    socket.connect("tcp://localhost:5559")  # Connect to the server

    while(True):
        # Send a "start" command to the server
        command = input()
        command = {'command': command}
        socket.send_json(command)
        print("Sent command: ", command)

        # Wait for the reply from the server
        game_state = socket.recv_json()
        print("Received reply: ", game_state)

if __name__ == '__main__':
    zmq_client()
