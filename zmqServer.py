import zmq

def server():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect("tcp://localhost:5560")

    while True:
        # 从代理接收消息
        message = socket.recv_json()
        print("Received request: ", message)

        # 假设响应
        response = {'status': 'ok', 'data': 'Processed'}
        socket.send_json(response)

if __name__ == '__main__':
    server()
