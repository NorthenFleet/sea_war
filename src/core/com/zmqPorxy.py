import zmq

def start_proxy():
    context = zmq.Context()

    # 前端，接收客户端请求
    frontend = context.socket(zmq.ROUTER)
    frontend.bind("tcp://*:5559")

    # 后端，连接到服务端
    backend = context.socket(zmq.DEALER)
    backend.bind("tcp://*:5560")

    # 使用ZMQ的内置代理功能
    zmq.proxy(frontend, backend)

    frontend.close()
    backend.close()
    context.term()

if __name__ == "__main__":
    start_proxy()
