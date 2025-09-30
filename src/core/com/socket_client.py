import socket
import pickle


class Network:
    def __init__(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server = "127.0.0.1"  # 服务器IP地址
        self.port = 5555
        self.addr = (self.server, self.port)
        self.p = self.connect()

    def connect(self):
        try:
            self.client.connect(self.addr)
            return self.client.recv(2048).decode()
        except:
            pass

    def send(self, data):
        try:
            self.client.send(pickle.dumps(data))
            return pickle.loads(self.client.recv(2048))
        except socket.error as e:
            print(e)


class Entity:
    def __init__(self, id, network=None):
        self.id = id
        self.network = network
        self.components = []
        self.state_machine = EntityStateMachine(self)
        self.render_buffer = render_buffer

    def add_component(self, component):
        self.components.append(component)

    def set_state(self, state):
        self.state_machine.change_state(state)

    def update(self, delta_time):
        self.state_machine.update(delta_time)
        # 更新渲染缓冲区
        self.render_buffer.update_buffer(
            self.id, {'position': self.component.position.position})

        # 发送状态到服务器
        if self.network:
            self.network.send({
                'id': self.id,
                'position': self.component.position.position,
                'state': type(self.state_machine.current_state).__name__
            })

# 主程序实现多人游戏


def main():
    global render_buffer
    render_buffer = RenderBuffer()

    # 创建网络连接
    network = Network()

    # 创建实体
    player = Entity(1, network)
    player.add_component(PlayerComponent())
    player.component.health = HealthComponent(100)  # 设置初始生命值
    player.set_state(IdleState())

    # 创建系统
    combat_system = CombatSystem()
    ui_system = UISystem()

    # 创建渲染线程
    renderer_thread = RendererThread(render_buffer)
    renderer_thread.start()

    # 游戏循环
    while True:
        delta_time = 0.016  # 假设每帧时间间隔为0.016秒
        player.update(delta_time)
        combat_system.update([player], delta_time)
        # 在这里可以加入更多的逻辑
        time.sleep(0.016)  # 模拟游戏逻辑更新时间
        break  # 为了演示，这里使用break退出循环

    # 停止渲染线程
    renderer_thread.stop()
    renderer_thread.join()


if __name__ == "__main__":
    main()
