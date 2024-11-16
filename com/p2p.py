import socket
import threading
import queue
import pickle
import time


class Component:
    def __init__(self):
        pass


class Entity:
    def __init__(self):
        self.components = []

    def add_component(self, component):
        self.components.append(component)


class System:
    def __init__(self):
        pass

    def update(self, entities, delta_time):
        raise NotImplementedError("Subclasses should implement this!")


class EventManager:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_name, callback):
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)

    def trigger_event(self, event_name, *args, **kwargs):
        if event_name in self.subscribers:
            for callback in self.subscribers[event_name]:
                callback(*args, **kwargs)


class HealthComponent(Component):
    def __init__(self, initial_health=100):
        super().__init__()
        self.health = initial_health

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            # 触发死亡事件
            EventManager.get_instance().trigger_event('PlayerDeath')


class PlayerComponent(Component):
    def __init__(self):
        super().__init__()
        self.health = None


class UISystem(System):
    def __init__(self):
        super().__init__()
        self.subscribe_to_events()

    def subscribe_to_events(self):
        EventManager.get_instance().subscribe('PlayerDeath', self.on_player_death)

    def on_player_death(self):
        print("Player has died!")


class EnemyComponent(Component):
    def __init__(self):
        super().__init__()
        self.health = None


class DamageOverTimeComponent(Component):
    def __init__(self, damage_per_tick, duration, tick_interval):
        self.damage_per_tick = damage_per_tick  # 每个时间间隔的伤害量
        self.duration = duration  # 持续伤害总时长
        self.tick_interval = tick_interval  # 伤害触发间隔
        self.elapsed_time = 0  # 已经过的时间
        self.time_since_last_tick = 0  # 自上次伤害触发以来过去的时间


class DamageOverTimeSystem(System):
    def update(self, delta_time):
        for entity in self.entities:
            dot = entity.get_component(DamageOverTimeComponent)
            if dot:
                dot.elapsed_time += delta_time
                dot.time_since_last_tick += delta_time

                # 检查是否达到触发伤害的时间间隔
                if dot.time_since_last_tick >= dot.tick_interval:
                    self.apply_damage(entity, dot.damage_per_tick)
                    dot.time_since_last_tick = 0

                # 检查持续伤害是否结束
                if dot.elapsed_time >= dot.duration:
                    entity.remove_component(DamageOverTimeComponent)

    def apply_damage(self, entity, damage):
        health = entity.get_component(HealthComponent)
        if health:
            health.current_health -= damage
            print(
                f"Entity {entity.id} took {damage} damage, remaining health {health.current_health}")


class Peer:
    def __init__(self, host, port, peers):
        self.host = host
        self.port = port
        self.peers = peers
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.message_queue = queue.Queue()
        self.running = True

        # 开始接收线程
        receive_thread = threading.Thread(target=self.receive)
        receive_thread.start()

    def receive(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(1024)
                message = pickle.loads(data)
                self.message_queue.put((message, addr))
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def send(self, data, peer):
        try:
            self.sock.sendto(pickle.dumps(data), peer)
        except Exception as e:
            print(f"Error sending data: {e}")

    def close(self):
        self.running = False
        self.sock.close()


class State:
    def enter(self, entity):
        pass

    def execute(self, entity, delta_time):
        pass

    def exit(self, entity):
        pass


class IdleState(State):
    def enter(self, entity):
        print(f"{entity.id} entered Idle state.")

    def execute(self, entity, delta_time):
        pass

    def exit(self, entity):
        print(f"{entity.id} exiting Idle state.")


class MovingState(State):
    def enter(self, entity):
        print(f"{entity.id} entered Moving state.")

    def execute(self, entity, delta_time):
        # 实现移动逻辑
        pass

    def exit(self, entity):
        print(f"{entity.id} exiting Moving state.")


class EntityStateMachine:
    def __init__(self, entity):
        self.entity = entity
        self.current_state = None

    def change_state(self, new_state):
        if self.current_state:
            self.current_state.exit(self.entity)
        self.current_state = new_state
        self.current_state.enter(self.entity)

    def update(self, delta_time):
        if self.current_state:
            self.current_state.execute(self.entity, delta_time)


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

        # 发送状态到其他客户端
        if self.network:
            data = {
                'id': self.id,
                'position': self.component.position.position,
                'state': type(self.state_machine.current_state).__name__
            }
            for peer in self.network.peers:
                self.network.send(data, peer)

    def handle_message(self, message):
        # 处理来自其他客户端的状态更新
        if message['id'] != self.id:
            self.render_buffer.update_buffer(
                message['id'], {'position': message['position']})


class RenderBuffer:
    def __init__(self):
        self.buffer = {}
        self.lock = threading.Lock()

    def update_buffer(self, entity_id, data):
        with self.lock:
            self.buffer[entity_id] = data

    def clear_buffer(self):
        with self.lock:
            self.buffer.clear()

    def get_buffer(self):
        with self.lock:
            return self.buffer.copy()


class CombatSystem(System):
    def update(self, entities, delta_time):
        for entity in entities:
            player = entity.get_component(PlayerComponent)
            enemy = entity.get_component(EnemyComponent)
            if player and enemy:
                # 检查状态
                if isinstance(entity.state_machine.current_state, IdleState):
                    # 如果处于Idle状态，则开始移动
                    entity.set_state(MovingState())
                elif isinstance(entity.state_machine.current_state, MovingState):
                    # 如果处于Moving状态，则进行攻击
                    player.health.take_damage(10)


class RendererThread(threading.Thread):
    def __init__(self, render_buffer):
        super().__init__()
        self.render_buffer = render_buffer
        self.running = True

    def run(self):
        while self.running:
            buffer = self.render_buffer.get_buffer()
            # 渲染逻辑
            print("Rendering:", buffer)
            time.sleep(0.016)  # 模拟渲染时间

    def stop(self):
        self.running = False


class EventManagerSingleton(EventManager):
    _instance = None

    @staticmethod
    def get_instance():
        if EventManagerSingleton._instance is None:
            EventManagerSingleton._instance = EventManager()
        return EventManagerSingleton._instance


def main():
    global render_buffer
    render_buffer = RenderBuffer()

    # 创建网络连接
    peers = [("127.0.0.1", 5001), ("127.0.0.1", 5002)]  # 其他客户端的地址
    network = Peer("127.0.0.1", 5000, peers)  # 创建本地客户端

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

        # 处理网络消息
        while not network.message_queue.empty():
            message, addr = network.message_queue.get()
            player.handle_message(message)

        break  # 为了演示，这里使用break退出循环

    # 停止渲染线程
    renderer_thread.stop()
    renderer_thread.join()

    # 关闭网络连接
    network.close()


if __name__ == "__main__":
    main()
