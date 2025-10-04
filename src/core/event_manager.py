class Event:
    def __init__(self, name, unit_id=None, action_type=None, target=None, affliated_id=None, data=None, source=None):
        # 统一事件字段，提供最小兼容层
        self.name = name
        self.type = name  # 兼容旧代码使用 event.type
        self.unit_id = unit_id
        self.action_type = action_type
        self.affliated_id = affliated_id
        self.target = target
        self.source = source
        self.data = data
        self.terminated = False


class EventManager:
    def __init__(self):
        self.listeners = {}

    def subscribe(self, event_type, listener):
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(listener)

    def post(self, event):
        for listener in self.listeners.get(event.type, []):
            listener.handle_event(event)


class EventHandler:
    def handle_event(self, event):
        pass
