class Event:
    def __init__(self, type, data):
        self.type = type
        self.data = data


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
