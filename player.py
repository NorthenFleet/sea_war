class Player:
    def __init__(self, name, agents, player_type, communication):
        self.name = name
        self.agents = agents if isinstance(agents, list) else [agents]
        self.player_type = player_type
        self.communication = communication
        self.input_event_listener = None

    def set_input_event_listener(self, listener):
        if self.player_type == 'Human':
            self.input_event_listener = listener

    def handle_input_event(self, event):
        if self.player_type == 'AI':
            return
        if self.input_event_listener:
            self.input_event_listener(event)

    def receive_state_update(self, state):
        for agent in self.agents:
            agent.choose_action(state)
