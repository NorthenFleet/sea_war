class GameLogicManager:
    def __init__(self, env, network_server):
        self.env = env
        self.network_server = network_server
        self.ais = {}

    def load_ais(self):
        self.ais['ai_1'] = DQNAgent(
            name="DQN_Agent_1", state_size=84*84, action_size=2)
        self.ais['ai_2'] = DQNAgent(
            name="DQN_Agent_2", state_size=84*84, action_size=2)

    def run_game(self):
        self.load_ais()
        while True:
            # Wait until all actions are collected
            while len(self.network_server.actions) < len(self.network_server.clients) + len(self.ais):
                time.sleep(0.01)

            action_dict = self.receive_actions()
            observations, rewards, done, _ = self.env.update(action_dict)
            self.send_observations(observations, rewards)
            if done:
                self.env.reset_game(config)

    def receive_actions(self):
        action_dict = self.network_server.collect_actions()
        for ai_name, ai in self.ais.items():
            action_dict[ai_name] = ai.choose_action(self.env.state)
        return action_dict

    def send_observations(self, observations, rewards):
        for addr in self.network_server.clients:
            message = f"{observations[addr]}, {rewards[addr]}"
            self.network_server.communication.send(message, addr)
        for ai_name in self.ais:
            message = f"{observations[ai_name]}, {rewards[ai_name]}"
            # Assuming AIs do not need to receive observations in this example
