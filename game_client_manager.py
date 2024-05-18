import threading


class NetworkClient:
    def __init__(self, server_host, server_port):
        self.communication = Communication('0.0.0.0', 0)  # 客户端绑定到任意可用端口
        self.server_address = (server_host, server_port)
        self.action = None

    def start(self):
        threading.Thread(target=self.receive_data).start()
        while True:
            if self.action is not None:
                self.communication.send(self.action, self.server_address)
                self.action = None

    def receive_data(self):
        while True:
            data, _ = self.communication.receive()
            if data:
                print(f"Received from server: {data}")

    def send_action(self, action):
        self.action = action


class GameClientManager:
    def __init__(self, server_host, server_port):
        self.network_client = NetworkClient(server_host, server_port)
        self.env = Env(name="SC2Env", player_config=player_config)
        self.human_player = HumanPlayer(name="HumanPlayer")

    def start(self):
        threading.Thread(target=self.network_client.start).start()
        self.run_game()

    def run_game(self):
        state = self.env.reset_game(config)
        done = False
        while not done:
            action = self.human_player.choose_action(state)
            self.network_client.send_action(action)
            data, _ = self.network_client.communication.receive()
            if data:
                observations, rewards = data.split(",")
                done = self.env.update(
                    {self.network_client.server_address: action})
            if done:
                state = self.env.reset_game(config)

    def get_human_action(self):
        # Implement method to get human player action
        return 'move_up'  # Example action


if __name__ == "__main__":
    game_client_manager = GameClientManager(
        server_host='127.0.0.1', server_port=9999)
    game_client_manager.start()
