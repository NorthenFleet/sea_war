import threading, time
from env import Env
from com_client import CommunicationClient
from player_human import HumanPlayer


class GameClientManager:
    def __init__(self, server_host, server_port):
        self.network_client = CommunicationClient(server_host, server_port)
        self.env = Env(name="SC2Env", player_config=player_config)
        self.human_player = HumanPlayer(name="HumanPlayer")

    def start(self):
        threading.Thread(target=self.network_client.start).start()
        self.run_game()

    def run_game(self):
        state = self.env.reset_game()
        done = False
        while not done:
            action = self.human_player.choose_action(state)
            self.network_client.send_action(action)
            while self.network_client.received_actions is None:
                # Wait for the server to broadcast the actions
                time.sleep(0.01)
            # Convert the received string back to a dictionary
            action_dict = eval(self.network_client.received_actions)
            self.network_client.received_actions = None
            observations, rewards, done, _ = self.env.update(action_dict)
            if done:
                state = self.env.reset_game()

    def get_human_action(self):
        # Implement method to get human player action
        return 'move_up'  # Example action


if __name__ == "__main__":
    game_client_manager = GameClientManager(
        server_host='127.0.0.1', server_port=9999)
    game_client_manager.start()
