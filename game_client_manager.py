import time
import threading
from env import Env
from player_human import HumanPlayer
from player_AI import AIPlayer


class GameClientManager:
    def __init__(self, server_host, server_port, use_ai=False):
        self.network_client = NetworkClient(server_host, server_port)
        self.env = Env(name="SC2Env", player_config=player_config)
        self.human_player = HumanPlayer(name="HumanPlayer")

        # 玩家设置
        player_config = {
            # "red": ("agents.ai_agent", "AI_Agent", "model"),
            # "blue": ("agents.rule_agent", "Rule_Agent")
            "red": ("player_AI", "AI_Agent", "model"),
            "blue": ("rule_agent", "Rule_Agent")
        }

        self.players = {}
        for name, (module, cls, model) in player_config.items():
            player_class = getattr(__import__(module), cls)
            if model is not None:
                self.players[name] = player_class(name, model)
            else:
                self.players[name] = player_class(name)

    def start(self):
        threading.Thread(target=self.network_client.start).start()
        self.run_game()

    def run_game(self):
        state = self.env.reset_game(config)
        done = False
        while not done:
            if self.use_ai:
                action = self.ai_agent.act(state)
            else:
                action = self.human_player.choose_action(state)

            self.network_client.send_action(action)
            while self.network_client.received_actions is None:
                # Wait for the server to broadcast the actions
                time.sleep(0.01)
            # Convert the received string back to a dictionary
            action_dict = eval(self.network_client.received_actions)
            self.network_client.received_actions = None
            observations, rewards, done, _ = self.env.update(action_dict)
            if self.use_ai:
                self.ai_agent.remember(
                    state, action, rewards[0], observations, done)
                if len(self.ai_agent.memory) > 32:
                    self.ai_agent.replay(32)
            if done:
                state = self.env.reset_game(config)

    def get_human_action(self):
        # Implement method to get human player action
        return 'move_up'  # Example action


if __name__ == "__main__":
    use_ai = '--ai' in sys.argv
    if '--train' in sys.argv:
        ray.init()
        tune.run(train_dqn, config=ray_config)
    else:
        game_client_manager = GameClientManager(
            server_host='127.0.0.1', server_port=9999, use_ai=use_ai)
        game_client_manager.start()
