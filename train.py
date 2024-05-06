from env import GameEnv
from init import Map, Weapon, Scenario
import torch.optim as optim
from network import *

class Train():
    def __init__(self) -> None:
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/scenario.json'
        map_path = 'data/map.json'

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)

        self.config = {"scenario": scenario,
                       "map": map,
                       "weapon": weapon}

        agent_modules = {
            "agent1": ("agents.ai_agent", "AI_Agent"),
            "agent2": ("agents.rule_agent", "Rule_Agent")
        }

        # 游戏逻辑
        self.game_env = GameEnv(name, agent_modules)
        self.current_step = None
        self.max_step = 1000

    def run(self):
        network_config = {
            "network_type": "dqn",
            "input_dim": 10,
            "output_dim": 5,
            "hidden_layers": 3,
            "hidden_units": [256, 256, 256]
        }
        model = configure_network(network_config)
        optimizer = optim.Adam(model.parameters())

        observation = self.game_env.reset_game(self.config)
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = {agent_name: agent.choose_action(
                observation) for agent_name, agent in self.game_env.agents.items()}
            observations, rewards, game_over, info = self.game_env.update(
                actions)

            self.current_step += 1
            if self.current_step > self.max_step:
                game_over = True
        print(self.current_step)


if __name__ == '__main__':
    train = Train()
    train.run()
