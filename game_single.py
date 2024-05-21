from render import Render
from env import Env
from init import Map, Weapon, Scenario
from game_logic import GameLogic


class Game():
    def __init__(self) -> None:
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/scenario.json'
        map_path = 'data/map.json'

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)

        # 环境设置
        self.env_config = {
            "name": name,
            "scenario": scenario,
            "map": map,
            "weapon": weapon
        }

        self.game_env = Env(self.env_config)
        self.current_step = None
        self.render = Render()
        self.max_step = 1000

        # 智能体
        AI_config = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001,
            "model": "PPO",
            "state_size": 100,
            "action_size": 50,
            "use_epsilon": True,
        }

        # 玩家设置
        player_config = {
            # "red": ("agents.ai_agent", "AI_Agent", "model"),
            # "blue": ("agents.rule_agent", "Rule_Agent")
            "red": ("player_AI", "AIPlayer", AI_config),
            "blue": ("player_rule", "RulePlayer", None)
        }

        self.players = {}
        for name, (path, module, config) in player_config.items():
            player_class = getattr(__import__(path), module)
            if config is not None:
                self.players[name] = player_class(config)
            else:
                self.players[name] = player_class()

        self.config = {
            "game_config": self.env_config,
            "player_config": player_config,
        }

    def run(self):
        observation = self.game_env.reset_game(self.config)
        game_over = False
        self.current_step = 0
        while not game_over:
            actions = {agent_name: agent.choose_action(
                observation) for agent_name, agent in self.players.items()}
            observations, rewards, game_over, info = self.game_env.update(
                actions)

            self.current_step += 1
            print(self.current_step)
            
            if self.current_step > self.max_step:
                game_over = True
            


# 使用示例
if __name__ == '__main__':
    game = Game()
    game.run()
