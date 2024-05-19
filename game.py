from render import Render
from env import Env
from init import Map, Weapon, Scenario
from gameLogic import GameLogic


class Game():
    def __init__(self) -> None:
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/scenario.json'
        map_path = 'data/map.json'

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)
        game_logic = GameLogic(scenario, map, weapon)

        # 环境设置
        self.game_config = {
            "name": name,
            "scenario": scenario,
            "map": map,
            "weapon": weapon,
            "GameLogic": game_logic}

        self.game_env = Env(self.game_config)
        self.current_step = None
        self.render = Render()
        self.max_step = 1000

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
            if self.current_step > self.max_step:
                game_over = True
        print(self.current_step)


# 使用示例
if __name__ == '__main__':
    game = Game()
    game.run()
