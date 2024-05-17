from render import Render
from env import Env
from init import Map, Weapon, Scenario


class Game():
    def __init__(self) -> None:
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/scenario.json'
        map_path = 'data/map.json'

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)

        self.game_config = {
                       "name":name, 
                       "scenario": scenario,
                       "map": map,
                       "weapon": weapon}

        # 智能体
        player_config = {
            "agent1": ("agents.ai_agent", "AI_Agent", self.ai_config),
            "agent2": ("agents.rule_agent", "Rule_Agent", None)
        }
        self.game_env = Env(name, player_config)

        # 游戏逻辑
        self.game_env = Env(name, player_config)
        self.current_step = None
        self.render = Render()
        self.max_step = 1000

    def run(self):
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


# 使用示例
if __name__ == '__main__':
    game = Game()
    game.run()
