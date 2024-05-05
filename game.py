import json

from render import Render
from env import GameEnv


class DataLoader:
    def __init__(self, path):
        self.data = self.load_json(path)

    @staticmethod
    def load_json(path):
        with open(path, 'r') as file:
            return json.load(file)


class Scenario(DataLoader):
    def __init__(self, path, name):
        super().__init__(path)
        self.name = name
        self.red = self.data["red"]
        self.blue = self.data["blue"]
        # self.units = self.create_units()

    def create_units(self):
        units = {}
        for faction, force_types in self.scenario.items():
            units[faction] = {}
            for force_type, details in force_types.items():
                units[faction][force_type] = [
                    {
                        "id": unit["id"],
                        "position": {"x": unit["x"], "y": unit["y"]},
                        "speed": {"x": unit["speed_x"], "y": unit["speed_y"]},
                        "health": unit["health"],
                        "endurance": unit["endurance"],
                        "weapons": unit["weapons"],
                        "equipment": unit["equipment"]
                    } for unit in details
                ]
        return units

    def display_units(self):
        for faction, force_types in self.units.items():
            print(f"Faction {faction}:")
            for force_type, units in force_types.items():
                print(f"  {force_type}:")
                for unit in units:
                    print(
                        f"    ID {unit['id']}: Position ({unit['position']['x']}, {unit['position']['y']}), Health {unit['health']}")
                    for weapon in unit['weapons']:
                        print(
                            f"      Weapon {weapon['type']}: {weapon['count']} units")
                    for equipment in unit['equipment']:
                        print(
                            f"      Equipment {equipment['type']}: {equipment['count']} units")


class Map(DataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.map_info = self.data['map_info']
        self.map_data = self.data['map_data']

    def display_map(self):
        for row in self.map_data:
            print(" ".join(map(str, row)))


class Weapon(DataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.platforms = self.data['platforms']
        self.weapons = self.data['weapons']

    def index(self):
        pass

    def update(self):
        pass


class Game():
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
