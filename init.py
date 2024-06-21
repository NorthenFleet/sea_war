import json
from entities.entity import EntityInfo, Entity
from device import *


class DataLoader:
    def __init__(self, path):
        self.data = self.load_json(path)

    @staticmethod
    def load_json(path):
        with open(path, 'r') as file:
            return json.load(file)


class GamePlayer:
    def __init__(self):
        self.flight = []
        self.ship = []
        self.submarine = []
        pass


class Scenario(DataLoader):
    def __init__(self, path, name):
        super().__init__(path)
        self.name = name
        self.path = path
        self.name = name
        self.players = {}
        self.entities = []

    def load_scenario(self):
        for color, units in self.data.items():
            player = GamePlayer()
            for unit_type, unit_list in units.items():
                if unit_list:
                    if unit_type == 'flight':
                        player.flight = self.create_units(color, unit_list)
                        self.entities.append(player.flight)
                    elif unit_type == 'ship':
                        player.ship = self.create_units(color, unit_list)
                        self.entities.append(player.ship)
                    elif unit_type == 'submarine':
                        player.submarine = self.create_units(color, unit_list)
                        self.entities.append(player.submarine)
                    # 添加更多的单位类型处理
            self.players[color] = player
        return self.players, self.entities

    def create_units(self, color, unit_data):
        units = []
        for unit in unit_data:
            entity_info = EntityInfo()
            entity_info.side = color
            entity_info.entity_id = unit['id']
            entity_info.entity_type = unit['entity_type']
            entity_info.position = [unit['x'], unit['y']]
            entity_info.speed = unit['speed']
            entity_info.direction = unit['course']
            entity_info.hp = unit['health']
            entity_info.attack_power = 0  # 示例中未提供攻击力
            entity_info.weapons = unit['weapons']
            entity_info.equipments = unit['equipment']
            entity_info.sensor = Sensor()
            entity_info.launcher = Launcher(entity_info.id, 'missile', 4)
            units.append(Entity(entity_info))
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
