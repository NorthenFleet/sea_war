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
        self.players = {}
        self.entities = []

    def load_scenario(self, device):
        for color, units in self.data.items():
            player = GamePlayer()
            for unit_type, unit_list in units.items():
                if unit_list:
                    if unit_type == 'flight':
                        player.flight = self.create_entity(
                            color, unit_list, device)
                        self.entities.append(player.flight)
                    elif unit_type == 'ship':
                        player.ship = self.create_entity(
                            color, unit_list, device)
                        self.entities.append(player.ship)
                    elif unit_type == 'submarine':
                        player.submarine = self.create_entity(
                            color, unit_list, device)
                        self.entities.append(player.submarine)
                    # 添加更多的单位类型处理
            self.players[color] = player
        return self.players, self.entities

    def create_entity(self, color, unit_data, device):
        entities = []
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
            entity_info.weapons = None
            entity_info.sensor = None
            entity_info.launcher = None

            entity = Entity(entity_info)
            for weapon_name in unit_data.weapons:
                entity.add_weapon(device.get_weapon(weapon_name))
            for sensor_name in unit_data.sensor:
                entity.add_sensor(device.get_sensor(sensor_name))

            entities.append()
        return entities

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
        pass

    def display_map(self):
        for row in self.data:
            print(" ".join(map(str, row)))


class Device(DataLoader):
    def __init__(self, path):
        super().__init__(path)

        self.weapons = {}
        self.sensors = {}
        self.launchers = {}

        for weapon_data in self.data['weapons']:
            weapon = Weapon(**weapon_data)
            self.weapons[weapon.name] = weapon
        for sensor_data in self.data['sensors']:
            sensor = Sensor(**sensor_data)
            self.sensors[sensor.name] = sensor

    def get_weapon(self, name):
        return self.weapon.get(name)

    def get_sensor(self, name):
        return self.sensor.get(name)


class Initializer():
    def __init__(self, game_config):
        device_path = game_config['device_path']
        scenarios_path = game_config['scenarios_path']
        map_path = game_config['map_path']

        self.map = Map(map_path)
        self.device_table = Device(device_path)
        self.scenario = Scenario(scenarios_path, game_config["name"])
        players = self.scenario.load_scenario(self.device_table)

        env_config = {
            "name": game_config["name"],
            "scenario": self.scenario,
            "map": self.map,
            "weapon": self.device_table,
            "players": players
        }

        return env_config
