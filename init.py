import json
from entities.entity import EntityInfo, Entity
from device import *
from object_pool import ObjectPool


class DataLoader:
    def __init__(self, path):
        self.data = self.load_json(path)

    @staticmethod
    def load_json(path):
        with open(path, 'r') as file:
            return json.load(file)


class Side:
    def __init__(self):
        self.units = []


class Scenario(DataLoader):
    def __init__(self, path, name):
        super().__init__(path)
        self.name = name
        self.path = path
        self.players = {}
        self.entities = []
        self.entity_pool = ObjectPool(self.create_entity)
        self.entity_registry = {}  # Centralized registry

    def load_scenario(self, device):
        for color, unit_list in self.data.items():
            side = Side()
            for unitid, unit in unit_list.items():
                entity_info = EntityInfo(
                    entity_id=unit['id'],
                    entity_type=unit['entity_type'],
                    position=(unit['x'], unit['y']),
                    speed=unit['speed'],
                    direction=unit['course'],
                    hp=unit['health'],
                    weapons=[w['type'] for w in unit['weapons']],
                    sensor=[s['type'] for s in unit['sensor']]
                )
                entity = self.entity_pool.acquire(entity_info, device)
                # Register the entity by ID
                self.entity_registry[unit["id"]] = entity
                # setattr(side, entities)
                self.entities.append(entity)
            self.players[color] = side
        return self.players, self.entities, self.entity_registry

    def create_entity(self, entity_info, device):
        entity = Entity(entity_info)
        for weapon_name in entity_info.weapons:
            weapon = device.get_weapon(weapon_name)
            if weapon:
                entity.add_weapon(weapon)
        for sensor_name in entity_info.sensor:
            sensor = device.get_sensor(sensor_name)
            if sensor:
                entity.add_sensor(sensor)
        return entity

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
        return self.weapons.get(name)

    def get_sensor(self, name):
        return self.sensors.get(name)


class GameData:
    def __init__(self):
        # 从全局配置字典中加载参数
        self.enities = []

    def reset(self):
        self.enities = []

    def set_enities(self, enities):
        self.enities = enities


class Initializer:
    def __init__(self, game_config):
        device_path = game_config['device_path']
        scenarios_path = game_config['scenarios_path']
        map_path = game_config['map_path']

        self.map = Map(map_path)
        self.device_table = Device(device_path)
        self.scenario = Scenario(scenarios_path, game_config["name"])
        sides, entities, entity_registry = self.scenario.load_scenario(
            self.device_table)

        game_data = GameData()
        game_data.set_enities(entities)

        self.env_config = {
            "name": game_config["name"],
            "scenario": self.scenario,
            "map": self.map,
            "weapon": self.device_table,
            "sides": sides,
            "entity_registry": entity_registry,
            "game_data": game_data
        }

    def get_env_config(self):
        return self.env_config
