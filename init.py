import json
from device import *


class DataLoader:
    def __init__(self, path):
        self.data = self.load_json(path)

    @staticmethod
    def load_json(path):
        with open(path, 'r') as file:
            return json.load(file)


class Side:
    def __init__(self, name):
        self.name = name
        self.entities = {}
        self.enemys = {}

    def set_entities(self, entities):
        self.entities = entities

    def get_entities(self, id):
        return self.entities.get(id)


class Scenario(DataLoader):
    def __init__(self, path):
        super().__init__(path)
        self.path = path


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
