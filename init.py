import json


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
        self.players = {}
        for color, units in self.data.items():
            self.players[color] = {
                "flight": [],
                "ship": [],
                "submarine": [],
                "missile": [],
                "anti_air": []
            }
            for unit_type, unit_list in units.items():
                for unit in unit_list:
                    self.players[color][unit_type].append(unit)
        self.num_players = len(self.players)

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
