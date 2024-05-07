class Weapon:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Equipment:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Entity:
    def __init__(self, id, x, y, speed_x, speed_y, health, endurance, weapons, equipment):
        self.id = id
        self.position = (x, y)
        self.speed = (speed_x, speed_y)
        self.health = health
        self.endurance = endurance
        self.weapons = [Weapon(**w) for w in weapons]
        self.equipment = [Equipment(**e) for e in equipment]


class Player:
    def __init__(self, name):
        self.name = name
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)
