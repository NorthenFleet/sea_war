class Devices:
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
        self.weapons = [Devices(**w) for w in weapons]
        self.equipment = [Equipment(**e) for e in equipment]