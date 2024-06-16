class Carrier:
    def __init__(self, id, health, speed):
        self.id = id
        self.health = health
        self.speed = speed

    def move(self, x, y):
        print(f"Carrier {self.id} is moving to ({x}, {y})")


class Sensor:
    def __init__(self, range):
        self.range = range

    def detect(self):
        print(f"Sensor with range {self.range} is detecting")


class Launcher:
    def __init__(self, ammo_type, capacity):
        self.ammo_type = ammo_type
        self.capacity = capacity

    def fire(self):
        if self.capacity > 0:
            self.capacity -= 1
            print(
                f"Fired {self.ammo_type}, remaining capacity: {self.capacity}")
        else:
            print("No ammo left to fire")


class Ammo:
    def __init__(self, damage):
        self.damage = damage

    def explode(self):
        print(f"Ammo explodes with {self.damage} damage")
