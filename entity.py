import numpy as np 

class Devices:
    def __init__(self, type, count):
        self.type = type
        self.count = count

class Equipment:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Entity:
    def __init__(self, entity_id, entity_type, position, speed, faction, hp, attack_power, weapons, equipments):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.position = np.array(position)
        self.speed = speed
        self.faction = faction
        self.hp = hp
        self.attack_power = attack_power
        self.alive = True

        self.weapons = [Devices(**w) for w in weapons]
        self.equipment = [Equipment(**e) for e in equipments]

    def update_position(self, move_direction, move_distance):
        if not self.alive:
            return
        self.position += np.array(move_direction) * move_distance

    def update_action(self, speed):
        self.speed = speed


    def take_damage(self, damage):
        if not self.alive:
            return
        self.hp -= damage
        if self.hp <= 0:
            self.alive = False
