import numpy as np
from entity import Entity
from entity import Devices
from entity import Equipment


class Tank(Entity):
    def __init__(self, entity_id, entity_type, position, speed, direction, faction, hp, attack_power, weapons, equipments):
        super().__init__()
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.position = np.array(position)
        self.speed = speed
        self.direction = np.array(direction)
        self.faction = faction
        self.hp = hp
        self.set_observation = []
        self.attack_power = attack_power
        self.alive = True
        self.ration = 1
        self.weapons = [Devices(**w) for w in weapons]
        self.equipment = [Equipment(**e) for e in equipments]

    def set_ration(self, ration):
        self.ration = ration

    def local_move(self, move_direction):
        if not self.alive:
            return
        self.position += np.array(move_direction) * self.speed * self.ratio

    def global_move(self, entity_id, destination):
        direction_vector = np.array(destination) - np.array(self.position)
        distance = np.linalg.norm(direction_vector)
        speed = self.entities[entity_id].get('speed', 1)

        if distance < speed:
            new_position = destination
        else:
            direction_vector_normalized = direction_vector / distance
            new_position = self.position + direction_vector_normalized * speed

        self.entities[entity_id]['position'] = new_position
        print(f"Entity {entity_id} moved to {new_position}")

    def set_observer(self, observer):
        self.set_observer = observer

    def update_state(self, speed=None,  devices=None, equipments=None):
        if speed is not None:
            self.speed = speed
        if devices is not None:
            self.devices = devices
        if equipments is not None:
            self.equipments = equipments

    def take_damage(self, damage):
        if not self.alive:
            return
        self.hp -= damage
        if self.hp <= 0:
            self.alive = False
