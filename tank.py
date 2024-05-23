import numpy as np 
from entity import Entity
from entity import Devices
from entity import Equipment

class Entity(Entity):
    def __init__(self, entity_id, entity_type, position, speed, faction, hp, attack_power, weapons, equipments):
        super().__init__()
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.position = np.array(position)
        self.speed = speed
        self.faction = faction
        self.hp = hp
        self.set_observation = []
        self.attack_power = attack_power
        self.alive = True

        self.weapons = [Devices(**w) for w in weapons]
        self.equipment = [Equipment(**e) for e in equipments]

    def update_position(self, move_direction, move_distance):
        if not self.alive:
            return
        self.position += np.array(move_direction) * move_distance

    
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
