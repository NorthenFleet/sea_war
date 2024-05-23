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
        pass
    
    def update_position(self):
        pass

    
    def set_observer(self, observer):
        pass

    def update_state(self):
        pass
             
    def take_damage(self, damage):
        pass
