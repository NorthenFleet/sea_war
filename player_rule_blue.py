from player import Player
from entities.entity import *


class BluePlayer(Player):
    def __init__(self, name, device_table):
        super().__init__()
        self.name = name
        self.device_table = device_table
        self.entities = []
        self.enemies = []

    def choose_action(self, side):
        print("我是蓝方智能体")
        self.data_process(side)
        for entity in self.entities:
            entity_type = entity.get_component(EntityTypeComponent)
            position = entity.get_component(PositionComponent)
            

    def data_process(self, data):
        self.entities = data.entities
        self.enemies = data.enemies
