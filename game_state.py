import numpy as np


class GameState:
    def __init__(self):
        self.entities = {}
        self.entity_matrix = None

    def add_entity(self, entity):
        self.entities[entity.entity_id] = entity
        self.update_entity_matrix()

    def remove_entity(self, entity_id):
        if entity_id in self.entities:
            del self.entities[entity_id]
            self.update_entity_matrix()

    def update_entity_matrix(self):
        # 使用numpy数组来存储所有实体的关键属性
        num_entities = len(self.entities)
        if num_entities == 0:
            self.entity_matrix = None
            return
        # 假设我们需要存储位置(x, y)、速度、HP、存活状态等信息
        self.entity_matrix = np.zeros((num_entities, 5))
        for i, entity in enumerate(self.entities.values()):
            self.entity_matrix[i, 0] = entity.position[0]
            self.entity_matrix[i, 1] = entity.position[1]
            self.entity_matrix[i, 2] = entity.speed
            self.entity_matrix[i, 3] = entity.hp
            self.entity_matrix[i, 4] = entity.alive
