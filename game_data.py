from entities.entity import Entity, EntityInfo


class ObjectPool:
    def __init__(self, create_func):
        self.create_func = create_func
        self.pool = []

    def acquire(self, *args, **kwargs):
        if self.pool:
            entity = self.pool.pop()
            # Reset the entity to its initial state
            entity.reset(*args, **kwargs)
            return entity
        else:
            return self.create_func(*args, **kwargs)

    def release(self, entity):
        self.pool.append(entity)


class GameData:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GameData, cls).__new__(cls)
            cls._instance.initialize()  # 初始化所有属性
        return cls._instance

    def initialize(self):
        """初始化或重置游戏数据"""
        self.units = set()  # 全部单位的集合
        self.units_ids = set()  # 存储所有实体ID集合
        self.player_units = dict()  # 玩家到其单位的映射
        self.unit_owner = dict()  # 单位到其玩家的映射
        self.object_pool = ObjectPool(self.create_entity)

    def reset(self):
        """重置游戏数据到初始状态"""
        # 释放所有实体回对象池
        for unit in self.units:
            self.object_pool.release(unit)
        # 重置所有数据结构
        self.initialize()

    def add_entity(self, entity_info, device, player_id):
        # Check if the entity ID already exists
        if entity_info.entity_id in self.units_ids:
            print(f"Entity with ID {entity_info.entity_id} already exists.")
            return None

        entity = self.object_pool.acquire(entity_info, device)
        self.units.add(entity)
        if player_id not in self.player_units:
            self.player_units[player_id] = set()
        self.player_units[player_id].add(entity)
        self.unit_owner[entity] = player_id
        self.units_ids.add(entity_info.entity_id)  # Track the ID
        return entity

    def remove_entity(self, entity):
        if entity in self.units:
            self.units.remove(entity)
            player_id = self.unit_owner.pop(entity, None)
            if player_id and entity in self.player_units[player_id]:
                self.player_units[player_id].remove(entity)
            self.units_ids.remove(entity.id)  # Remove the ID from tracking
            self.object_pool.release(entity)  # Return the entity to the pool

    def get_all_units(self):
        """返回所有单位的集合的字典"""
        return set(self.units)

    def get_player_units(self, player_id):
        """根据玩家ID返回该玩家的单位集合的字典"""
        return set(self.player_units.get(player_id, []))

    def get_unit_owner(self, unit):
        """返回给定单位的玩家ID"""
        return self.unit_owner.get(unit, None)

    def create_entity(self, entity_info, device):
        # This function initializes a new entity
        return Entity(entity_info)

    def configure_entity(self, entity, entity_info, device):
        # Reset the entity with new configuration
        entity.reset(entity_info, device)
