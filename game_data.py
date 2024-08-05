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
            cls._instance.units = set()  # 全部单位的集合
            cls._instance.player_units = dict()  # 玩家到其单位的映射
            cls._instance.unit_owner = dict()  # 单位到其玩家的映射
            cls._instance.object_pool = ObjectPool(cls._instance.create_entity)
        return cls._instance

    def reset(self):
        self.units = set()
        self.player_units = dict()
        self.unit_owner = dict()

    def get_all_units(self):
        """返回所有单位的集合的字典"""
        return set(self.units)

    def get_player_units(self, player_id):
        """根据玩家ID返回该玩家的单位集合的字典"""
        return set(self.player_units.get(player_id, []))

    def get_unit_owner(self, unit):
        """返回给定单位的玩家ID"""
        return self.unit_owner.get(unit, None)

    def add_entity(self, entity_info, device, player_id):
        # Use the object pool to acquire an entity
        entity = self.object_pool.acquire(entity_info, device)
        # Add the entity to the main set and player-specific mapping
        self.units.add(entity)
        if player_id not in self.player_units:
            self.player_units[player_id] = set()
        self.player_units[player_id].add(entity)
        self.unit_owner[entity] = player_id

    def remove_entity(self, entity):
        if entity in self.units:
            self.units.remove(entity)
            player_id = self.unit_owner.pop(entity, None)
            if player_id and entity in self.player_units[player_id]:
                self.player_units[player_id].remove(entity)
            self.object_pool.release(entity)  # Return the entity to the pool

    def create_entity(self, entity_info, device):
        # This function initializes a new entity
        entity = Entity(entity_info)
        self.configure_entity(entity, entity_info, device)
        return entity

    def configure_entity(self, entity, entity_info, device):
        # Configure weapons and sensors for the entity
        for weapon_name in entity_info.weapons:
            weapon = device.get_weapon(weapon_name)
            if weapon:
                entity.add_weapon(weapon)
        for sensor_name in entity_info.sensor:
            sensor = device.get_sensor(sensor_name)
            if sensor:
                entity.add_sensor(sensor)
