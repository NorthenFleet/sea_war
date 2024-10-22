from entities.entity import *


class ObjectPool:
    def __init__(self, create_func):
        self.create_func = create_func
        self.pool = []

    def acquire(self, *args, **kwargs):
        if self.pool:
            entity = self.pool.pop()
            # 重置实体并传入事件管理器
            entity.reset(*args, **kwargs)
            return entity
        else:
            # 使用传入的参数（包括event_manager）创建实体
            return self.create_func(*args, **kwargs)

    def release(self, entity):
        """释放实体回到对象池中，准备重用。"""
        entity.reset(None, None, None)  # 在重置时传递空值
        self.pool.append(entity)


class GameData:
    _instance = None

    def __new__(cls, event_manager, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GameData, cls).__new__(cls)
            cls._instance.initialize()  # 初始化所有属性
        return cls._instance

    def initialize(self):
        """初始化或重置游戏数据。"""
        self.units = set()  # 全部单位的集合
        self.player_units = dict()  # 玩家到其单位的映射
        self.unit_owner = dict()  # 单位到其玩家的映射
        self.entity_ids = set()  # 存储所有实体的ID
        self.distance_table = {} 
        self.object_pool = ObjectPool(self.create_entity)

    def reset(self):
        """重置游戏数据到初始状态。"""
        # 将所有实体释放回对象池
        for entity in self.units:
            self.object_pool.release(entity)
        # 重新初始化数据结构
        self.initialize()

    def distance_table_compute(self):
        """计算距离表。"""
        self.distance_table = {}
        

    def add_entity(self, entity_info, device, player_id):
        """通过对象池添加一个新实体到游戏数据。"""
        if entity_info.entity_id in self.units:
            print(f"Entity with ID {entity_info.entity_id} already exists.")
            return None

        # 从对象池获取实体，并将其添加到游戏数据中
        # entity = self.object_pool.acquire(
        #     entity_info, device, self.event_manager)
        entity = self.object_pool.acquire(
            entity_info.entity_id, entity_info.entity_type)

        entity.add_component(PositionComponent(entity_info.position))

        if entity_info.speed:
            entity.add_component(MovementComponent(
                entity_info.speed, entity_info.heading))
        if entity_info.sensors:
            for sensor in entity_info.sensors:
                entity.add_component(SensorComponent(
                    sensor["sensor_type"]))
        if entity_info.health:
            entity.add_component(HealthComponent(entity_info.health))

        # 映射玩家到该实体
        if player_id not in self.player_units:
            self.player_units[player_id] = set()
        self.player_units[player_id].add(entity)
        self.units.add(entity)

        # 映射实体到它的所属玩家
        self.unit_owner[entity_info.entity_id] = player_id

        position = entity.get_component(PositionComponent)

        return entity

    def remove_entity(self, entity_id):
        """从游戏数据中移除一个实体。"""
        if entity_id in self.units:
            entity = self.units.pop(entity_id)
            player_id = self.unit_owner.pop(entity_id, None)
            if player_id:
                self.player_units[player_id].discard(entity_id)
            # 将实体释放回对象池
            self.object_pool.release(entity)

    def get_entity_pos(self, entity_id):
        """Get the position of an entity by its ID."""
        for entity in self.units:
            if entity.entity_id == entity_id:
                return entity.get_component(PositionComponent)

    def get_entity(self, entity_id):
        """Get the position of an entity by its ID."""
        for entity in self.units:
            if entity.entity_id == entity_id:
                return entity

    def get_all_entities(self):
        """Return a list of all unit IDs."""
        return self.units

    def get_player_unit_ids(self, player_id):
        """Return a list of unit IDs for a given player."""
        pass

    def get_unit_owner(self, entity_id):
        """Return the owner player ID of a given entity."""
        return self.unit_owner.get(entity_id, None)

    def create_entity(self, entity_id, entity_type):
        """工厂函数，创建一个新实体。"""
        return Entity(entity_id, entity_type)

    def configure_entity(self, entity, entity_info, device):
        """配置实体的属性和设备。"""
        entity.reset(entity_info, device, self.event_manager)
