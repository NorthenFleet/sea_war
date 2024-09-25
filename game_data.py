from entities.entity import Entity, EntityInfo


class ObjectPool:
    def __init__(self, create_func):
        self.create_func = create_func
        self.pool = []

    def acquire(self, *args, **kwargs):
        if self.pool:
            entity = self.pool.pop()
            # Reset the entity to its initial state using provided args
            entity.reset(*args, **kwargs)
            return entity
        else:
            return self.create_func(*args, **kwargs)

    def release(self, entity):
        """Release entity back into the pool after resetting its state."""
        entity.reset(None, None)  # Optionally reset entity before reuse
        self.pool.append(entity)


class GameData:
    _instance = None

    def __new__(cls, event_manager, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GameData, cls).__new__(cls)
            cls._instance.initialize(event_manager)  # 初始化所有属性
        return cls._instance

    def initialize(self, event_manager):
        """初始化或重置游戏数据，并保存事件管理器的引用。"""
        self.units = {}  # 存储所有实体，键为实体ID
        self.player_units = {}  # 玩家到其单位ID的映射
        self.unit_owner = {}  # 单位到其玩家的映射
        self.event_manager = event_manager  # 保存事件管理器引用
        self.object_pool = ObjectPool(self.create_entity)

    def reset(self):
        """重置游戏数据到初始状态。"""
        # 将所有实体释放回对象池
        for entity in self.units.values():
            self.object_pool.release(entity)
        # 重新初始化数据结构
        self.initialize(self.event_manager)

    def add_entity(self, entity_info, device, player_id):
        """通过对象池添加一个新实体到游戏数据。"""
        if entity_info.entity_id in self.units:
            print(f"Entity with ID {entity_info.entity_id} already exists.")
            return None

        # 从对象池获取实体，并将其添加到游戏数据中
        entity = self.object_pool.acquire(
            entity_info, device, self.event_manager)
        self.units[entity_info.entity_id] = entity

        # 映射玩家到该实体
        if player_id not in self.player_units:
            self.player_units[player_id] = set()
        self.player_units[player_id].add(entity_info.entity_id)

        # 映射实体到它的所属玩家
        self.unit_owner[entity_info.entity_id] = player_id

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
        if entity_id in self.units:
            return self.units[entity_id].get_position()
        return None

    def get_all_unit_ids(self):
        """Return a list of all unit IDs."""
        return list(self.units.keys())

    def get_player_unit_ids(self, player_id):
        """Return a list of unit IDs for a given player."""
        return list(self.player_units.get(player_id, []))

    def get_unit_owner(self, entity_id):
        """Return the owner player ID of a given entity."""
        return self.unit_owner.get(entity_id, None)

    def create_entity(self, entity_info, device, event_manager):
        """工厂函数，创建一个新实体。"""
        return Entity(entity_info, device, event_manager)

    def configure_entity(self, entity, entity_info, device):
        """配置实体的属性和设备。"""
        entity.reset(entity_info, device, self.event_manager)
