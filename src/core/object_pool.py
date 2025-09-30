# 独立的对象池，直接接收event_manager

class ObjectPool:
    def __init__(self, create_func, event_manager):
        self.create_func = create_func
        self.pool = []
        self.event_manager = event_manager  # 直接管理 event_manager

    def acquire(self, *args, **kwargs):
        if self.pool:
            entity = self.pool.pop()
            # 传递 event_manager，重置实体状态
            entity.reset(*args, self.event_manager, **kwargs)
            return entity
        else:
            # 创建新实体，并传递 event_manager
            return self.create_func(*args, self.event_manager, **kwargs)

    def release(self, entity):
        """将实体重置后放回池中"""
        entity.reset(None, None)  # 清除实体状态
        self.pool.append(entity)
