# object_pool.py
class ObjectPool:
    def __init__(self, create_func):
        self.create_func = create_func
        self.pool = []

    def acquire(self, *args):
        if self.pool:
            entity = self.pool.pop()
            entity.reset(*args)  # Reset the entity to its initial state
            return entity
        else:
            return self.create_func(*args)

    def release(self, entity):
        self.pool.append(entity)
