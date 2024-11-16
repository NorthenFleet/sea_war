class ObjectPool:
    def __init__(self, cls, *args, **kwargs):
        self._cls = cls
        self._args = args
        self._kwargs = kwargs
        self._available = []
        self._in_use = set()

    def acquire(self):
        if self._available:
            obj = self._available.pop()
        else:
            obj = self._cls(*self._args, **self._kwargs)
        self._in_use.add(obj)
        return obj

    def release(self, obj):
        self._in_use.remove(obj)
        self._available.append(obj)

# 示例对象类


class GameUnit:
    def __init__(self, unit_type):
        self.unit_type = unit_type


# 创建一个对象池
unit_pool = ObjectPool(GameUnit, 'Soldier')

# 获取对象
unit = unit_pool.acquire()

# 使用对象
print(unit.unit_type)  # 输出: Soldier

# 释放对象
unit_pool.release(unit)
