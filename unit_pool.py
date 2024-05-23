class UnitPool:
    def __init__(self, unit_type):
        self.unit_type = unit_type
        self._available_units = []
        self._in_use_units = set()

    def acquire_unit(self):
        if self._available_units:
            unit = self._available_units.pop()
        else:
            unit = self.unit_type()
        self._in_use_units.add(unit)
        return unit

    def release_unit(self, unit):
        self._in_use_units.remove(unit)
        self._available_units.append(unit)


class Soldier:
    def __init__(self):
        self.health = 100
        self.position = (0, 0)

    def reset(self):
        self.health = 100
        self.position = (0, 0)


# 创建一个士兵对象池
soldier_pool = UnitPool(Soldier)

# 获取一个士兵对象
soldier = soldier_pool.acquire_unit()

# 使用士兵对象
soldier.position = (10, 20)

# 释放士兵对象并重置状态
soldier.reset()
soldier_pool.release_unit(soldier)
