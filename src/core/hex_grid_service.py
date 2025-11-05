import math
from typing import Tuple


class HexGridService:
    """
    点顶六角格坐标换算与吸附服务（轴坐标/立方体坐标）

    - size: 六角格半径（世界坐标单位）
    - 原点: 以世界坐标原点为六角网格原点（可按需扩展偏移）
    """

    def __init__(self, size: float, origin: Tuple[float, float] = (0.0, 0.0)):
        self.size = float(size)
        self.origin = origin

    # 轴坐标(q, r) -> 世界坐标(x, y)
    def axial_to_world(self, q: float, r: float) -> Tuple[float, float]:
        sz = self.size
        x = sz * math.sqrt(3) * (q + r / 2.0)
        y = sz * 1.5 * r
        return (x + self.origin[0], y + self.origin[1])

    # 世界坐标(x, y) -> 轴坐标(q, r)（浮点）
    def world_to_axial(self, x: float, y: float) -> Tuple[float, float]:
        sz = self.size
        # 移除原点偏移
        px = x - self.origin[0]
        py = y - self.origin[1]
        q = (math.sqrt(3) / 3.0 * px - 1.0 / 3.0 * py) / sz
        r = (2.0 / 3.0 * py) / sz
        return (q, r)

    # 立方体坐标取整（用于将浮点轴坐标吸附到最近六角格中心）
    def _cube_round(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        rx = round(x)
        ry = round(y)
        rz = round(z)
        x_diff = abs(rx - x)
        y_diff = abs(ry - y)
        z_diff = abs(rz - z)
        if x_diff > y_diff and x_diff > z_diff:
            rx = -ry - rz
        elif y_diff > z_diff:
            ry = -rx - rz
        else:
            rz = -rx - ry
        return int(rx), int(ry), int(rz)

    def axial_round(self, q: float, r: float) -> Tuple[int, int]:
        x = q
        z = r
        y = -x - z
        rx, ry, rz = self._cube_round(x, y, z)
        return int(rx), int(rz)  # 返回(q, r)

    def snap_world_to_hex_center(self, x: float, y: float) -> Tuple[float, float]:
        """将世界坐标吸附到最近六角格中心并返回世界坐标中心点"""
        qf, rf = self.world_to_axial(x, y)
        q, r = self.axial_round(qf, rf)
        return self.axial_to_world(q, r)

    def neighbors(self, q: int, r: int):
        """返回六个邻接格（轴坐标）"""
        dirs = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]
        for dq, dr in dirs:
            yield (q + dq, r + dr)

    def distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        aq, ar = a
        bq, br = b
        return int((abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) / 2)