from typing import List, Tuple, Callable, Optional
import heapq


class PathPlanner:
    """
    六角格路径规划骨架（A*）

    - neighbor_fn: (q,r) -> Iterable[(q,r)]
    - passable_fn: (q,r) -> bool  障碍判定回调
    - heuristic_fn: (a,b) -> int  启发式（六角距离）
    """

    def __init__(self,
                 neighbor_fn: Callable[[int, int], List[Tuple[int, int]]],
                 passable_fn: Callable[[int, int], bool],
                 heuristic_fn: Callable[[Tuple[int, int], Tuple[int, int]], int]):
        self.neighbor_fn = neighbor_fn
        self.passable_fn = passable_fn
        self.heuristic_fn = heuristic_fn

    def a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        frontier: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for nxt in self.neighbor_fn(*current):
                if not self.passable_fn(*nxt):
                    continue
                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.heuristic_fn(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        # 回溯
        if goal not in came_from:
            return []
        cur = goal
        path = [cur]
        while came_from[cur] is not None:
            cur = came_from[cur]
            path.append(cur)
        path.reverse()
        return path