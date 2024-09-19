from entity import Entity
from component import PositionComponent, HealthComponent, MovementComponent, PathfindingComponent, WeaponComponent
from state_management import CruisingState


class Aircraft(Entity):
    def __init__(self, event_manager):
        super().__init__()
        self.state = CruisingState()
        self.event_manager = event_manager
        self.position = None
        self.movement = None
        self.pathfinding = None
        self.health = None
        self.weapon = None

    def initialize(self, position, health, movement_speed, weapon_damage, weapon_range):
        """初始化实体及其组件"""
        self.position = PositionComponent(*position)
        self.health = HealthComponent(health, self.event_manager)
        self.movement = MovementComponent(movement_speed)
        self.pathfinding = PathfindingComponent()
        self.weapon = WeaponComponent(weapon_damage, weapon_range)

    def reset(self):
        """重置实体状态，准备放回对象池"""
        self.position = None
        self.movement = None
        self.pathfinding = None
        self.health = None
        self.weapon = None

    def change_state(self, new_state):
        self.state = new_state
        self.state.perform_action(self)
