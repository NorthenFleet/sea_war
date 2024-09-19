class Component:
    pass


class PositionComponent(Component):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class HealthComponent(Component):
    def __init__(self, health):
        self.health = health


class System:
    def __init__(self):
        self.entities = []

    def add_entity(self, entity):
        self.entities.append(entity)


class MovementSystem(System):
    def update(self):
        for entity in self.entities:
            if hasattr(entity, 'position'):
                # Move entity
                entity.position.x += 1
                entity.position.y += 1
                print(
                    f'Entity {entity.id} moved to ({entity.position.x}, {entity.position.y})')
