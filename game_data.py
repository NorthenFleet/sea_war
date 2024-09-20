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

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GameData, cls).__new__(cls)
            cls._instance.initialize()  # Initialize all properties
        return cls._instance

    def initialize(self):
        """Initialize or reset game data."""
        self.units = {}  # Store all entities, key is entity ID
        self.player_units = {}  # Map player to their unit IDs
        self.unit_owner = {}  # Map entity to its owner player
        self.object_pool = ObjectPool(self.create_entity)

    def reset(self):
        """Reset game data to initial state."""
        # Release all entities back to the object pool
        for entity in self.units.values():
            self.object_pool.release(entity)
        # Reinitialize data structures
        self.initialize()

    def add_entity(self, entity_info, device, player_id):
        """Add a new entity to the game data, using object pool for reuse."""
        if entity_info.entity_id in self.units:
            print(f"Entity with ID {entity_info.entity_id} already exists.")
            return None

        # Acquire entity from object pool and add it to the game data
        entity = self.object_pool.acquire(entity_info, device)
        self.units[entity_info.entity_id] = entity

        # Map player to this entity
        if player_id not in self.player_units:
            self.player_units[player_id] = set()
        self.player_units[player_id].add(entity_info.entity_id)

        # Map entity to its owner
        self.unit_owner[entity_info.entity_id] = player_id

        return entity

    def remove_entity(self, entity_id):
        """Remove an entity from the game data."""
        if entity_id in self.units:
            entity = self.units.pop(entity_id)
            player_id = self.unit_owner.pop(entity_id, None)
            if player_id:
                self.player_units[player_id].discard(entity_id)
            # Release the entity back to the pool
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

    def create_entity(self, entity_info, device):
        """Factory function to create a new entity."""
        return Entity(entity_info)

    def configure_entity(self, entity, entity_info, device):
        """Configure an entity with the provided entity_info and device."""
        entity.reset(entity_info, device)
