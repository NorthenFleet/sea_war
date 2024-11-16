import numpy as np


class Env():
    metadata = {'render.modes': ['human']}

    def __init__(self, game_config):
        super(Env, self).__init__()

    def reset_game(self, config):
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    def load_scenario(self, scenario):
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    def create_entity(self, entity_id, entity_type, position, speed, faction, hp, attack_power):
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    def destroy_entity(self, entity_id):
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    def attack(self, attacker_id, target_id, attack_range):
        raise NotImplementedError(
            "This method should be overridden by subclasses")

    def update(self, action_dict):
        raise NotImplementedError(
            "This method should be overridden by subclasses")
