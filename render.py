import pygame

class Render:
    def __init__(self):
        pass

    def display(self, state):
        # display_units
        for faction, units in state.items():
            print(f"Faction {faction}:")
        for unit in units:
            print(f"  {unit['count']}x {unit['type']} with weapons:")
            for weapon, details in unit['weapons'].items():
                print(f"    - {weapon}: {details['count']} units")