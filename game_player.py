import json





def load_scenario(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    players = {}
    for player_color in ['red', 'blue']:
        player_data = data.get(player_color, {})
        player = Player()

        for entity_type in ['flight', 'ship', 'submarine']:
            entities = player_data.get(entity_type, [])
            setattr(player, entity_type, entities)

        players[player_color] = player

    return players


scenario_file_path = 'scenario.json'
players = load_scenario(scenario_file_path)

# 玩家红色的实体
print("Player Red Entities:")
for entity_type, entities in players['red'].__dict__.items():
    print(f"{entity_type}: {entities}")

# 玩家蓝色的实体
print("\nPlayer Blue Entities:")
for entity_type, entities in players['blue'].__dict__.items():
    print(f"{entity_type}: {entities}")
