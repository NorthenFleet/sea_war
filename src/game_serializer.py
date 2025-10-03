import json
import numpy as np

class GameSerializer:
    """
    游戏数据序列化类，用于将游戏数据转换为可通过网络传输的格式
    """
    
    @staticmethod
    def serialize_entity(entity):
        """序列化单个实体"""
        entity_data = {
            'id': entity.entity_id,
            'type': entity.entity_type,
            'position': entity.position.tolist() if isinstance(entity.position, np.ndarray) else entity.position,
            'direction': entity.direction.tolist() if isinstance(entity.direction, np.ndarray) else entity.direction,
            'faction': entity.faction,
            'hp': entity.hp,
            'alive': entity.alive,
            'speed': entity.speed
        }
        
        # 添加武器信息
        if hasattr(entity, 'weapons'):
            entity_data['weapons'] = [
                {
                    'type': weapon.type,
                    'range': weapon.range,
                    'damage': weapon.damage,
                    'cooldown': weapon.cooldown,
                    'current_cooldown': weapon.current_cooldown
                } for weapon in entity.weapons
            ]
            
        # 添加装备信息
        if hasattr(entity, 'equipment'):
            entity_data['equipment'] = [
                {
                    'type': equip.type,
                    'effect': equip.effect
                } for equip in entity.equipment
            ]
            
        return entity_data
    
    @staticmethod
    def serialize_game_state(game_data):
        """序列化整个游戏状态"""
        entities = game_data.get_all_entities()
        serialized_entities = [GameSerializer.serialize_entity(entity) for entity in entities]
        
        # 获取地图信息
        map_data = {
            'width': game_data.map_width,
            'height': game_data.map_height,
            'obstacles': game_data.obstacles if hasattr(game_data, 'obstacles') else []
        }
        
        # 获取阵营信息
        factions = {}
        for faction_name, faction in game_data.factions.items():
            factions[faction_name] = {
                'entity_ids': faction.entity_ids,
                'score': faction.score if hasattr(faction, 'score') else 0
            }
        
        # 构建完整游戏状态
        game_state = {
            'entities': serialized_entities,
            'map': map_data,
            'factions': factions,
            'game_time': game_data.game_time if hasattr(game_data, 'game_time') else 0
        }
        
        return game_state
    
    @staticmethod
    def deserialize_command(command_data):
        """反序列化命令数据"""
        from player import Command
        
        command_type = command_data.get('type')
        actor_id = command_data.get('actor_id')
        target = command_data.get('target')
        params = command_data.get('params', {})