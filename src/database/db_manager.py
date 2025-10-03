import sqlite3
import json
import os
import threading
from pathlib import Path

class DBManager:
    """数据库管理类，负责处理所有数据库操作"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DBManager, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance
    
    def __init__(self):
        if self.initialized:
            return
            
        # 确保数据库目录存在
        db_dir = Path('/Users/sunyi/WorkSpace/sea_war_python/data/db')
        db_dir.mkdir(exist_ok=True)
        
        self.db_path = db_dir / 'sea_war.db'
        self.conn = None
        self.cursor = None
        self.connect()
        self.initialized = True
    
    def connect(self):
        """连接到SQLite数据库"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 使查询结果可以通过列名访问
            self.cursor = self.conn.cursor()
            print(f"成功连接到数据库: {self.db_path}")
            self._create_tables()
        except sqlite3.Error as e:
            print(f"数据库连接错误: {e}")
    
    def _create_tables(self):
        """创建数据库表结构"""
        # 静态数据表
        
        # 武器平台参数表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS weapons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            damage REAL,
            range REAL,
            reload_time REAL,
            accuracy REAL,
            ammo_capacity INTEGER,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 装备参数表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS equipments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            effect TEXT,
            duration REAL,
            cooldown REAL,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 传感器参数表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensors (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            detection_range REAL,
            accuracy REAL,
            properties TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 场景表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS scenarios (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            map_id TEXT,
            config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 地图表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS maps (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            terrain_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 动态数据表
        
        # 游戏会话表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_sessions (
            id TEXT PRIMARY KEY,
            scenario_id TEXT,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            status TEXT,
            winner TEXT,
            config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scenario_id) REFERENCES scenarios (id)
        )
        ''')
        
        # 游戏状态表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            step INTEGER,
            state_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES game_sessions (id)
        )
        ''')
        
        # 玩家表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS players (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            faction TEXT,
            type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 玩家会话表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS player_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            session_id TEXT,
            faction TEXT,
            score INTEGER DEFAULT 0,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES players (id),
            FOREIGN KEY (session_id) REFERENCES game_sessions (id)
        )
        ''')
        
        # 命令历史表
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            player_id TEXT,
            step INTEGER,
            command_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES game_sessions (id),
            FOREIGN KEY (player_id) REFERENCES players (id)
        )
        ''')
        
        self.conn.commit()
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
    
    # 静态数据操作方法
    
    def import_weapons_from_json(self, json_path):
        """从JSON文件导入武器数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                weapons_data = json.load(f)
            
            for weapon_id, weapon in weapons_data.items():
                self.cursor.execute(
                    '''INSERT OR REPLACE INTO weapons 
                       (id, name, type, damage, range, reload_time, accuracy, ammo_capacity, properties)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (
                        weapon_id,
                        weapon.get('name', ''),
                        weapon.get('type', ''),
                        weapon.get('damage', 0),
                        weapon.get('range', 0),
                        weapon.get('reload_time', 0),
                        weapon.get('accuracy', 0),
                        weapon.get('ammo_capacity', 0),
                        json.dumps(weapon)
                    )
                )
            
            self.conn.commit()
            print(f"成功导入武器数据: {len(weapons_data)}条记录")
            return True
        except Exception as e:
            print(f"导入武器数据失败: {e}")
            return False
    
    def import_scenario_from_json(self, json_path):
        """从JSON文件导入场景数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scenario_data = json.load(f)
            
            scenario_id = os.path.basename(json_path).split('.')[0]
            scenario_name = scenario_data.get('name', scenario_id)
            
            self.cursor.execute(
                '''INSERT OR REPLACE INTO scenarios 
                   (id, name, description, config)
                   VALUES (?, ?, ?, ?)''',
                (
                    scenario_id,
                    scenario_name,
                    scenario_data.get('description', ''),
                    json.dumps(scenario_data)
                )
            )
            
            self.conn.commit()
            print(f"成功导入场景数据: {scenario_name}")
            return True
        except Exception as e:
            print(f"导入场景数据失败: {e}")
            return False
    
    def import_map_from_json(self, json_path):
        """从JSON文件导入地图数据"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            
            map_id = os.path.basename(json_path).split('.')[0]
            map_name = map_data.get('name', map_id)
            
            self.cursor.execute(
                '''INSERT OR REPLACE INTO maps 
                   (id, name, width, height, terrain_data)
                   VALUES (?, ?, ?, ?, ?)''',
                (
                    map_id,
                    map_name,
                    map_data.get('width', 0),
                    map_data.get('height', 0),
                    json.dumps(map_data)
                )
            )
            
            self.conn.commit()
            print(f"成功导入地图数据: {map_name}")
            return True
        except Exception as e:
            print(f"导入地图数据失败: {e}")
            return False
    
    # 动态数据操作方法
    
    def create_game_session(self, scenario_id, config=None):
        """创建新的游戏会话"""
        import uuid
        import datetime
        
        session_id = str(uuid.uuid4())
        start_time = datetime.datetime.now().isoformat()
        
        try:
            self.cursor.execute(
                '''INSERT INTO game_sessions 
                   (id, scenario_id, start_time, status, config)
                   VALUES (?, ?, ?, ?, ?)''',
                (
                    session_id,
                    scenario_id,
                    start_time,
                    'active',
                    json.dumps(config) if config else '{}'
                )
            )
            
            self.conn.commit()
            print(f"创建游戏会话: {session_id}")
            return session_id
        except Exception as e:
            print(f"创建游戏会话失败: {e}")
            return None
    
    def save_game_state(self, session_id, step, state_data):
        """保存游戏状态"""
        try:
            self.cursor.execute(
                '''INSERT INTO game_states 
                   (session_id, step, state_data)
                   VALUES (?, ?, ?)''',
                (
                    session_id,
                    step,
                    json.dumps(state_data)
                )
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"保存游戏状态失败: {e}")
            return False
    
    def save_command(self, session_id, player_id, step, command_data):
        """保存玩家命令"""
        try:
            self.cursor.execute(
                '''INSERT INTO command_history 
                   (session_id, player_id, step, command_data)
                   VALUES (?, ?, ?, ?)''',
                (
                    session_id,
                    player_id,
                    step,
                    json.dumps(command_data)
                )
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"保存命令历史失败: {e}")
            return False
    
    def get_game_state(self, session_id, step=None):
        """获取游戏状态"""
        try:
            if step is not None:
                self.cursor.execute(
                    '''SELECT * FROM game_states 
                       WHERE session_id = ? AND step = ?
                       ORDER BY timestamp DESC LIMIT 1''',
                    (session_id, step)
                )
            else:
                self.cursor.execute(
                    '''SELECT * FROM game_states 
                       WHERE session_id = ?
                       ORDER BY step DESC LIMIT 1''',
                    (session_id,)
                )
            
            row = self.cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'session_id': row['session_id'],
                    'step': row['step'],
                    'state_data': json.loads(row['state_data']),
                    'timestamp': row['timestamp']
                }
            return None
        except Exception as e:
            print(f"获取游戏状态失败: {e}")
            return None
    
    def get_scenario(self, scenario_id):
        """获取场景数据"""
        try:
            self.cursor.execute(
                'SELECT * FROM scenarios WHERE id = ?',
                (scenario_id,)
            )
            
            row = self.cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'description': row['description'],
                    'config': json.loads(row['config'])
                }
            return None
        except Exception as e:
            print(f"获取场景数据失败: {e}")
            return None