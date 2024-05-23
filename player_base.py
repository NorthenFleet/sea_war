class Player_Base:
    def __init__(self):
        # self.name = agents_config["name"]
        # self.player_type = agents_config["player_type"]
        # self.agents = agents_config if isinstance(
        #     agents_config, list) else [agents_config]


        # 动态导入智能体模块
        # self.players = {name: getattr(__import__(module), cls)(
        #     name) for name, (module, cls) in player_config.items()}

        self.memory = []

    def choose_action(self, state):
        pass
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))