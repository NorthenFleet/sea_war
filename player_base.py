class Player_Base:
    def __init__(self, communication):
        # self.name = agents_config["name"]
        # self.player_type = agents_config["player_type"]
        # self.agents = agents_config if isinstance(
        #     agents_config, list) else [agents_config]
        self.communication = communication
        

        # 动态导入智能体模块
        # self.players = {name: getattr(__import__(module), cls)(
        #     name) for name, (module, cls) in player_config.items()}

        # self.agents = {}
        # for name, (module, cls, training_config) in AI_config.items():
        #     agent_class = getattr(__import__(module), cls)
        #     if training_config["model"] is not None:
        #         self.players[name] = agent_class(name, training_config)
        #     else:
        #         self.players[name] = agent_class(name)

    def choose_action(self, state):
        pass
    
