from agents.base_agent import Base_Agent


class Rule_Agent(Base_Agent):
    def __init__(self, name, trainning_config=None, model=None):
        super().__init__()
        self.name = name
        self.model = model
        self.trainning_config = trainning_config

    def choose_action(self, observation, use_epsilon=None):
        print("我是规则智能体")
        return 1


    def __str__(self):
        return f"Rule_Agent({self.name})"
