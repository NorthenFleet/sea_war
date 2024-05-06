from agents.base_agent import Base_Agent


class Rule_Agent(Base_Agent):
    def __init__(self, name, model=None):
        super().__init__(name)
        self.model = model

    def choose_action(self, observation):
        print("我是规则智能体")
        return 1

    def __str__(self):
        return f"Rule_Agent({self.name})"
