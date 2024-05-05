from agents.base_agent import Base_Agent


class AI_Agent(Base_Agent):
    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, observation):
        print("我是AI智能体")
        return 1

    def __str__(self):
        return f"AI_Agent({self.name})"
