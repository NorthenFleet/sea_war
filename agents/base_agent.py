class Base_Agent:
    def __init__(self, name):
        self.name = name

    def choose_action(self, observation):
        # 简单的策略：总是返回 0
        return 0

    
