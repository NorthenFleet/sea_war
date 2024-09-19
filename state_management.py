class State:
    def on_enter(self, unit):
        pass

    def on_exit(self, unit):
        pass

    def update(self, unit):
        pass


class MovingState(State):
    def update(self, unit):
        print("Unit is moving.")
        # 实现移动逻辑


class AttackingState(State):
    def update(self, unit):
        print("Unit is attacking.")
        # 实现攻击逻辑




    
