class State:
    def on_enter(self, unit):
        pass

    def on_exit(self, unit):
        pass

    def update(self, unit):
        pass

    def perform_action(self, entity):
        pass


class MovingState(State):
    def perform_action(self, unit):
        print("Unit is moving.")
        # 实现移动逻辑


class CruisingState(State):
    def perform_action(self, entity):
        print(f"Entity {entity} is cruising.")


class AttackingState(State):
    def perform_action(self, entity):
        print(f"Entity {entity} is attacking.")



