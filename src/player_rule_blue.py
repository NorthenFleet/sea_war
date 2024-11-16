from player import *
from entities.entity import *


class BluePlayer(Player):
    def __init__(self, name, device_table):
        super().__init__(name)
        self.name = name
        self.device_table = device_table
        self.entities = []
        self.enemies = []

    def choose_action(self, side):
        print("我是蓝方智能体")
        """智能体根据状态选择动作，返回指令列表"""
        command_list = CommandList()
        Command.set_command_list(command_list)  # 设置命令列表

        self.detect(side)
        self.attackCheck(side)
        self.data_process(side)
        self.target_distribute()
        self.attack()
        self.move()

        # 返回收集到的所有指令
        return command_list

    def detect(self, side):
        self.entities = side.entities
        self.enemies = side.enemies

    def attackCheck(self, side):
        # 创建指令（自动加入到 command_list 中）
        # 创建指令（自动加入到 command_list 中）

        AttackCommand(
            actor=self.entities[0].entity_id, target="enemy_1", weapon="missile")

    def target_distribute(self):
        pass

    def attack(self):
        pass

    def move(self):
        actor_id = "水面舰S5"
        MoveCommand(actor_id,
                    target_position=(100, 1200), speed=100)

    def data_process(self, data):
        for entity in self.entities:
            entity_type = entity.entity_type
            position = entity.get_component(PositionComponent)
