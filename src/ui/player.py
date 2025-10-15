class Command:
    command_list = None  # 静态变量，所有指令共享的 CommandList

    @classmethod
    def set_command_list(cls, command_list):
        """为所有命令类设置一个共享的 CommandList"""
        cls.command_list = command_list

    def __init__(self, command_type, actor, target=None, params=None):
        self.command_type = command_type  # 指令类型，如移动、攻击
        self.actor = actor  # 发起指令的主体
        self.target = target  # 指令目标
        self.params = params or {}  # 指令的具体参数

        # 当一个命令实例化时，自动加入 CommandList
        if Command.command_list is not None:
            Command.command_list.add_command(self)

# 具体的移动指令


class MoveCommand(Command):
    def __init__(self, actor, target_position, speed):
        super().__init__('move', actor=actor,
                         target=target_position, params={'speed': speed})

# 具体的攻击指令


class AttackCommand(Command):
    def __init__(self, actor, target, weapon):
        super().__init__('attack', actor=actor,
                         target=target, params={'weapon': weapon})

# 停止指令


class StopCommand(Command):
    def __init__(self, actor):
        super().__init__('stop', actor=actor)

# 设定单位速度（绝对值）
class SetSpeedCommand(Command):
    def __init__(self, actor, speed):
        super().__init__('set_speed', actor=actor, params={'speed': speed})

# 旋转航向（相对角度，度数，顺时针为正）
class RotateHeadingCommand(Command):
    def __init__(self, actor, delta_deg):
        super().__init__('rotate_heading', actor=actor, params={'delta_deg': float(delta_deg)})

# 传感器开关切换（若未携带 enabled，则在环境中取反）
class ToggleSensorCommand(Command):
    def __init__(self, actor, enabled=None):
        super().__init__('sensor_toggle', actor=actor, params={'enabled': enabled})

# 攻击最近的敌方单位（无需事先点选目标）
class AttackNearestCommand(Command):
    def __init__(self, actor):
        super().__init__('attack_nearest', actor=actor)

# 指令列表，用于管理智能体生成的所有指令


class CommandList:
    def __init__(self):
        self.commands = []

    def add_command(self, command):
        """将指令加入命令列表"""
        self.commands.append(command)

    def get_commands(self):
        """返回所有收集到的指令"""
        return self.commands

    def reset(self):
        """清空指令列表"""
        self.commands.clear()

# 智能体基类，代表玩家或AI角色


class Player:
    def __init__(self, name):
        self.name = name  # 玩家名称
        self.memory = []  # 存储智能体的记忆（用于学习算法等）

    # 智能体选择动作的核心方法，生成指令列表
    def choose_action(self, state):
        pass

    # 存储智能体的记忆（用于学习算法等）
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
