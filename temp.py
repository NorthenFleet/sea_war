from env import Env
from init import Map, Weapon, Scenario
import numpy as np
import torch.optim as optim
from replay_bufer import ReplayBuffer
from model_config import *
import torch
import torch.nn as nn
from gym import spaces


class Devices:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Equipment:
    def __init__(self, type, count):
        self.type = type
        self.count = count


class Entity:
    def __init__(self, id, x, y, speed_x, speed_y, health, endurance, weapons, equipment):
        self.id = id
        self.position = (x, y)
        self.speed = (speed_x, speed_y)
        self.health = health
        self.endurance = endurance
        self.weapons = [Weapon(**w) for w in weapons]
        self.equipment = [Equipment(**e) for e in equipment]


class GameLogic:
    def __init__(self, scenario_config, map_config, weapon_config):
        self.scenario = scenario_config
        self.map = map_config
        self.weapon = weapon_config
        self.entities = {}
        self.current_step = 0
        self.game_over = False

    def load_scenario(self, scenario):
        self.scenario = scenario

    def create_entity(self, entity_id, entity_type, position, speed, faction, hp, attack_power):
        self.entities[entity_id] = {
            "type": entity_type,
            "position": position,
            "speed": speed,
            "faction": faction,
            "hp": hp,
            "attack_power": attack_power
        }

    def delete_entity(self, entity_id):
        if entity_id in self.entities:
            del self.entities[entity_id]

    def local_move(self, entity_id, move_direction, move_distance=None):
        if entity_id not in self.entities:
            print(f"Entity {entity_id} does not exist.")
            return

        current_position = self.entities[entity_id]['position']
        speed = self.entities[entity_id].get('speed', 1)  # 假设实体有速度属性
        move_distance = move_distance if move_distance is not None else speed

        # 计算新位置
        new_position = current_position + \
            np.array(move_direction) * move_distance
        self.entities[entity_id]['position'] = new_position
        print(f"Entity {entity_id} moved locally to {new_position}")

    def global_move(self, entity_id, destination):
        if entity_id not in self.entities:
            print(f"Entity {entity_id} does not exist.")
            return

        current_position = self.entities[entity_id]['position']
        direction_vector = np.array(destination) - np.array(current_position)
        distance = np.linalg.norm(direction_vector)
        speed = self.entities[entity_id].get('speed', 1)  # 假设实体有速度属性

        if distance < speed:
            new_position = destination
        else:
            direction_vector_normalized = direction_vector / distance
            new_position = current_position + direction_vector_normalized * speed

        self.entities[entity_id]['position'] = new_position
        print(f"Entity {entity_id} moved to {new_position}")

    def detect_entities(self, entity_id, detection_range):
        if entity_id not in self.entities:
            return {}

        current_position = np.array(self.entities[entity_id]['position'])
        visible_entities = {}
        for other_id, data in self.entities.items():
            if other_id != entity_id:
                other_position = np.array(data['position'])
                if np.linalg.norm(current_position - other_position) <= detection_range:
                    visible_entities[other_id] = data
        return visible_entities

    def attack(self, attacker_id, target_id, attack_range):
        if attacker_id not in self.entities or target_id not in self.entities:
            return "Invalid entity"

        attacker = self.entities[attacker_id]
        target = self.entities[target_id]

        # 检查目标是否在攻击范围内
        attacker_pos = np.array(attacker['position'])
        target_pos = np.array(target['position'])
        if np.linalg.norm(attacker_pos - target_pos) > attack_range:
            return "Target out of range"

        # 执行攻击
        damage = attacker['attack_power']
        target['hp'] -= damage
        if target['hp'] <= 0:
            self.delete_entity(target_id)
            return f"Target {target_id} destroyed"
        return f"Attacked {target_id}, {damage} damage dealt"

    def step(self, actions):
        # 处理动作，更新状态
        for entity_id, action in actions.items():
            if action == 'move':
                # 假设每次移动改变位置1
                self.move_entity(
                    entity_id, self.entities[entity_id]['position'] + 1)
            elif action == 'delete':
                self.delete_entity(entity_id)

        self.current_step += 1
        if self.current_step > 100:  # 示例结束条件
            self.game_over = True

        return self.detect_entities()


class Env():
    metadata = {'render.modes': ['human']}

    def __init__(self, name, agent_modules):
        super(Env, self).__init__()
        # 动态导入智能体模块
        # self.agents = {name: getattr(__import__(module), cls)(
        #     name) for name, (module, cls) in agent_modules.items()}

        self.agents = {}
        for name, (module, cls, training_config, model) in agent_modules.items():
            agent_class = getattr(__import__(module), cls)
            if model is not None:
                self.agents[name] = agent_class(name, training_config, model)
            else:
                self.agents[name] = agent_class(name)

        self.action_space = spaces.Discrete(2)  # 假设每个智能体的动作空间相同
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32)

        self.name = name
        self.Scenario = None
        self.map = None
        self.weapon = None
        self.state = None

        self.current_step = None

    def reset_game(self, config):
        # 重置游戏
        self.state = config["scenario"]
        self.map = config["map"]
        self.weapon = config["weapon"]
        self.current_step = 0
        self.game_over = False
        print("Game starts with the following units:")

        return {name: self.observation_space.sample() for name in self.agents}

    def update(self, action_dict):
        # 解析动作字典，执行动作
        rewards = {}
        for agent_name, action in action_dict.items():
            # 这里的奖励函数非常简单：如果动作是1，奖励为1，否则为0
            rewards[agent_name] = 1 if action == 1 else 0

        # 简单示例：观测值为随机
        observations = {name: self.observation_space.sample()
                        for name in self.agents}
        done = False
        return observations, rewards, done, {}


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def save_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, name, episodes):
        file_name = 'models/' + name + '-' + str(episodes) + '.pth'
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()  # Set the model to evaluation mode


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, observations, actions, rewards,
             next_observations, done):
        self.memory.append((observations, actions, rewards,
                            next_observations, done))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class BodyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_layers=2, hidden_units=128):
        super(BodyNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PolicyNetwork(nn.Module):
    def __init__(self, body_network, output_dim):
        super(PolicyNetwork, self).__init__()
        self.body_network = body_network
        self.head = nn.Linear(body_network.layers[-2].out_features, output_dim)

    def forward(self, x):
        features = self.body_network(x)
        return torch.softmax(self.head(features), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, body_network):
        super(ValueNetwork, self).__init__()
        self.body_network = body_network
        self.head = nn.Linear(body_network.layers[-2].out_features, 1)

    def forward(self, x):
        features = self.body_network(x)
        return self.head(features)


class ActorCritic(BaseModel):
    def __init__(self, input_dim, output_dim, hidden_layers=2, hidden_units=128):
        super(AC, self).__init__()
        self.body_network = BodyNetwork(input_dim, hidden_layers, hidden_units)
        self.policy_network = PolicyNetwork(self.body_network, output_dim)
        self.value_network = ValueNetwork(self.body_network)

    def forward(self, x):
        return self.policy_network(x), self.value_network(x)


class Player:
    def __init__(self, name):
        self.name = name


class Base_Agent:
    def __init__(self):
        pass

    def choose_action(self, observation):
        # 简单的策略：总是返回 0
        return 0


class AI_Agent(Base_Agent):
    def __init__(self, name, trainning_config=None, model=None):
        super().__init__()
        self.name = name
        self.model = model
        self.trainning_config = trainning_config

    def choose_action(self, state, use_epsilon):
        print("我是AI智能体")
        if use_epsilon and np.random.rand() <= self.trainning_config["epsilon"]:
            return random.randrange(self.model.output_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())

    def __str__(self):
        return f"AI_Agent({self.name})"


class Rule_Agent(Base_Agent):
    def __init__(self, name, trainning_config=None, model=None):
        super().__init__()
        self.name = name
        self.model = model
        self.trainning_config = trainning_config

    def choose_action(self, observation, use_epsilon=None):
        print("我是规则智能体")
        return 1

    def add_entity(self, entity):
        self.entities.append(entity)

    def __str__(self):
        return f"Rule_Agent({self.name})"


class Train():
    def __init__(self) -> None:
        name = 'battle_royale'
        weapons_path = 'data/weapons.json'
        scenarios_path = 'data/scenario.json'
        map_path = 'data/map.json'

        # weapons_path = weapons_json
        # scenarios_path = scenario_json
        # map_path = map_json

        scenario = Scenario(scenarios_path, name)
        map = Map(map_path)
        weapon = Weapon(weapons_path)

        # 环境
        self.input_dim = 10
        self.output_dim = 5
        self.game_config = {"scenario": scenario,
                            "map": map,
                            "weapon": weapon}
        self.current_step = None
        self.max_step = 1000

        # 训练
        network_config = {
            "model_type": "PPO",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }
        self.model = model_config(**network_config)
        self.use_epsilon = True
        self.replay_buffer = ReplayBuffer(capacity=2000)

        self.training_config = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.995,
            "learning_rate": 0.001
        }

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.training_config["learning_rate"])

        # 智能体
        agent_modules = {
            "agent1": ("agents.ai_agent", "AI_Agent", self.training_config, self.model),
            "agent2": ("agents.rule_agent", "Rule_Agent", None, None)
        }
        self.game_env = Env(name, agent_modules)

    def run(self):
        obs = self.game_env.reset_game(self.game_config)
        done = False
        self.current_step = 0
        while not done:
            actions = {agent_name: agent.choose_action(
                obs, self.use_epsilon) for agent_name, agent in self.game_env.agents.items()}
            next_obs, rewards, done, info = self.game_env.update(
                actions)
            next_obs = np.reshape(
                next_obs, [1, self.input_dim])

            self.replay_buffer.push(obs, actions, rewards,
                                    next_obs, done)

            obs = next_obs

            self.current_step += 1
            if self.current_step > self.max_step:
                done = True
        print(self.current_step)


if __name__ == '__main__':
    train = Train()
    train.run()
