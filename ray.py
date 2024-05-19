import ray


@ray.remote
class DistributedGameEnv:
    def __init__(self, config):
        self.env = Env(name="SC2Env", player_config=config["player_config"])
        self.agent1 = DQNAgent(
            name="DQN_Agent_1", state_size=84*84, action_size=2)
        self.agent2 = DQNAgent(
            name="DQN_Agent_2", state_size=84*84, action_size=2)

    def run_episode(self):
        state = self.env.reset_game()
        state = state.flatten()
        done = False
        total_reward1 = 0
        total_reward2 = 0
        while not done:
            action1 = self.agent1.choose_action(state)
            action2 = self.agent2.choose_action(state)
            actions = [action1, action2]
            next_state, reward, done, _ = self.env.update(actions)
            next_state = next_state.flatten()
            self.agent1.remember(state, action1, reward, next_state, done)
            self.agent2.remember(state, action2, reward, next_state, done)
            state = next_state
            total_reward1 += reward
            total_reward2 += reward
        return total_reward1, total_reward2

    def train(self, batch_size=32):
        if len(self.agent1.memory) > batch_size:
            self.agent1.train(batch_size)
        if len(self.agent2.memory) > batch_size:
            self.agent2.train(batch_size)

    def close(self):
        self.env.close()


ray.init()

num_envs = 4
envs = [DistributedGameEnv.remote(
    config={'max_steps': 100, 'player_config': player_config}) for _ in range(num_envs)]


def run_training(envs, num_episodes):
    for i in range(num_episodes):
        results = ray.get([env.run_episode.remote() for env in envs])
        for env in envs:
            env.train.remote()
        print(f"Episode {i+1}, results: {results}")


run_training(envs, 100)

for env in envs:
    env.close.remote()

ray.shutdown()
