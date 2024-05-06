def ppo_train(env, model, optimizer, epochs=1000, max_steps=1000, gamma=0.99, epsilon=0.2, clip_value=0.2):
    for epoch in range(epochs):
        observations = []
        actions = []
        old_probs = []
        rewards = []

        for step in range(max_steps):
            observation = env.reset()
            done = False
            while not done:
                observations.append(observation)
                action_probs, _ = model(torch.tensor(observation, dtype=torch.float32))
                action_distribution = torch.distributions.Categorical(action_probs)
                action = action_distribution.sample()
                old_prob = action_distribution.log_prob(action)
                actions.append(action)
                old_probs.append(old_prob)

                observation, reward, done, _ = env.step(action.item())
                rewards.append(reward)

            # 计算累计奖励
            cumulative_rewards = []
            running_reward = 0
            for reward in reversed(rewards):
                running_reward = reward + gamma * running_reward
                cumulative_rewards.insert(0, running_reward)

            # 标准化累计奖励
            cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float32)
            cumulative_rewards = (cumulative_rewards - cumulative_rewards.mean()) / (cumulative_rewards.std() + 1e-5)

            # 计算策略和价值网络的损失
            actions = torch.tensor(actions, dtype=torch.int64)
            old_probs = torch.stack(old_probs)
            observations = torch.tensor(observations, dtype=torch.float32)
            new_probs, values = model(observations)
            new_probs = new_probs.gather(1, actions.view(-1, 1)).squeeze()
            ratios = torch.exp(new_probs - old_probs)
            advantages = cumulative_rewards - values.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - epsilon, 1 + epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (advantages ** 2).mean()

            # 计算总损失
            loss = policy_loss + 0.5 * value_loss

            # 清空缓存
            observations = []
            actions = []
            old_probs = []
            rewards = []

            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 输出训练信息
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

# 使用示例
if __name__ == "__main__":
    env = YourEnvironment()
    model = PPO(input_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ppo_train(env, model, optimizer)
