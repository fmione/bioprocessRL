import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import gymnasium as gym
import numpy as np
import keras
from keras import ops
from keras import layers
import tensorflow as tf
import KiwiGym_createEnv_v2


# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards

env = gym.make("kiwiGym-v2")
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

"""
## Implement Actor Critic network

This network learns two functions:

1. Actor: This takes as input the state of our environment and returns a
probability value for each action in its action space.
2. Critic: This takes as input the state of our environment and returns
an estimate of total rewards in the future.

In our implementation, they share the initial layer.
"""

max_steps_per_episode = 100
num_inputs = env.unwrapped.observation_space.shape[0] 
num_actions = len(env.unwrapped.action_space)
num_hidden_1 = 8192 # 2048, 4096, 8192, 16384, 32768
num_hidden_2 = 4096

inputs = layers.Input(shape=(num_inputs,))
hidden_1 = layers.Dense(num_hidden_1, activation="relu")(inputs)
hidden_2 = layers.Dense(num_hidden_1, activation="relu")(hidden_1)
outputs = [layers.Dense(mbr_action, activation="softmax")(hidden_2) for mbr_action in env.unwrapped.action_space.nvec] # creates each softmax output
critic = layers.Dense(1, activation='linear')(hidden_2)
model = keras.Model(inputs=inputs, outputs=outputs+[critic])

"""
## Train
"""

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

env.reset(seed=seed)

while True:  # Run until solved
    state, _ = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            state = ops.convert_to_tensor(state)
            state = ops.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            *action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0][0])

            # Sample action from action probability distribution
            actions = [np.random.choice(np.arange(len(np.squeeze(mbr_action))), p=np.squeeze(mbr_action)) for mbr_action in action_probs]
            action_probs_history.append(ops.log([action_probs[mbr][0][action] for mbr, action in enumerate(actions)]))

            # Apply the sampled action in our environment
            state, reward, done, _, _ = env.step(actions)
            
            # print(timestep, actions, reward)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up receiving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(tf.reduce_mean(-log_prob * diff))  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 100:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
"""
## Visualizations
In early stages of training:
![Imgur](https://i.imgur.com/5gCs5kH.gif)

In later stages of training:
![Imgur](https://i.imgur.com/5ziiZUD.gif)
"""