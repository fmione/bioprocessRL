import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import gymnasium as gym
import numpy as np
import keras
from keras import ops
from keras import layers
import KiwiGym_createEnv_v5
import tensorflow as tf
# %% Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards

env = gym.make("kiwiGym-v5")
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
max_steps_per_episode = 100

num_inputs = env.unwrapped.observation_space.shape[0] 
num_actions = 1
# num_hidden_1 = 16384  # 2048, 4096, 8192, 16384, 32768
# num_hidden_2 = 4096
num_hidden_1 = 512  # 2048, 4096, 8192, 16384, 32768
num_hidden_2 = 264

lrn_rate = 1e-4

action_space=21

# %% Agents
def Actor(num_inputs ,num_actions ,num_hidden_1 ,num_hidden_2 ,action_space):
   
    inputs = layers.Input(shape=(num_inputs,))
    masked_input = layers.Masking(mask_value=-1)(inputs)

    hidden_1 = layers.Dense(num_hidden_1, activation="relu",kernel_initializer='he_uniform')(masked_input)
    hidden_2 = layers.Dense(num_hidden_2, activation="relu",kernel_initializer='he_uniform')(hidden_1)
    output = layers.Dense(action_space, activation="softmax", kernel_initializer='glorot_uniform')(hidden_2)  # creates each softmax output
    
    return keras.models.Model(inputs=inputs, outputs=output)


# %% Initialize

Encoding_matrix=np.eye(action_space)

huber_loss = keras.losses.Huber()
critic_value_history = []
rewards_history = []
state_episode=[]
state_action_episode=[]
action_episode=[]
returns_history=[]


action_log_probs_history=[]

agent=Actor(num_inputs ,num_actions ,num_hidden_1 ,num_hidden_2 ,action_space)
optimizer_agent = keras.optimizers.Adam(learning_rate=lrn_rate)

cnt_flag=0
running_reward = 0
episode_count = 0

env.reset(seed=seed)

# %%
while True:  # Run until solved
    episode_reward = 0
    cnt_flag+=1
    with tf.GradientTape() as tape:
        state, _ = env.reset()
        for timestep in range(1, max_steps_per_episode):

            state = ops.convert_to_tensor(state)
            state = ops.expand_dims(state, 0)

            action_probs=agent(state)

            actions =np.random.choice(np.arange(len(np.squeeze(action_probs))), p=np.squeeze(action_probs)) 
            
            action_log_probs_history.append( ops.log(action_probs+1e-9)[0])


            action_index=actions
            encoding_vector=np.array(Encoding_matrix[action_index,:])
            action_episode.append(ops.convert_to_tensor(encoding_vector,dtype="float32"))
            
            state, reward, done, _, _ = env.step(actions)
            
            
            if done==1:
                pass
                # print('reward: ',reward)
                # env.render()
            else:
                
                print('step: ',timestep,'actions: ',actions)
                
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        

        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns_np = np.array(returns)
        returns = (returns_np - np.mean(returns_np)) / (np.std(returns_np) + eps)
        returns_history.append(returns.tolist())

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        print('EPISODE: ',episode_count,'returns: ',episode_reward,' running reward: ',running_reward)
        
        if cnt_flag==1:

            cnt_flag=0

            action_log_probs_tofit = ops.stack(action_log_probs_history)
            returns_tofit=ops.stack(returns_history[0])
            action_vector_tofit = ops.stack(action_episode)
            
            history = zip(action_log_probs_tofit, action_vector_tofit, returns_tofit)
            
            agent_losses = []
            for log_prob, act_vec, ret in history:
                agent_losses.append(tf.reduce_sum(-log_prob * act_vec * ret))  


            print('EPISODE: ',episode_count, "LOSS: " ,float(sum(agent_losses)))
            
            actor_losses_sum=sum(agent_losses)
            
            actor_grads = tape.gradient(actor_losses_sum, agent.trainable_variables)
            optimizer_agent.apply_gradients(zip(actor_grads, agent.trainable_variables))
            
            action_log_probs_history.clear()
            rewards_history.clear()
            action_episode.clear()
            returns_history.clear()

        episode_count += 1
