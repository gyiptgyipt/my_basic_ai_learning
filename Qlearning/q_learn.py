import gymnasium as gym
import numpy as np

# 1. Environment ဖန်တီးခြင်း (is_slippery=False သည် Deterministic ဖြစ်စေရန်)
env = gym.make("FrozenLake-v1", is_slippery=False)

# State နှင့် Action အရေအတွက်ကို ရယူခြင်း
state_space_size = env.observation_space.n   # FrozenLake 4x4 အတွက် 16
action_space_size = env.action_space.n       # ဘယ်၊ ညာ၊ အပေါ်၊ အောက် အတွက် 4

# 2. Q-Table ကို Zeros များဖြင့် စတင်တည်ဆောက်ခြင်း (Shape: 16 x 4)
q_table = np.zeros((state_space_size, action_space_size))

# 3. Hyperparameters သတ်မှတ်ခြင်း
total_episodes = 2000        # Training ပေးမည့် Episode အရေအတွက်
learning_rate = 0.8          # Alpha (α)
discount_factor = 0.95       # Gamma (γ)

# Epsilon-Greedy Strategy အတွက် Parameters
epsilon = 1.0                # Initial Exploration Rate (100% Random)
max_epsilon = 1.0            # Exploration Probability အမြင့်ဆုံးတန်ဖိုး
min_epsilon = 0.01           # Exploration Probability အနိမ့်ဆုံးတန်ဖိုး
decay_rate = 0.005           # Epsilon လျှော့ချမည့် Rate

# 4. Training Loop
for episode in range(total_episodes):
    state, info = env.reset()
    done = False
    
    for step in range(100): # Maximum Steps per Episode
        # Epsilon-Greedy ရွေးချယ်မှု
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore: Random Action ယူမည်
        else:
            action = np.argmax(q_table[state, :]) # Exploit: Q-Table မှ အမြင့်ဆုံး Action ယူမည်

        # Action ပြုလုပ်ပြီး ရလဒ်များ ရယူခြင်း
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Bellman Equation ဖြင့် Q-Table ကို Update လုပ်ခြင်း
        # Q(s,a) = Q(s,a) + α * [R + γ * max Q(s',a') - Q(s,a)]
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state

        if done:
            break

    # Epsilon (Exploration Rate) ကို တဖြည်းဖြည်း လျှော့ချခြင်း (Decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

print("Training ပြီးစီးပါပြီ။")
print("\nသင်ယူထားသော Q-Table Matrix:")
print(np.round(q_table, 2))

env.close()

# 5. Trained Agent ကို Render လုပ်၍ Visual Test ပြုလုပ်ခြင်း
test_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state, info = test_env.reset()

print("\nTrained Agent ဖြင့် စမ်းသပ်မောင်းနှင်နေပါသည်။")
for step in range(100):
    # Exploitation သီးသန့် (အကောင်းဆုံး Action ကိုသာ ရွေးမည်)
    action = np.argmax(q_table[state, :])
    new_state, reward, terminated, truncated, info = test_env.step(action)
    
    if terminated or truncated:
        if reward == 1.0:
            print("Goal သို့ အောင်မြင်စွာ ရောက်ရှိပါပြီ။")
        else:
            print("ရေခဲပေါက်ထဲ ပြုတ်ကျသွားပါသည်။")
        break
    state = new_state

test_env.close()
