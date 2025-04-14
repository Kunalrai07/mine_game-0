import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from diamond import DiamondMinesEnv

# Create environment
env = DiamondMinesEnv()

# Optional: Sanity check to validate env
check_env(env)

# Define the model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=500_000)


# Save the trained model
model.save("diamond_dqn_model")

print("🎯 RL Model Trained & Saved Successfully ✅")
