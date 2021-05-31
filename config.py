import growspace
import experiment_buddy
import numpy as np
## this setting is for rainbow on pong

env = "GrowSpaceEnv-Control-v0"

stop_timesteps=1000000
num_atoms=51
noisy=True
gamma= 0.99
lr: .0001
hiddens=[512]
learning_starts=10000
buffer_size=50000
rollout_fragment_length=4
train_batch_size=32

epsilon_timesteps=2
final_epsilon=0.0
target_network_update_freq=500
prioritized_replay=True
prioritized_replay_alpha=0.5
final_prioritized_replay_beta=1.0
prioritized_replay_beta_annealing_timesteps=400000
n_step=3
v_min=20
v_max=0
gpu=True

grayscale=False
zero_mean=False
dim =42
experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(
    "mila",
    sweep_yaml="",
    proc_num=1,
    wandb_kwargs={"entity": "growspace"}
)