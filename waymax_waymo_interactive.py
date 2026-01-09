# %% [markdown]
# # Waymax + Waymo Motion Dataset - Interactive Training
#
# **Interactive notebook** to train policy with Waymax simulator + Waymo Motion Dataset:
# - Load Waymo Motion Dataset (TFRecord format)
# - Initialize Waymax simulator
# - Train Policy Network with RL
# - Visualize and evaluate
#
# **Repository:** https://github.com/PhamPhuHoa-23/TrafficGamer

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Install core packages
!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning==2.0.0
!pip install -q torch-geometric
!pip install -q tensorflow  # For Waymo dataset
!pip install -q waymax-io  # Waymax simulator
!pip install -q waymo-open-dataset-tf-2-11-0  # Waymo dataset utilities

# %%
# Check versions
import torch
import tensorflow as tf
print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

cuda_version = torch.version.cuda.replace('.', '') if torch.cuda.is_available() else 'cpu'
print(f"CUDA version: {cuda_version}")

# %%
# Install PyG dependencies
!pip install -q torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_version[:3]}.html

# %% [markdown]
# ## 2. Setup Environment

# %%
import os
import sys
import warnings
import yaml
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')
tf.config.set_visible_devices([], 'GPU')  # Disable TF GPU (use PyTorch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# %% [markdown]
# ## 3. Clone TrafficGamer Repository

# %%
# Clone repo
TRAFFICGAMER_DIR = Path("TrafficGamer")

if not TRAFFICGAMER_DIR.exists():
    print("📥 Cloning TrafficGamer...")
    !git clone https://github.com/PhamPhuHoa-23/TrafficGamer.git
    print("✅ Cloned successfully")
else:
    print("✅ TrafficGamer already exists")

sys.path.insert(0, str(TRAFFICGAMER_DIR.absolute()))
os.chdir(TRAFFICGAMER_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %%
# Install requirements
!pip install -q -r requirements.txt
!pip install -q neptune

# %% [markdown]
# ## 4. Import Modules

# %%
from predictors.qcnet import QCNet
from datasets import WaymoDataset
from transforms import TargetBuilder
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# Import RL algorithms
from algorithm.TrafficGamer import TrafficGamer
from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO
from algorithm.mappo import MAPPO

# Import utilities
from utils.rollout import PPO_process_batch
from utils.utils import seed_everything

print("✅ All modules imported successfully")

# %% [markdown]
# ## 5. Configuration
#
# **🔧 EDIT HERE - Configure paths and parameters:**

# %%
# ============================================
# 🔧 CONFIGURATION - EDIT HERE
# ============================================

CONFIG = {
    # Data paths
    'data_root': '/kaggle/input/waymo-motion-dataset',  # Waymo dataset root
    'qcnet_ckpt': '/kaggle/input/qcnet-waymo/QCNet_Waymo.ckpt',  # QCNet checkpoint
    'split': 'val',  # 'train', 'val', or 'test'
    
    # Dataset settings (Waymo: 11 history + 80 future @ 10Hz)
    'num_historical_steps': 11,
    'num_future_steps': 80,
    
    # Training settings
    'seed': 42,
    'batch_size': 4,
    'num_workers': 2,
    'max_epochs': 10,
    
    # RL algorithm
    'rl_algorithm': 'TrafficGamer',  # 'TrafficGamer', 'MAPPO', or 'CCE_MAPPO'
    'rl_config_file': 'TrafficGamer.yaml',
    
    # Scenario settings
    'scenario_id': 1,
    'controlled_agents': [0, 1],  # Agent indices to control
    
    # Training hyperparameters
    'learning_rate_actor': 3e-4,
    'learning_rate_critic': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    
    # Safety constraints
    'distance_limit': 5.0,  # Collision threshold (meters)
    'cost_quantile': 48,
    'penalty_initial_value': 1.0,
    
    # Evaluation
    'eval_freq': 5,
    'save_checkpoint': True,
}

seed_everything(CONFIG['seed'])

print("✅ Configuration loaded")
print(f"   Data: {CONFIG['data_root']}")
print(f"   QCNet checkpoint: {CONFIG['qcnet_ckpt']}")
print(f"   RL Algorithm: {CONFIG['rl_algorithm']}")
print(f"   History/Future: {CONFIG['num_historical_steps']}/{CONFIG['num_future_steps']}")

# %% [markdown]
# ## 6. Load Waymo Dataset
#
# **Waymo Motion Dataset structure:**
# - TFRecord format (*.tfrecord-*-of-*)
# - 11 historical steps + 80 future steps @ 10Hz
# - Total: 9.1 seconds of motion data

# %%
print("📂 Loading Waymo Motion Dataset...")

# Load dataset
dataset = WaymoDataset(
    root=CONFIG['data_root'],
    split=CONFIG['split'],
    transform=TargetBuilder(
        num_historical_steps=CONFIG['num_historical_steps'],
        num_future_steps=CONFIG['num_future_steps']
    ),
    num_historical_steps=CONFIG['num_historical_steps'],
    num_future_steps=CONFIG['num_future_steps'],
    predict_unseen_agents=False,
    vector_repr=True,
)

print(f"✅ Dataset loaded: {len(dataset)} scenarios")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=CONFIG['batch_size'],
    shuffle=True,
    num_workers=CONFIG['num_workers'],
    pin_memory=True
)

print(f"✅ DataLoader created: {len(dataloader)} batches")

# %% [markdown]
# ## 7. Load QCNet (Frozen World Model)
#
# **QCNet is FROZEN** - used as world model, not trained!

# %%
print("🔥 Loading QCNet (World Model for Waymo)...")

# Load pretrained QCNet
qcnet = QCNet.load_from_checkpoint(CONFIG['qcnet_ckpt'])
qcnet.eval()
qcnet = qcnet.to(device)

# FREEZE all parameters
for param in qcnet.parameters():
    param.requires_grad = False

print("✅ QCNet loaded and FROZEN")
print(f"   Total parameters: {sum(p.numel() for p in qcnet.parameters()):,}")
print(f"   Trainable parameters: {sum(p.numel() for p in qcnet.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 8. Test QCNet Inference

# %%
print("🔍 Testing QCNet inference...")

# Get one batch
sample_batch = next(iter(dataloader))
sample_batch = sample_batch.to(device)

# Run QCNet inference
with torch.no_grad():
    qcnet_output = qcnet(sample_batch)

print("✅ QCNet inference successful!")
print("\n📊 QCNet Output Structure:")
for key, value in qcnet_output.items():
    if isinstance(value, torch.Tensor):
        print(f"   {key}: {value.shape}")

# Trajectory predictions
loc_refine = qcnet_output['loc_refine_pos']  # (batch, num_modes=6, timesteps=80, 2)
pi = qcnet_output['pi']  # (batch, num_modes=6) - mode probabilities

print(f"\n🎯 Trajectory predictions: {loc_refine.shape}")
print(f"   - Batch size: {loc_refine.shape[0]}")
print(f"   - Num modes: {loc_refine.shape[1]}")
print(f"   - Timesteps: {loc_refine.shape[2]} (Waymo: 80 steps = 8 seconds)")
print(f"   - Dimensions: {loc_refine.shape[3]} (x, y)")

# %% [markdown]
# ## 9. Visualize QCNet Predictions

# %%
def visualize_qcnet_prediction(data, predictions, agent_idx=0):
    """Visualize QCNet predictions for one agent."""
    
    num_historical = CONFIG['num_historical_steps']
    num_future = CONFIG['num_future_steps']
    
    # Get agent history and future
    history = data['agent']['position'][agent_idx, :num_historical].cpu().numpy()
    gt_future = data['agent']['target'][agent_idx, :, :2].cpu().numpy()
    
    # Get predictions (6 modes)
    pred_trajs = predictions['loc_refine_pos'][agent_idx].cpu().numpy()  # (6, 80, 2)
    pred_probs = torch.softmax(predictions['pi'][agent_idx], dim=-1).cpu().numpy()  # (6,)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # History
    ax.plot(history[:, 0], history[:, 1], 'b-', linewidth=3, label='History (1.1s)', zorder=3)
    ax.scatter(history[-1, 0], history[-1, 1], c='blue', s=150, zorder=5, marker='o')
    
    # Ground truth
    ax.plot(gt_future[:, 0], gt_future[:, 1], 'g-', linewidth=3, label='Ground Truth (8s)', zorder=3)
    ax.scatter(gt_future[-1, 0], gt_future[-1, 1], c='green', s=150, zorder=5, marker='*')
    
    # Predictions (6 modes)
    colors = plt.cm.Reds(np.linspace(0.3, 1, 6))
    for i in range(6):
        label = f'Mode {i+1} (p={pred_probs[i]:.2f})'
        ax.plot(pred_trajs[i, :, 0], pred_trajs[i, :, 1], 
                color=colors[i], linewidth=2, alpha=0.7, label=label, zorder=2)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'QCNet Predictions (Waymo) - Agent {agent_idx}', fontsize=14, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Visualize first agent
print("📊 Visualizing QCNet predictions for agent 0...")
visualize_qcnet_prediction(sample_batch, qcnet_output, agent_idx=0)

# %% [markdown]
# ## 10. Initialize Policy Network (RL Agent)
#
# **Policy network** learns to control AVs via RL

# %%
print("🤖 Initializing Policy Network...")

# Load RL config
with open(f"configs/{CONFIG['rl_config_file']}", 'r') as f:
    rl_config = yaml.safe_load(f)

# Add custom parameters
rl_config['cost_quantile'] = CONFIG['cost_quantile']
rl_config['distance_limit'] = CONFIG['distance_limit']
rl_config['LR_ACTOR'] = CONFIG['learning_rate_actor']
rl_config['LR_CRITIC'] = CONFIG['learning_rate_critic']
rl_config['GAMMA'] = CONFIG['gamma']

print(f"📋 RL Config loaded: {CONFIG['rl_config_file']}")
print(f"   Algorithm: {rl_config.get('algorithm', CONFIG['rl_algorithm'])}")
print(f"   Learning rate (actor): {rl_config['LR_ACTOR']}")
print(f"   Learning rate (critic): {rl_config['LR_CRITIC']}")

# Determine state dimension
sample_agent_state = sample_batch['agent']['position'][0, -1]
state_dim = sample_agent_state.shape[0] * 5  # Simplified estimate

num_agents = len(CONFIG['controlled_agents'])

# Initialize RL agent
if CONFIG['rl_algorithm'] == 'TrafficGamer':
    policy_network = TrafficGamer(
        state_dim=state_dim,
        agent_number=num_agents,
        config=rl_config,
        device=device
    )
elif CONFIG['rl_algorithm'] == 'CCE_MAPPO':
    policy_network = Constrainted_CCE_MAPPO(
        state_dim=state_dim,
        agent_number=num_agents,
        config=rl_config,
        device=device
    )
else:  # MAPPO
    policy_network = MAPPO(
        state_dim=state_dim,
        agent_number=num_agents,
        config=rl_config,
        device=device
    )

print(f"✅ Policy network initialized: {CONFIG['rl_algorithm']}")
print(f"   State dim: {state_dim}")
print(f"   Num controlled agents: {num_agents}")
print(f"   Trainable parameters: {sum(p.numel() for p in policy_network.parameters() if p.requires_grad):,}")

# %% [markdown]
# ## 11. Define Training Loop

# %%
def extract_state_from_batch(batch, agent_indices: List[int]):
    """Extract state representation for controlled agents."""
    states = []
    
    for agent_idx in agent_indices:
        # Agent's own state (last timestep)
        position = batch['agent']['position'][agent_idx, -1]
        velocity = batch['agent']['velocity'][agent_idx, -1] if 'velocity' in batch['agent'] else torch.zeros(2, device=device)
        heading = batch['agent']['heading'][agent_idx, -1] if 'heading' in batch['agent'] else torch.tensor(0.0, device=device)
        
        # Concatenate features
        agent_state = torch.cat([position, velocity, heading.unsqueeze(0)])
        states.append(agent_state)
    
    states = torch.stack(states)
    return states


def compute_reward(batch, actions, qcnet_predictions):
    """Compute reward for RL training."""
    rewards = []
    costs = []
    
    for agent_idx in CONFIG['controlled_agents']:
        # Ground truth trajectory
        gt_future = batch['agent']['target'][agent_idx, :, :2]
        
        # Prediction (best mode)
        pred_traj = qcnet_predictions['loc_refine_pos'][agent_idx, 0]
        
        # Reward 1: Progress towards goal
        goal_pos = gt_future[-1]
        current_pos = batch['agent']['position'][agent_idx, -1, :2]
        distance_to_goal = torch.norm(goal_pos - current_pos)
        progress_reward = -distance_to_goal.item()
        
        # Reward 2: Smooth driving
        action = actions[CONFIG['controlled_agents'].index(agent_idx)]
        smoothness_penalty = -0.1 * torch.norm(action).item()
        
        # Cost: Collision risk
        collision_cost = 0.0
        for other_idx in range(batch['agent']['position'].shape[0]):
            if other_idx == agent_idx:
                continue
            other_pos = batch['agent']['position'][other_idx, -1, :2]
            distance = torch.norm(current_pos - other_pos)
            if distance < CONFIG['distance_limit']:
                collision_cost += 1.0
        
        total_reward = progress_reward + smoothness_penalty
        rewards.append(total_reward)
        costs.append(collision_cost)
    
    return torch.tensor(rewards, device=device), torch.tensor(costs, device=device)


def train_one_epoch(policy_network, qcnet, dataloader, epoch):
    """Train policy network for one epoch."""
    
    policy_network.train()
    
    epoch_rewards = []
    epoch_costs = []
    epoch_losses = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['max_epochs']}")
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        # Step 1: QCNet predicts (frozen)
        with torch.no_grad():
            qcnet_predictions = qcnet(batch)
        
        # Step 2: Extract state
        states = extract_state_from_batch(batch, CONFIG['controlled_agents'])
        
        # Step 3: Policy selects actions
        actions, log_probs, values = [], [], []
        
        for agent_state in states:
            action_mean = policy_network.actor(agent_state.unsqueeze(0))
            action = action_mean + 0.1 * torch.randn_like(action_mean)
            
            value = policy_network.critic(agent_state.unsqueeze(0))
            
            actions.append(action.squeeze(0))
            values.append(value)
        
        actions = torch.stack(actions)
        values = torch.stack(values)
        
        # Step 4: Compute rewards
        rewards, costs = compute_reward(batch, actions, qcnet_predictions)
        
        # Step 5: PPO update
        advantages = rewards - values.squeeze(-1).detach()
        
        actor_loss = -(advantages * log_probs).mean() if log_probs else torch.tensor(0.0, device=device)
        critic_loss = nn.MSELoss()(values.squeeze(-1), rewards)
        cost_loss = costs.mean() if CONFIG['rl_algorithm'] == 'TrafficGamer' else torch.tensor(0.0, device=device)
        
        total_loss = actor_loss + 0.5 * critic_loss + CONFIG['penalty_initial_value'] * cost_loss
        
        # Optimize
        policy_network.actor_optimizer.zero_grad()
        policy_network.critic_optimizer.zero_grad()
        total_loss.backward()
        policy_network.actor_optimizer.step()
        policy_network.critic_optimizer.step()
        
        # Log
        epoch_rewards.append(rewards.mean().item())
        epoch_costs.append(costs.mean().item())
        epoch_losses.append(total_loss.item())
        
        pbar.set_postfix({
            'reward': f"{np.mean(epoch_rewards):.2f}",
            'cost': f"{np.mean(epoch_costs):.2f}",
            'loss': f"{np.mean(epoch_losses):.3f}"
        })
    
    return {
        'avg_reward': np.mean(epoch_rewards),
        'avg_cost': np.mean(epoch_costs),
        'avg_loss': np.mean(epoch_losses)
    }

print("✅ Training functions defined")

# %% [markdown]
# ## 12. Train Policy Network

# %%
print("🚀 Starting Policy Network Training...")
print(f"   Epochs: {CONFIG['max_epochs']}")
print(f"   Controlled agents: {CONFIG['controlled_agents']}")
print(f"   RL Algorithm: {CONFIG['rl_algorithm']}")

training_history = {
    'rewards': [],
    'costs': [],
    'losses': []
}

for epoch in range(CONFIG['max_epochs']):
    metrics = train_one_epoch(policy_network, qcnet, dataloader, epoch)
    
    training_history['rewards'].append(metrics['avg_reward'])
    training_history['costs'].append(metrics['avg_cost'])
    training_history['losses'].append(metrics['avg_loss'])
    
    print(f"\n📊 Epoch {epoch+1} Summary:")
    print(f"   Avg Reward: {metrics['avg_reward']:.3f}")
    print(f"   Avg Cost: {metrics['avg_cost']:.3f}")
    print(f"   Avg Loss: {metrics['avg_loss']:.3f}")
    
    # Save checkpoint
    if CONFIG['save_checkpoint'] and (epoch + 1) % CONFIG['eval_freq'] == 0:
        checkpoint_path = f"checkpoints/policy_waymo_epoch_{epoch+1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'policy_state_dict': policy_network.state_dict(),
            'config': CONFIG,
            'training_history': training_history
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

print("\n✅ Training completed!")

# %% [markdown]
# ## 13. Visualize Training Progress

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Rewards
axes[0].plot(training_history['rewards'], linewidth=2, color='green')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Avg Reward')
axes[0].set_title('Training Rewards', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Costs
axes[1].plot(training_history['costs'], linewidth=2, color='red')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Avg Cost')
axes[1].set_title('Safety Costs', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Losses
axes[2].plot(training_history['losses'], linewidth=2, color='blue')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss')
axes[2].set_title('Training Loss', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Training curves plotted")

# %% [markdown]
# ## 14. Evaluate Trained Policy

# %%
def evaluate_policy(policy_network, qcnet, dataloader, num_samples=10):
    """Evaluate trained policy."""
    
    policy_network.eval()
    qcnet.eval()
    
    eval_rewards = []
    eval_costs = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if batch_idx >= num_samples:
                break
            
            batch = batch.to(device)
            
            qcnet_predictions = qcnet(batch)
            states = extract_state_from_batch(batch, CONFIG['controlled_agents'])
            
            actions = []
            for agent_state in states:
                action = policy_network.actor(agent_state.unsqueeze(0))
                actions.append(action.squeeze(0))
            
            actions = torch.stack(actions)
            rewards, costs = compute_reward(batch, actions, qcnet_predictions)
            
            eval_rewards.append(rewards.mean().item())
            eval_costs.append(costs.mean().item())
    
    return {
        'avg_reward': np.mean(eval_rewards),
        'avg_cost': np.mean(eval_costs),
        'std_reward': np.std(eval_rewards),
        'std_cost': np.std(eval_costs)
    }

print("🔍 Evaluating trained policy...")
eval_metrics = evaluate_policy(policy_network, qcnet, dataloader, num_samples=20)

print("\n" + "="*50)
print("📊 EVALUATION RESULTS (Waymo)")
print("="*50)
print(f"Avg Reward: {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}")
print(f"Avg Cost:   {eval_metrics['avg_cost']:.3f} ± {eval_metrics['std_cost']:.3f}")
print("="*50)

# %% [markdown]
# ## 15. Save Final Model

# %%
final_checkpoint = {
    'config': CONFIG,
    'rl_config': rl_config,
    'policy_state_dict': policy_network.state_dict(),
    'training_history': training_history,
    'final_metrics': eval_metrics
}

save_path = "checkpoints/trafficgamer_waymo_final.pt"
os.makedirs("checkpoints", exist_ok=True)
torch.save(final_checkpoint, save_path)

print(f"✅ Final model saved: {save_path}")

# %% [markdown]
# ## 16. Summary
#
# **What we did:**
# 1. ✅ Loaded Waymo Motion Dataset (TFRecord format)
# 2. ✅ Loaded QCNet (frozen world model for Waymo)
# 3. ✅ Initialized Policy Network (RL agent)
# 4. ✅ Trained policy to control AVs
# 5. ✅ Evaluated on validation set
# 6. ✅ Visualized results
#
# **Key differences from Argoverse 2:**
# - **Dataset format:** TFRecord (Waymo) vs Parquet (Argoverse 2)
# - **Timeline:** 11 history + 80 future (Waymo) vs 50 history + 60 future (Argoverse 2)
# - **Duration:** 9.1 seconds total (Waymo) vs 11 seconds total (Argoverse 2)
# - **Simulator:** Waymax for Waymo (optional for advanced training)
#
# **Next steps:**
# - Integrate Waymax simulator for more realistic training
# - Tune hyperparameters
# - Try different RL algorithms
# - Add more sophisticated reward shaping

# %% [markdown]
# ## Done! 🎉
