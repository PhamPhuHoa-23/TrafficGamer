# %% [markdown]
# # Waymax + Waymo Motion Dataset - Interactive Training
#
# **Interactive notebook** to train policy with Waymax simulator:
# - **NO DOWNLOAD**: Data streams from Google Cloud
# - **NO QCNet**: Focus on Waymax simulator + RL
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
!pip install -q av  # Video encoding (needed by source imports)
!pip install -q git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax  # Waymax simulator
!pip install -q waymo-open-dataset-tf-2-12-0==1.6.4  # Waymo dataset utilities

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
from transforms import TargetBuilder
from torch_geometric.data import Batch

# Import RL algorithms
from algorithm.TrafficGamer import TrafficGamer
from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO
from algorithm.mappo import MAPPO

# Import utilities
from utils.rollout import PPO_process_batch
from utils.utils import seed_everything

print("✅ TrafficGamer modules imported successfully")

# %% [markdown]
# ## 5. Configuration
#
# **🔧 EDIT HERE - Configure paths and parameters:**

# %%
# ============================================
# 🔧 CONFIGURATION - EDIT HERE
# ============================================

CONFIG = {
    # Data split
    'split': 'val',  # 'train', 'val', or 'test'
    
    # Dataset settings (Waymo: 11 history + 80 future @ 10Hz)
    'num_historical_steps': 11,
    'num_future_steps': 80,
    
    # Training settings
    'seed': 42,
    'batch_size': 4,
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
print(f"   Split: {CONFIG['split']}")
print(f"   RL Algorithm: {CONFIG['rl_algorithm']}")
print(f"   History/Future: {CONFIG['num_historical_steps']}/{CONFIG['num_future_steps']}")

# %% [markdown]
# ## 6. Apply for Waymo Dataset Access (REQUIRED FIRST!)
#
# **⚠️ IMPORTANT: You MUST apply for dataset access BEFORE running this notebook!**
#
# ### 📝 Steps to get access:
#
# 1. **Apply for Waymo Open Dataset access:**
#    - Go to: https://waymo.com/open/download/
#    - Click "Register" and fill out the form
#    - Agree to Terms of Service
#    - Use your email (or service account email if using Kaggle)
#
# 2. **Wait for approval:**
#    - ⏱️ Usually takes a few hours to 1-2 days
#    - 📧 You'll receive email confirmation when approved
#
# 3. **Then proceed with authentication below**
#
# ### 📚 Official documentation:
# - Waymax: https://github.com/waymo-research/waymax#configure-access-to-waymo-open-motion-dataset
# - TFDS GCS: https://www.tensorflow.org/datasets/gcs

# %% [markdown]
# ## 7. Authenticate with Google Cloud
#
# **After getting approved, authenticate to access dataset from GCS**
# 
# Choose ONE method:
# - **Colab**: Use `auth.authenticate_user()` (easiest)
# - **Local/VM**: Use `gcloud auth application-default login`
# - **Kaggle**: Use Service Account Key

# %%
print("🔑 Authenticating with Google Cloud...")

# ============================================
# METHOD 1: Colab Authentication (EASIEST)
# ============================================
try:
    from google.colab import auth
    auth.authenticate_user()
    print("✅ Authenticated via Colab")
    print("   Make sure your Google account has Waymo dataset access!")
except ImportError:
    print("⚠️  Not in Colab, trying alternative methods...")
    
    # ============================================
    # METHOD 2: Service Account Key (Kaggle/Local)
    # ============================================
    # For Kaggle or local environments
    # Steps:
    # 1. Apply for Waymo access with service account email
    # 2. Download JSON key from: https://console.cloud.google.com/iam-admin/serviceaccounts
    # 3. Upload to Kaggle as dataset or save locally
    
    service_key_path = '/kaggle/input/gcs-credentials/service-account-key.json'  # Change this path
    
    if os.path.exists(service_key_path):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
        print(f"✅ Authenticated via service account: {service_key_path}")
        print("   Make sure this service account has Waymo dataset access!")
    else:
        print("❌ No authentication found!")
        print("\n📝 To authenticate:")
        print("   METHOD A (Local/VM): Run in terminal:")
        print("      gcloud auth application-default login")
        print("   METHOD B (Colab): Use auth.authenticate_user() above")
        print("   METHOD C (Kaggle): Upload service account JSON key")
        print("\n⚠️  Remember: You must apply for Waymo access first!")
        print("   Apply at: https://waymo.com/open/download/")
        raise RuntimeError("Authentication required to access Waymo dataset from GCS")

# %% [markdown]
# ## 8. Load Waymo Dataset with Waymax
#
# **Waymax streams data from Google Cloud - NO download needed!**
# - Data streams directly from cloud via GCS bucket: `gs://waymo_open_dataset_motion_v_1_2_0/`
# - 11 historical steps + 80 future steps @ 10Hz (9.1s total)
# - Requires Waymo dataset access (see Section 6)

# %%
print("📂 Loading Waymo Motion Dataset with Waymax...")

# Import Waymax
from waymax import config as waymax_config
from waymax import dataloader as waymax_dataloader

# Stream data from cloud
if CONFIG['split'] == 'train':
    dataset_config = waymax_config.WOD_1_1_0_TRAINING
elif CONFIG['split'] == 'val':
    dataset_config = waymax_config.WOD_1_1_0_VALIDATION
else:
    dataset_config = waymax_config.WOD_1_1_0_TESTING

# Configure
import dataclasses
dataset_config = dataclasses.replace(
    dataset_config,
    max_num_objects=32,  # Limit objects for memory
    batch_dims=(CONFIG['batch_size'],),
)

# Create generator
waymo_iterator = waymax_dataloader.simulator_state_generator(dataset_config)

print(f"✅ Waymax iterator created for {CONFIG['split']} split")
print(f"   Data streams from: {dataset_config.path}")
print(f"   Batch size: {CONFIG['batch_size']}")

# %% [markdown]
# ## 9. Test Waymax Data Loading

# %%
print("🔍 Testing Waymax data loading...")

# Recreate iterator (in case it was consumed)
waymo_iterator = waymax_dataloader.simulator_state_generator(dataset_config)

try:
    # Get one scenario
    sample_scenario = next(waymo_iterator)
    
    print("✅ Scenario loaded successfully!")
    print("\n📊 Scenario Structure:")
    print(f"   Timesteps: {sample_scenario.num_timesteps}")
    print(f"   Objects: {sample_scenario.num_objects}")
    print(f"   Valid objects: {sample_scenario.object_metadata.is_valid.sum()}")
    
    # Print trajectory shape
    print(f"\n🚗 Trajectory data:")
    print(f"   Position: {sample_scenario.log_trajectory.xy.shape}")
    print(f"   Velocity: {sample_scenario.log_trajectory.vel_x.shape}")
    print(f"   Heading: {sample_scenario.log_trajectory.yaw.shape}")
    
except StopIteration:
    print("\n" + "="*60)
    print("❌ NO DATA FOUND - ACCESS DENIED")
    print("="*60)
    print("\n🔒 Your account doesn't have Waymo dataset access yet!")
    print("\n📝 TO FIX THIS:")
    print("   1. Apply for access: https://waymo.com/open/download/")
    print("   2. Use the email you authenticated with (check above)")
    print("   3. Wait for approval email (usually few hours to 1 day)")
    print("   4. Re-run this notebook after approval")
    print("\n💡 Alternative: Use Colab with a Google account that already has access")
    print("="*60)
    raise
except Exception as e:
    print(f"\n❌ Error loading data: {e}")
    print("\n🔍 Common issues:")
    print("   - PermissionDeniedError: Need Waymo dataset access")
    print("   - Apply at: https://waymo.com/open/download/")
    raise

# %% [markdown]
# ## 9. Visualize Waymo Scenario (Optional)

# %%
# Visualize scenario with Waymax
from waymax import visualization as waymax_viz

try:
    img = waymax_viz.plot_simulator_state(sample_scenario, use_log_traj=True)
    
    # Display
    try:
        import mediapy
        mediapy.show_image(img)
    except ImportError:
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Waymo Scenario Visualization', fontsize=14, fontweight='bold')
        plt.show()
    
    print("✅ Scenario visualization complete")
except Exception as e:
    print(f"⚠️  Visualization skipped: {e}")

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
# Extract from Waymax scenario
sample_agent_pos = sample_scenario.log_trajectory.xy[0, -1]  # (2,)
state_dim = 10  # Simplified: position(2) + velocity(2) + heading(1) + neighbors(5)

num_agents = len(CONFIG['controlled_agents'])

# Initialize RL agent
if CONFIG['rl_algorithm'] == 'TrafficGamer':
    policy_network = TrafficGamer(
      markdown]
# ## 11. Define Training Loop (Waymax Integration)ts,
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

# %% [0. Define Training Loop (Waymax Integration)
# ## 11. Define Training Loop

# %%waymax_scenario(scenario, agent_indices: List[int]):
    """Extract state representation from Waymax scenario."""
    import jax.numpy as jnp
    
    states = []
    
    for agent_idx in agent_indices:
        # Agent's own state (last timestep)
        position = scenario.log_trajectory.xy[agent_idx, -1]  # (2,)
        velocity = jnp.array([
            scenario.log_trajectory.vel_x[agent_idx, -1],
            scenario.log_trajectory.vel_y[agent_idx, -1]
        ])  # (2,)
        heading = scenario.log_trajectory.yaw[agent_idx, -1]  # scalar
        
        # Convert JAX to PyTorch
        agent_state = np.concatenate([
            np.array(position),
            np.array(velocity),
            np.arrascenario, actions):
    """Compute reward for RL training from Waymax scenario."""
    import jax.numpy as jnp
    
    rewards = []
    costs = []
    
    for agent_idx in CONFIG['controlled_agents']:
        # Ground truth trajectory (future)
        gt_future = scenario.log_trajectory.xy[agent_idx, -CONFIG['num_future_steps']:]
        goal_pos = gt_future[-1]
        
        # Current position
        current_pos = scenario.log_trajectory.xy[agent_idx, -1]
        distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(current_pos))
        progress_reward = -distance_to_goal
        
        # Smoothness penalty
        action = actions[CONFIG['controlled_agents'].index(agent_idx)]
        smoothness_penalty = -0.1 * torch.norm(action).item()
        
        # Collision cost
        collision_cost = 0.0
        for other_idx in range(scenario.num_objects):
            if other_idx == agent_idx:
                continue
            if not scenario.object_metadata.is_valid[other_idx]:
                continue
            other_pos = scenario.log_trajectory.xy[other_idx, -1]
            distance = np.linalg.norm(np.array(current_pos) - np.array(other_pos)
        action = actions[CONFIG['controlled_agents'].index(agent_idx)]
        smoothness_penalty = -0.1 * torch.norm(action).item()
        
        # Cost: Collision risk
        collision_cost = 0.0
        for other_idx in range(batch['agent']['position'].shape[0]):
            if other_idx == agent_idx:
                continue
            other_pos = batch['agent']['position'][other_idx, -1, :2]
            distance = torch.norm(cuwaymo_iterator, epoch, num_batches=100):
    """Train policy network for one epoch with Waymax."""
    
    policy_network.train()
    
    epoch_rewards = []
    epoch_costs = []
    epoch_losses = []
    
    pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{CONFIG['max_epochs']}")
    
    for batch_idx in pbar:
        # Step 1: Get scenario from Waymax
        scenario = next(waymo_iterator)
        
        # Step 2: Extract state
        states = extract_state_from_waymax_scenario(scenario, CONFIG['controlled_agents'])
        
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
        rewards, costs = compute_reward(scenario, aan)
            
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

# %%1. Train Policy Network

# %%
print("🚀 Starting Policy Network Training with Waymax...")
print(f"   Epochs: {CONFIG['max_epochs']}")
print(f"   Controlled agents: {CONFIG['controlled_agents']}")
print(f"   RL Algorithm: {CONFIG['rl_algorithm']}")

training_history = {
    'rewards': [],
    'costs': [],
    'losses': []
}

for epoch in range(CONFIG['max_epochs']):
    metrics = train_one_epoch(policy_network, waymo_iterator, epoch, num_batches=100
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

# %%2
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
def ev3luate_policy(policy_network, waymo_iterator, num_samples=20):
    """Evaluate trained policy on Waymax scenarios."""
    
    policy_network.eval()
    
    eval_rewards = []
    eval_costs = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Evaluating"):
            scenario = next(waymo_iterator)
            
            states = extract_state_from_waymax_scenario(scenario, CONFIG['controlled_agents'])
            
            actions = []
            for agent_state in states:
                action = policy_network.actor(agent_state.unsqueeze(0))
                actions.append(action.squeeze(0))
            
            actions = torch.stack(actions)
            rewards, costs = compute_reward(scenario, actions)
            
            eval_rewards.append(rewards.mean().item())
            eval_costs.append(costs.mean().item())
    
    return {
        'avg_reward': np.mean(eval_rewards),
        'avg_cost': np.mean(eval_costs),
        'std_reward': np.std(eval_rewards),
        'std_cost': np.std(eval_costs)
    }

print("🔍 Evaluating trained policy...")
eval_metrics = evaluate_policy(policy_network, waymo_iterator, num_samples=20)

print("\n" + "="*50)
print("📊 EVALUATION RESULTS (Waymo + Waymax)")
print("="*50)
print(f"Avg Reward: {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}")
print(f"Avg Cost:   {eval_metrics['avg_cost']:.3f} ± {eval_metrics['std_cost']:.3f}")
print("="*50)

# %% [markdown]
# ## 15. Save Final Model

# %%
final_checkpoint = {
    'config': CONFIG,
    'r4_config': rl_config,
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
# 3. ✅5. Summary
#
# **What we did:**
# 1. ✅ Streamed Waymo Motion Dataset from Google Cloud (NO download!)
# 2. ✅ Used Waymax simulator (JAX-based)
# 3. ✅ Initialized Policy Network (RL agent)
# 4. ✅ Trained policy to control AVs
# 5. ✅ Evaluated on validation set
# 6. ✅ Visualized results
#
# **Key features:**
# - **NO QCNet dependency** - pure Waymax + RL
# - **Cloud streaming** - authenticate with `gcloud auth login`
# - **Timeline:** 11 history + 80 future (Waymo) = 9.1 seconds
# - **TF 2.12** - matches TrafficGamer source (not 2.11)
# - **av package** - included for source imports
#
# **Next steps:**
# - Integrate Waymax dynamics for realistic simulation
# - Add Waymax metrics (overlap, offroad, etc.)
