# %% [markdown]
# # 🖱️ Cursor Trajectory — Fast CVAE Pipeline
#
# **Pipeline: Raw Data → Segmentation (64 points) → CVAE Model**
#
# This replaces the SIREN/VQVAE pipeline with a lightweight Conditional VAE.
# Training takes < 3 minutes and uses < 2 GB RAM.
# All checkpoints saved to HuggingFace Hub automatically.

# %% [markdown]
# ## Cell 1: Setup

# %%
import os, subprocess, sys

if not os.path.exists('/content/cursor-trajectory'):
    subprocess.run(['git', 'clone', 'https://github.com/Ramcharan747/cursor-trajectory.git'], check=True)
os.chdir('/content/cursor-trajectory')
sys.path.insert(0, '/content/cursor-trajectory')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'numpy', 'matplotlib', 'tqdm', 'huggingface_hub', 'scipy'], check=True)

from huggingface_hub import HfApi, login, upload_file as _hf_upload
from google.colab import userdata
import torch, numpy as np
from scipy.interpolate import interp1d

HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN)
REPO_ID = "Baka7/cursor-trajectory-checkpoints"

# Safe upload wrapper
def upload_file(**kwargs):
    kwargs['token'] = HF_TOKEN
    try:
        _hf_upload(**kwargs)
    except Exception as e:
        print(f"⚠️ Upload skipped: {e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
print("✅ Setup complete")

# %% [markdown]
# ## Cell 2: Load & Segment Raw Data (Resampled to 64 Points)

# %%
import gzip, json, glob
from tqdm import tqdm
from trajectory_gen.data.segmentation import segment_trajectory

data_dir = 'data/raw'
x_all, y_all, t_all = [], [], []

files = sorted(glob.glob(f'{data_dir}/cursor_*.jsonl.gz')) + sorted(glob.glob(f'{data_dir}/cursor_*.jsonl'))
print(f"Loading {len(files)} files...")

for f in tqdm(files, desc="Loading"):
    opener = gzip.open if f.endswith('.gz') else open
    with opener(f, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            try:
                e = json.loads(line)
                x_all.append(float(e['x']))
                y_all.append(float(e['y']))
                t_all.append(int(e['t']))
            except: continue

x_all = np.array(x_all, dtype=np.float64)
y_all = np.array(y_all, dtype=np.float64)
t_all = np.array(t_all, dtype=np.int64)

order = np.argsort(t_all)
x_all, y_all, t_all = x_all[order], y_all[order], t_all[order]
print(f"✅ Loaded {len(x_all):,} events")

# Segment boundaries (idle > 2.0s)
dt = np.diff(t_all) / 1e6
split_indices = np.where(dt > 2.0)[0] + 1
boundaries = [0] + split_indices.tolist() + [len(x_all)]

all_segments = []
for i in tqdm(range(len(boundaries) - 1), desc="Segmenting"):
    s, e = boundaries[i], boundaries[i + 1]
    if e - s < 10: continue
    segs = segment_trajectory(x_all[s:e], y_all[s:e], t_all[s:e],
        min_points=15, min_duration_s=0.05, max_duration_s=3.0)
    all_segments.extend(segs)

# Resample every segment to EXACTLY 64 points using spatial/temporal interpolation
SEQ_LEN = 64
resampled_data = []
conditions = []

for s in tqdm(all_segments, desc="Resampling to 64 points"):
    if len(s.x) < 5: continue
    # Normalize time to [0, 1]
    t_norm = (s.t - s.t[0]) / max((s.t[-1] - s.t[0]), 1e-6)
    
    # Interpolate
    fx = interp1d(t_norm, s.x, kind='linear')
    fy = interp1d(t_norm, s.y, kind='linear')
    
    t_new = np.linspace(0, 1, SEQ_LEN)
    x_new = fx(t_new)
    y_new = fy(t_new)
    
    # Condition: displacement (dx, dy)
    dx = x_new[-1] - x_new[0]
    dy = y_new[-1] - y_new[0]
    
    # Shift trajectory to start at (0,0) for the model to learn translation invariance
    x_new_shifted = x_new - x_new[0]
    y_new_shifted = y_new - y_new[0]
    
    trajectory = np.stack([x_new_shifted, y_new_shifted], axis=-1) # (64, 2)
    
    resampled_data.append(trajectory)
    conditions.append([dx, dy])

resampled_data = np.array(resampled_data, dtype=np.float32)
conditions = np.array(conditions, dtype=np.float32)

print(f"✅ Created {len(resampled_data):,} fixed-length trajectories")
print(f"💾 Dataset size in RAM: {resampled_data.nbytes / 1e6:.1f} MB (was 4.2 GB in old pipeline!)")

np.save('data/cvae_trajectories.npy', resampled_data)
np.save('data/cvae_conditions.npy', conditions)

# Free raw data
del x_all, y_all, t_all, all_segments
import gc; gc.collect()

# %% [markdown]
# ## Cell 3: Train Diffusion Model

# %%
import time
from torch.utils.data import DataLoader, TensorDataset
from trajectory_gen.models.diffusion import TrajectoryDiffusion

resampled_data = np.load('data/cvae_trajectories.npy')
conditions = np.load('data/cvae_conditions.npy')

# Normalize the input data to roughly [-1, 1] scale for stable CNN training
SCALE_FACTOR = 2000.0

# Transpose x from (B, 64, 2) to (B, 2, 64) for Conv1d
resampled_data_t = np.transpose(resampled_data, (0, 2, 1))

x_tensor = torch.tensor(resampled_data_t / SCALE_FACTOR, dtype=torch.float32)
c_tensor = torch.tensor(conditions / SCALE_FACTOR, dtype=torch.float32)

dataset = TensorDataset(x_tensor, c_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

model = TrajectoryDiffusion(seq_len=64, channels=2, cond_dim=2, timesteps=100).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

NUM_EPOCHS = 100
best_loss = float('inf')
t0 = time.time()

print(f"Training Diffusion Model ({sum(p.numel() for p in model.parameters()):,} params)...")

for epoch in range(NUM_EPOCHS):
    model.train()
    ep_loss, nb = 0, 0
    
    for x_batch, c_batch in loader:
        x_batch, c_batch = x_batch.to(device), c_batch.to(device)
        
        optimizer.zero_grad()
        loss = model.compute_loss(x_batch, c_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        ep_loss += loss.item()
        nb += 1
        
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = ep_loss / nb
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | MSE Noise Loss: {avg_loss:.5f} | Time: {(time.time()-t0)/60:.1f}min")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': {'seq_len': 64, 'channels': 2, 'cond_dim': 2, 'timesteps': 100, 'scale_factor': SCALE_FACTOR}
            }, 'diffusion_checkpoint.pt')

upload_file(path_or_fileobj='diffusion_checkpoint.pt', path_in_repo='diffusion_checkpoint.pt', repo_id=REPO_ID)
print(f"✅ Training complete in {(time.time()-t0)/60:.1f} minutes!")

# %% [markdown]
# ## Cell 4: Generate and Visualize Trajectories

# %%
import matplotlib.pyplot as plt

model.eval()
routes = [
    ((100, 200), (800, 600), "Short diagonal"),
    ((50, 500), (1800, 100), "Long diagonal"),
    ((900, 50), (900, 900), "Vertical"),
    ((200, 200), (400, 250), "Micro movement")
]

fig, axes = plt.subplots(4, 4, figsize=(20, 16))

with torch.no_grad():
    for row, (start, end, label) in enumerate(routes):
        # Calculate condition
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        c = torch.tensor([[dx, dy]], dtype=torch.float32, device=device) / SCALE_FACTOR
        
        for col in range(4):
            # Generate trajectory using 100-step DDPM reverse process
            gen_traj_scaled = model.sample(c, batch_size=1) # (1, 2, 64)
            
            # Transpose back to (64, 2) and scale
            gen_traj = gen_traj_scaled.transpose(1, 2).cpu().numpy()[0] * SCALE_FACTOR
            
            # The model predicts the trajectory relative to (0,0), but the endpoints might not match exactly.
            # We can force the generated path to perfectly hit the endpoint using a linear offset correction.
            actual_dx = gen_traj[-1, 0] - gen_traj[0, 0]
            actual_dy = gen_traj[-1, 1] - gen_traj[0, 1]
            error_x = dx - actual_dx
            error_y = dy - actual_dy
            
            # Apply correction linearly over the sequence
            alpha = np.linspace(0, 1, 64)
            gen_traj[:, 0] += alpha * error_x
            gen_traj[:, 1] += alpha * error_y
            
            # Shift to start point
            gen_traj[:, 0] = gen_traj[:, 0] - gen_traj[0, 0] + start[0]
            gen_traj[:, 1] = gen_traj[:, 1] - gen_traj[0, 1] + start[1]
            
            axes[row, col].plot(gen_traj[:, 0], gen_traj[:, 1], '-', lw=2, color='royalblue')
            axes[row, col].plot(*start, 'go', ms=10, label='Start')
            axes[row, col].plot(*end, 'rs', ms=10, label='End')
            
            if col == 0: axes[row, col].set_ylabel(label, fontsize=12)
            axes[row, col].set_title(f'Sample #{col+1}')
            axes[row, col].set_aspect('equal', adjustable='datalim')
            axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Diffusion Generated Point-to-Point Cursor Trajectories', fontsize=16)
plt.tight_layout()
plt.savefig('diffusion_samples.png', dpi=150, bbox_inches='tight')
plt.show()

upload_file(path_or_fileobj='diffusion_samples.png', path_in_repo='diffusion_samples.png', repo_id=REPO_ID)
print("🎉 All done! Generated trajectories saved.")
