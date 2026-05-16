# %% [markdown]
# # 🖱️ Cursor Trajectory — Full Training Pipeline
# 
# **Pipeline: Raw Data → Segmentation → SIREN → VQ-VAE → Latent ODE**
# 
# All checkpoints saved to HuggingFace Hub automatically.
# Designed for Colab T4 GPU (16GB VRAM).

# %% [markdown]
# ## Cell 1: Setup — Clone repo, install deps, login to HF

# %%
import os, subprocess, sys

# Clone repo
if not os.path.exists('cursor-trajectory'):
    subprocess.run(['git', 'clone', 'https://github.com/Ramcharan747/cursor-trajectory.git'], check=True)
os.chdir('cursor-trajectory')

# Install dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'numpy', 'matplotlib', 'tqdm', 'torchdiffeq', 'huggingface_hub'], check=True)

# HuggingFace login
from huggingface_hub import HfApi, login
from google.colab import userdata
HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN)
api = HfApi()

# Create HF repo for checkpoints
REPO_ID = "Ramcharan747/cursor-trajectory-checkpoints"
try:
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    print(f"✅ HF repo ready: https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"HF repo: {e}")

# Verify GPU
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("⚠️ No GPU — training will be very slow!")

print("✅ Setup complete")

# %% [markdown]
# ## Cell 2: Load & Segment Raw Data

# %%
import gzip, json, glob
import numpy as np
from tqdm import tqdm

# Load all recordings
data_dir = 'data/raw'
x_all, y_all, t_all = [], [], []

files = sorted(glob.glob(f'{data_dir}/cursor_*.jsonl.gz')) + sorted(glob.glob(f'{data_dir}/cursor_*.jsonl'))
print(f"Loading {len(files)} recording files...")

for f in tqdm(files, desc="Loading"):
    opener = gzip.open if f.endswith('.gz') else open
    with opener(f, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                x_all.append(float(e['x']))
                y_all.append(float(e['y']))
                t_all.append(int(e['t']))
            except:
                continue

x_all = np.array(x_all, dtype=np.float64)
y_all = np.array(y_all, dtype=np.float64)
t_all = np.array(t_all, dtype=np.int64)

# Sort by time
order = np.argsort(t_all)
x_all, y_all, t_all = x_all[order], y_all[order], t_all[order]
print(f"✅ Loaded {len(x_all):,} events")

# %%
# Split at idle periods (>2s gap) and segment
from trajectory_gen.data.segmentation import segment_trajectory, Segment

# Split at idle gaps
dt = np.diff(t_all) / 1e6  # microseconds → seconds
idle_mask = dt > 2.0
split_indices = np.where(idle_mask)[0] + 1
boundaries = [0] + split_indices.tolist() + [len(x_all)]

print(f"Found {len(boundaries)-1} active periods")

# Segment each active period
all_segments = []
for i in tqdm(range(len(boundaries) - 1), desc="Segmenting"):
    s, e = boundaries[i], boundaries[i + 1]
    if e - s < 10:
        continue
    segs = segment_trajectory(
        x_all[s:e], y_all[s:e], t_all[s:e],
        min_points=15, min_duration_s=0.05, max_duration_s=3.0,
    )
    all_segments.extend(segs)

print(f"✅ {len(all_segments):,} segments extracted")
print(f"   Avg duration: {np.mean([s.duration_s for s in all_segments]):.2f}s")
print(f"   Avg points:   {np.mean([s.num_points for s in all_segments]):.0f}")

# Save segments to disk
np.save('data/segments.npy', [{'t': s.t, 'x': s.x, 'y': s.y} for s in all_segments], allow_pickle=True)
print(f"💾 Saved segments to data/segments.npy")

# %% [markdown]
# ## Cell 3: Fit SIRENs to All Segments (Phase 1)

# %%
import torch
from trajectory_gen.models.siren import SIREN, SIRENFitter
from huggingface_hub import upload_file
import time

segments_data = np.load('data/segments.npy', allow_pickle=True)
print(f"Fitting SIRENs to {len(segments_data):,} segments...")

fitter = SIRENFitter(
    hidden_features=64,
    hidden_layers=3,
    omega_0=30.0,
    lr=1e-4,
    num_iterations=500,
    device=device,
)

weight_vectors = []
metadata_list = []
errors = []
start_time = time.time()

for i in tqdm(range(len(segments_data)), desc="SIREN fitting"):
    seg = segments_data[i]
    try:
        model, weights, meta = fitter.fit(seg['t'], seg['x'], seg['y'])
        weight_vectors.append(weights.cpu().numpy())
        metadata_list.append(meta)
        errors.append(meta['mean_pixel_error'])
    except Exception as e:
        continue

    # Checkpoint every 5000 segments
    if (i + 1) % 5000 == 0:
        elapsed = time.time() - start_time
        print(f"  [{i+1}/{len(segments_data)}] avg_error={np.mean(errors[-5000:]):.2f}px, time={elapsed/60:.1f}min")

weight_matrix = np.stack(weight_vectors)
print(f"\n✅ SIREN fitting complete: {len(weight_vectors):,} segments")
print(f"   Weight matrix shape: {weight_matrix.shape}")
print(f"   Mean pixel error: {np.mean(errors):.2f}px")
print(f"   Max pixel error:  {np.max(errors):.2f}px")
print(f"   Time: {(time.time() - start_time)/60:.1f} minutes")

# Save and upload to HF
np.save('data/siren_weights.npy', weight_matrix)
np.save('data/siren_metadata.npy', metadata_list, allow_pickle=True)
upload_file(path_or_fileobj='data/siren_weights.npy', path_in_repo='siren_weights.npy', repo_id=REPO_ID, token=HF_TOKEN)
upload_file(path_or_fileobj='data/siren_metadata.npy', path_in_repo='siren_metadata.npy', repo_id=REPO_ID, token=HF_TOKEN)
print(f"💾 Uploaded SIREN weights to HuggingFace: {REPO_ID}")

# %% [markdown]
# ## Cell 4: Train VQ-VAE (Phase 2)

# %%
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from trajectory_gen.models.vqvae import VQVAE
from huggingface_hub import upload_file
import time, json

# Load precomputed SIREN weights
weight_matrix = np.load('data/siren_weights.npy')
print(f"Training VQ-VAE on {weight_matrix.shape[0]:,} weight vectors of dim {weight_matrix.shape[1]}")

# Normalize weight vectors
w_mean = weight_matrix.mean(axis=0)
w_std = weight_matrix.std(axis=0) + 1e-8
weight_norm = (weight_matrix - w_mean) / w_std

# Create dataloader
dataset = TensorDataset(torch.tensor(weight_norm, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)

# Initialize VQ-VAE (scaled for VRAM)
vqvae = VQVAE(
    input_dim=weight_matrix.shape[1],
    hidden_dim=2048,
    embedding_dim=256,
    num_embeddings=512,
    commitment_cost=0.25,
    ema_decay=0.99,
).to(device)

optimizer = torch.optim.Adam(vqvae.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = torch.amp.GradScaler('cuda')

print(f"VQ-VAE params: {sum(p.numel() for p in vqvae.parameters()):,}")
print(f"Batches per epoch: {len(loader)}")

# Training loop
NUM_EPOCHS = 100
best_loss = float('inf')
start_time = time.time()
history = {'recon': [], 'vq': [], 'perplexity': []}

for epoch in range(NUM_EPOCHS):
    vqvae.train()
    epoch_recon, epoch_vq, epoch_perp = 0, 0, 0
    n_batches = 0

    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            x_recon, vq_loss, indices, perplexity = vqvae(batch)
            recon_loss = F.mse_loss(x_recon, batch)
            loss = recon_loss + vq_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_recon += recon_loss.item()
        epoch_vq += vq_loss.item()
        epoch_perp += perplexity.item()
        n_batches += 1

    scheduler.step()
    avg_recon = epoch_recon / n_batches
    avg_vq = epoch_vq / n_batches
    avg_perp = epoch_perp / n_batches
    history['recon'].append(avg_recon)
    history['vq'].append(avg_vq)
    history['perplexity'].append(avg_perp)

    if (epoch + 1) % 10 == 0:
        elapsed = (time.time() - start_time) / 60
        print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | recon={avg_recon:.6f} vq={avg_vq:.6f} perplexity={avg_perp:.1f} | {elapsed:.1f}min")

    # Save best + periodic checkpoint
    total_loss = avg_recon + avg_vq
    if total_loss < best_loss or (epoch + 1) % 25 == 0:
        if total_loss < best_loss:
            best_loss = total_loss
        ckpt = {
            'epoch': epoch,
            'model_state_dict': vqvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
            'config': {
                'input_dim': weight_matrix.shape[1],
                'hidden_dim': 2048, 'embedding_dim': 256,
                'num_embeddings': 512,
                'w_mean': w_mean.tolist(), 'w_std': w_std.tolist(),
            },
            'history': history,
        }
        torch.save(ckpt, 'vqvae_checkpoint.pt')
        upload_file(path_or_fileobj='vqvae_checkpoint.pt', path_in_repo='vqvae_checkpoint.pt', repo_id=REPO_ID, token=HF_TOKEN)

print(f"\n✅ VQ-VAE training complete in {(time.time() - start_time)/60:.1f} minutes")
print(f"   Best loss: {best_loss:.6f}")
print(f"   Final perplexity: {history['perplexity'][-1]:.1f} / 512")
print(f"💾 Saved to HuggingFace: {REPO_ID}/vqvae_checkpoint.pt")

# %% [markdown]
# ## Cell 5: Train Latent ODE (Phase 3)

# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from trajectory_gen.models.latent_ode import LatentODE
from huggingface_hub import upload_file
import time

# Load VQ-VAE and encode all segments to primitive sequences
vqvae.eval()
weight_tensor = torch.tensor(weight_norm, dtype=torch.float32).to(device)

with torch.no_grad():
    z_e = vqvae.encoder(weight_tensor)  # (N, 256)
    _, _, indices, _ = vqvae.vq(z_e)  # (N,) codebook indices
    embeddings = vqvae.vq.embedding(indices)  # (N, 256) quantized

embeddings_np = embeddings.cpu().numpy()
print(f"Encoded {len(embeddings_np):,} segments to {embeddings_np.shape[1]}d embeddings")

# Create sequences of consecutive primitives
SEQ_LEN = 16
sequences = []
for i in range(0, len(embeddings_np) - SEQ_LEN, SEQ_LEN // 2):  # 50% overlap
    sequences.append(embeddings_np[i:i+SEQ_LEN])

sequences = np.stack(sequences)
print(f"Created {len(sequences):,} sequences of length {SEQ_LEN}")

# Create dataloader
seq_tensor = torch.tensor(sequences, dtype=torch.float32)
time_points = torch.linspace(0, 1, SEQ_LEN)  # normalized time
seq_dataset = TensorDataset(seq_tensor)
seq_loader = DataLoader(seq_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

# Initialize Latent ODE
latent_ode = LatentODE(
    input_dim=256,
    latent_dim=64,
    rec_hidden_dim=256,
    gen_hidden_dim=512,
    use_adjoint=True,
).to(device)

optimizer = torch.optim.Adamax(latent_ode.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

print(f"Latent ODE params: {latent_ode.num_parameters:,}")

# Training loop with KL annealing
NUM_EPOCHS = 200
best_loss = float('inf')
start_time = time.time()
kl_anneal_rate = 0.99  # per supplement Section 6
history = {'total': [], 'recon': [], 'kl': []}

for epoch in range(NUM_EPOCHS):
    latent_ode.train()
    kl_weight = 1.0 - kl_anneal_rate ** (epoch + 1)  # ramp 0 → 1

    epoch_total, epoch_recon, epoch_kl = 0, 0, 0
    n_batches = 0

    for (batch,) in seq_loader:
        batch = batch.to(device)
        t = time_points.to(device)
        optimizer.zero_grad()

        try:
            x_recon, mu, logvar = latent_ode(batch, t)
            loss, metrics = latent_ode.compute_loss(
                batch, x_recon, mu, logvar,
                kl_weight=kl_weight, obs_variance=0.01,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_ode.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_total += metrics['total_loss']
            epoch_recon += metrics['recon_loss']
            epoch_kl += metrics['kl_loss']
            n_batches += 1
        except Exception as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise

    scheduler.step()

    if n_batches > 0:
        avg_total = epoch_total / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl = epoch_kl / n_batches
        history['total'].append(avg_total)
        history['recon'].append(avg_recon)
        history['kl'].append(avg_kl)

        if (epoch + 1) % 20 == 0:
            elapsed = (time.time() - start_time) / 60
            nfe = latent_ode.ode_func.nfe
            latent_ode.ode_func.nfe = 0
            print(f"  Epoch {epoch+1:3d}/{NUM_EPOCHS} | loss={avg_total:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f} kl_w={kl_weight:.3f} NFE={nfe} | {elapsed:.1f}min")

        # Checkpoint
        if avg_total < best_loss or (epoch + 1) % 50 == 0:
            if avg_total < best_loss:
                best_loss = avg_total
            ckpt = {
                'epoch': epoch,
                'model_state_dict': latent_ode.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_total,
                'config': {
                    'input_dim': 256, 'latent_dim': 64,
                    'rec_hidden_dim': 256, 'gen_hidden_dim': 512,
                    'seq_len': SEQ_LEN,
                    'siren_param_count': weight_matrix.shape[1],
                    'hidden_features': 64, 'hidden_layers': 3,
                    'embedding_dim': 256, 'num_embeddings': 512,
                },
                'history': history,
            }
            torch.save(ckpt, 'latent_ode_checkpoint.pt')
            upload_file(path_or_fileobj='latent_ode_checkpoint.pt', path_in_repo='latent_ode_checkpoint.pt', repo_id=REPO_ID, token=HF_TOKEN)

print(f"\n✅ Latent ODE training complete in {(time.time() - start_time)/60:.1f} minutes")
print(f"   Best loss: {best_loss:.4f}")
print(f"💾 Saved to HuggingFace: {REPO_ID}/latent_ode_checkpoint.pt")

# %% [markdown]
# ## Cell 6: Save Final Combined Checkpoint + Test Generation

# %%
# Save combined checkpoint with everything needed for inference
final_checkpoint = {
    'config': {
        'siren_param_count': weight_matrix.shape[1],
        'hidden_features': 64, 'hidden_layers': 3,
        'embedding_dim': 256, 'num_embeddings': 512,
        'latent_dim': 64, 'rec_hidden_dim': 256, 'gen_hidden_dim': 512,
        'w_mean': w_mean.tolist(), 'w_std': w_std.tolist(),
    },
    'vqvae_state_dict': vqvae.state_dict(),
    'latent_ode_state_dict': latent_ode.state_dict(),
}
torch.save(final_checkpoint, 'trajectory_model_final.pt')
upload_file(path_or_fileobj='trajectory_model_final.pt', path_in_repo='trajectory_model_final.pt', repo_id=REPO_ID, token=HF_TOKEN)
print(f"💾 Final model uploaded to HuggingFace: {REPO_ID}/trajectory_model_final.pt")

# %%
# Test: generate a trajectory
import matplotlib.pyplot as plt
from trajectory_gen.models.siren import SIREN

latent_ode.eval()
vqvae.eval()

with torch.no_grad():
    # Sample from Latent ODE prior
    z0 = torch.randn(3, 64, device=device)
    t_gen = torch.linspace(0, 1, SEQ_LEN, device=device)
    z_traj = latent_ode.decode_latent_trajectory(z0, t_gen)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for sample_idx in range(3):
        embs = z_traj[sample_idx]  # (SEQ_LEN, 256)
        _, _, indices, _ = vqvae.vq(embs)
        siren_weights_batch = vqvae.decode_indices(indices)

        # Denormalize
        w_mean_t = torch.tensor(w_mean, dtype=torch.float32, device=device)
        w_std_t = torch.tensor(w_std, dtype=torch.float32, device=device)
        siren_weights_batch = siren_weights_batch * w_std_t + w_mean_t

        all_x, all_y = [], []
        for i in range(SEQ_LEN):
            siren = SIREN(in_features=1, hidden_features=64, hidden_layers=3, out_features=2).to(device)
            siren.set_weight_vector(siren_weights_batch[i])
            t_pts, x_pts, y_pts = siren.compute_trajectory(num_points=50)
            all_x.extend(x_pts)
            all_y.extend(y_pts)

        axes[sample_idx].plot(all_x, all_y, '-', linewidth=0.8, alpha=0.8)
        axes[sample_idx].set_title(f'Sample {sample_idx + 1}')
        axes[sample_idx].set_aspect('equal')
        axes[sample_idx].grid(True, alpha=0.3)

    plt.suptitle('Generated Trajectories (from Latent ODE prior)', fontsize=14)
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=150)
    plt.show()
    upload_file(path_or_fileobj='generated_samples.png', path_in_repo='generated_samples.png', repo_id=REPO_ID, token=HF_TOKEN)
    print("💾 Sample plot uploaded to HuggingFace")

print("\n🎉 PIPELINE COMPLETE!")
print(f"   Model: https://huggingface.co/{REPO_ID}")
print(f"   Files: trajectory_model_final.pt, vqvae_checkpoint.pt, latent_ode_checkpoint.pt")
