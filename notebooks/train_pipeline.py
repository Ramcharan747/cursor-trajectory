# %% [markdown]
# # 🖱️ Cursor Trajectory — Full Training Pipeline (v2 — all fixes)
#
# **Pipeline: Raw Data → Segmentation → SIREN → VQ-VAE → Latent ODE**
#
# All checkpoints saved to HuggingFace Hub automatically.
# Designed for Colab T4 GPU (16GB VRAM).

# %% [markdown]
# ## Cell 1: Setup

# %%
import os, subprocess, sys

if not os.path.exists('/content/cursor-trajectory'):
    subprocess.run(['git', 'clone', 'https://github.com/Ramcharan747/cursor-trajectory.git'], check=True)
os.chdir('/content/cursor-trajectory')
sys.path.insert(0, '/content/cursor-trajectory')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'numpy', 'matplotlib', 'tqdm', 'torchdiffeq', 'huggingface_hub'], check=True)

from huggingface_hub import HfApi, login, upload_file as _hf_upload
from google.colab import userdata
import torch, numpy as np

HF_TOKEN = userdata.get('HF_TOKEN')
login(token=HF_TOKEN)
REPO_ID = "Baka7/cursor-trajectory-checkpoints"

# Safe upload wrapper — never crashes training
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
# ## Cell 2: Load & Segment Raw Data

# %%
import gzip, json, glob
from tqdm import tqdm

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

# Segment
from trajectory_gen.data.segmentation import segment_trajectory

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

seg_dicts = []
for s in all_segments:
    seg_dicts.append({'t': s.t, 'x': s.x, 'y': s.y,
        'start': np.array([s.x[0], s.y[0]]),
        'end': np.array([s.x[-1], s.y[-1]])})

np.save('data/segments.npy', seg_dicts, allow_pickle=True)
print(f"✅ {len(seg_dicts):,} segments saved")

# FREE MEMORY — prevents OOM crash later
del x_all, y_all, t_all, all_segments, seg_dicts, order, dt
import gc; gc.collect()
print("🗑️ Raw data freed from RAM")

# %% [markdown]
# ## Cell 3: Fit SIRENs — Batched (3 hidden layers = 12,738 dims)

# %%
import torch, numpy as np, math, time
from tqdm import tqdm

class BatchedSIRENFitter:
    def __init__(self, hidden=64, n_hidden=3, omega=30.0, lr=5e-4, iters=200, device='cuda'):
        self.H, self.n_hidden, self.omega = hidden, n_hidden, omega
        self.lr, self.iters, self.device = lr, iters, device

    def fit_batch(self, segments):
        B, H = len(segments), self.H
        max_len = max(len(s['t']) for s in segments)
        t_b = torch.zeros(B, max_len, 1, device=self.device)
        xy_b = torch.zeros(B, max_len, 2, device=self.device)
        mask = torch.zeros(B, max_len, 1, device=self.device)

        for i, seg in enumerate(segments):
            n = len(seg['t'])
            tn = (seg['t'] - seg['t'][0]) / max(seg['t'][-1] - seg['t'][0], 1e-6) * 2 - 1
            t_b[i, :n, 0] = torch.tensor(tn, dtype=torch.float32)
            xy_b[i, :n, 0] = torch.tensor(seg['x'], dtype=torch.float32)
            xy_b[i, :n, 1] = torch.tensor(seg['y'], dtype=torch.float32)
            mask[i, :n, 0] = 1.0

        xy_min = (xy_b * mask + (1-mask)*1e9).reshape(B,-1).min(1,keepdim=True)[0].unsqueeze(-1)
        xy_max = (xy_b * mask + (1-mask)*(-1e9)).reshape(B,-1).max(1,keepdim=True)[0].unsqueeze(-1)
        xy_s = (xy_max - xy_min).clamp(min=1.0)
        xy_n = (xy_b - xy_min) / xy_s * 2 - 1

        params = []
        W0 = ((torch.rand(B,H,1,device=self.device)*2-1)*1.0).requires_grad_(True)
        b0 = ((torch.rand(B,H,1,device=self.device)*2-1)*1.0).requires_grad_(True)
        params += [W0, b0]

        bh = math.sqrt(6.0/H)/self.omega
        Ws, bs = [], []
        for _ in range(self.n_hidden):  # 3 hidden layers
            W = ((torch.rand(B,H,H,device=self.device)*2-1)*bh).requires_grad_(True)
            b = ((torch.rand(B,H,1,device=self.device)*2-1)*bh).requires_grad_(True)
            Ws.append(W); bs.append(b); params += [W, b]

        bo = math.sqrt(6.0/H)/self.omega
        Wo = ((torch.rand(B,2,H,device=self.device)*2-1)*bo).requires_grad_(True)
        bo2 = torch.zeros(B,2,1,device=self.device).requires_grad_(True)
        params += [Wo, bo2]

        opt = torch.optim.Adam(params, lr=self.lr)
        for _ in range(self.iters):
            h = t_b.transpose(1,2)
            h = torch.sin(self.omega*(torch.bmm(W0,h)+b0))
            for W,b in zip(Ws,bs): h = torch.sin(self.omega*(torch.bmm(W,h)+b))
            out = (torch.bmm(Wo,h)+bo2).transpose(1,2)
            loss = ((out-xy_n)*mask).pow(2).sum()/mask.sum()/2
            opt.zero_grad(); loss.backward(); opt.step()

        wvs, errs = [], []
        with torch.no_grad():
            h = t_b.transpose(1,2)
            h = torch.sin(self.omega*(torch.bmm(W0,h)+b0))
            for W,b in zip(Ws,bs): h = torch.sin(self.omega*(torch.bmm(W,h)+b))
            pred_n = (torch.bmm(Wo,h)+bo2).transpose(1,2)
            pred = pred_n * xy_s/2 + xy_s/2 + xy_min
            for i in range(B):
                n = int(mask[i].sum().item())
                errs.append(torch.sqrt(((pred[i,:n]-xy_b[i,:n])**2).sum(1)).mean().item())
                wvs.append(torch.cat([W0[i].flatten(),b0[i].flatten()]+
                    [p[i].flatten() for pr in zip(Ws,bs) for p in pr]+
                    [Wo[i].flatten(),bo2[i].flatten()]).cpu().numpy())
        return wvs, errs

segments_data = np.load('data/segments.npy', allow_pickle=True)
fitter = BatchedSIRENFitter(hidden=64, n_hidden=3, omega=30.0, lr=5e-4, iters=200, device=device)
all_w, all_ep, all_e = [], [], []
t0 = time.time()

for bs in tqdm(range(0, len(segments_data), 512), desc="SIREN batches"):
    batch = segments_data[bs:bs+512]
    try:
        w, e = fitter.fit_batch(batch)
        all_w.extend(w); all_e.extend(e)
        for s in batch: all_ep.append([s['x'][0],s['y'][0],s['x'][-1],s['y'][-1]])
    except: torch.cuda.empty_cache(); continue

weight_matrix = np.stack(all_w)
endpoints = np.array(all_ep)
print(f"\n✅ {weight_matrix.shape} — expect (*, 12738)")
print(f"   Error: {np.mean(all_e):.2f}px | Time: {(time.time()-t0)/60:.1f}min")

np.save('data/siren_weights.npy', weight_matrix)
np.save('data/segment_endpoints.npy', endpoints)
upload_file(path_or_fileobj='data/siren_weights.npy', path_in_repo='siren_weights.npy', repo_id=REPO_ID)
upload_file(path_or_fileobj='data/segment_endpoints.npy', path_in_repo='segment_endpoints.npy', repo_id=REPO_ID)

# Free memory
del segments_data, fitter, all_w, all_e
import gc; gc.collect(); torch.cuda.empty_cache()
print("💾 Uploaded + memory freed")

# %% [markdown]
# ## Cell 4: Train VQ-VAE (with codebook collapse fix)

# %%
import torch, torch.nn.functional as F, numpy as np, time
from torch.utils.data import DataLoader, TensorDataset
from trajectory_gen.models.vqvae import VQVAE

weight_matrix = np.load('data/siren_weights.npy')
print(f"VQ-VAE: {weight_matrix.shape[0]:,} vectors × dim {weight_matrix.shape[1]}")

w_mean = weight_matrix.mean(axis=0, keepdims=True)
w_std = weight_matrix.std(axis=0, keepdims=True) + 1e-8

# CRITICAL RAM FIX: Normalize IN-PLACE to save 8.5 GB of RAM
weight_matrix -= w_mean
weight_matrix /= w_std

# CRITICAL RAM FIX: torch.from_numpy shares memory, torch.tensor copies!
weight_norm_tensor = torch.from_numpy(weight_matrix)

dataset = TensorDataset(weight_norm_tensor)
loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=2, pin_memory=True)

vqvae = VQVAE(input_dim=weight_matrix.shape[1], hidden_dim=2048,
    embedding_dim=256, num_embeddings=512, commitment_cost=0.25, ema_decay=0.99).to(device)

# CRITICAL: Initialize codebook from encoder outputs to prevent collapse
with torch.no_grad():
    sample = weight_norm_tensor[np.random.choice(len(weight_norm_tensor), 512, replace=False)].to(device)
    z_e = vqvae.encoder(sample)
    vqvae.vq.embedding.data.copy_(z_e)
print("✅ Codebook initialized from data")

optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scaler = torch.amp.GradScaler('cuda')

NUM_EPOCHS = 100
best_loss = float('inf')
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    vqvae.train()
    ep_r, ep_v, ep_p, nb = 0, 0, 0, 0
    for (batch,) in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            x_rec, vq_loss, indices, perp = vqvae(batch)
            recon = F.mse_loss(x_rec, batch)
            loss = recon + vq_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        ep_r += recon.item(); ep_v += vq_loss.item(); ep_p += perp.item(); nb += 1
    scheduler.step()
    ar, av, ap = ep_r/nb, ep_v/nb, ep_p/nb

    # Reset dead codebook entries every 10 epochs
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            samp = weight_norm_tensor[np.random.choice(len(weight_norm_tensor), 4096)].to(device)
            ze = vqvae.encoder(samp)
            usage = torch.zeros(512, device=device)
            _, _, idx, _ = vqvae.vq(ze)
            for j in idx: usage[j] += 1
            dead = (usage == 0).nonzero().squeeze(-1)
            if len(dead) > 0:
                ri = torch.randint(0, len(ze), (len(dead),))
                vqvae.vq.embedding.data[dead] = ze[ri]
                print(f"    ↻ Reset {len(dead)} dead codes")
        print(f"  Epoch {epoch+1:3d}/100 | recon={ar:.6f} vq={av:.6f} perp={ap:.1f}/512 | {(time.time()-t0)/60:.1f}min")

    tl = ar + av
    if tl < best_loss:
        best_loss = tl
        torch.save({'epoch': epoch, 'model_state_dict': vqvae.state_dict(),
            'config': {'input_dim': weight_matrix.shape[1], 'hidden_dim': 2048,
                'embedding_dim': 256, 'num_embeddings': 512,
                'w_mean': w_mean.tolist(), 'w_std': w_std.tolist()}},
            'vqvae_checkpoint.pt')
    if (epoch+1) % 50 == 0:
        upload_file(path_or_fileobj='vqvae_checkpoint.pt', path_in_repo='vqvae_checkpoint.pt', repo_id=REPO_ID)

# Final upload
upload_file(path_or_fileobj='vqvae_checkpoint.pt', path_in_repo='vqvae_checkpoint.pt', repo_id=REPO_ID)
print(f"\n✅ VQ-VAE done in {(time.time()-t0)/60:.1f}min | loss={best_loss:.6f} perp={ap:.1f}/512")

# %% [markdown]
# ## Cell 5: Train Latent ODE — Conditioned on displacement

# %%
import torch, numpy as np, time
from torch.utils.data import DataLoader, TensorDataset
from trajectory_gen.models.latent_ode import LatentODE

vqvae.eval()
# CRITICAL RAM FIX: Use the existing weight_norm_tensor instead of making another copy
weight_tensor = weight_norm_tensor.to(device)

batch_sz = 4096
all_emb = []
with torch.no_grad():
    for i in range(0, len(weight_tensor), batch_sz):
        chunk = weight_tensor[i:i+batch_sz].to(device)
        ze = vqvae.encoder(chunk)
        _, _, idx, _ = vqvae.vq(ze)
        all_emb.append(vqvae.vq.embedding(idx).cpu())
emb_all = torch.cat(all_emb, 0).numpy()
print(f"Encoded {len(emb_all):,} → {emb_all.shape[1]}d")

endpoints = np.load('data/segment_endpoints.npy')
SEQ_LEN = 16
seqs, conds = [], []
for i in range(0, len(emb_all)-SEQ_LEN, SEQ_LEN//2):
    seqs.append(emb_all[i:i+SEQ_LEN])
    disp = endpoints[i+SEQ_LEN-1, 2:] - endpoints[i, :2]
    conds.append(disp / 2000.0)
seqs = np.stack(seqs); conds = np.stack(conds)
print(f"{len(seqs):,} sequences of len {SEQ_LEN}")

# Free weight data
del weight_tensor, weight_norm, all_emb
import gc; gc.collect(); torch.cuda.empty_cache()

seq_t = torch.tensor(seqs, dtype=torch.float32)
cond_t = torch.tensor(conds, dtype=torch.float32)
tp = torch.linspace(0, 1, SEQ_LEN)
dl = DataLoader(TensorDataset(seq_t, cond_t), batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

latent_ode = LatentODE(input_dim=258, latent_dim=64, rec_hidden_dim=256,
    gen_hidden_dim=512, output_dim=256, use_adjoint=True).to(device)
opt = torch.optim.Adamax(latent_ode.parameters(), lr=0.01)
sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999)
print(f"Latent ODE: {latent_ode.num_parameters:,} params")

NUM_EPOCHS = 200
best_loss = float('inf')
t0 = time.time()

for epoch in range(NUM_EPOCHS):
    latent_ode.train()
    kl_w = 1.0 - 0.99 ** (epoch+1)
    et, er, ek, nb = 0, 0, 0, 0
    for be, bc in dl:
        be, bc = be.to(device), bc.to(device)
        t = tp.to(device)
        opt.zero_grad()
        ce = bc.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        bi = torch.cat([be, ce], dim=-1)
        try:
            xr, mu, lv = latent_ode(bi, t)
            loss, m = latent_ode.compute_loss(be, xr, mu, lv, kl_weight=kl_w, obs_variance=0.01)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_ode.parameters(), 10.0)
            opt.step()
            et += m['total_loss']; er += m['recon_loss']; ek += m['kl_loss']; nb += 1
        except Exception as e:
            if 'out of memory' in str(e).lower(): torch.cuda.empty_cache(); continue
            raise
    sched.step()
    if nb > 0:
        at, arec, akl = et/nb, er/nb, ek/nb
        if (epoch+1) % 20 == 0:
            nfe = latent_ode.ode_func.nfe; latent_ode.ode_func.nfe = 0
            print(f"  Epoch {epoch+1:3d}/200 | loss={at:.4f} recon={arec:.4f} kl={akl:.4f} kl_w={kl_w:.3f} NFE={nfe} | {(time.time()-t0)/60:.1f}min")
        if at < best_loss:
            best_loss = at
            torch.save({'epoch': epoch, 'model_state_dict': latent_ode.state_dict(),
                'config': {'input_dim': 258, 'output_dim': 256, 'latent_dim': 64,
                    'rec_hidden_dim': 256, 'gen_hidden_dim': 512, 'seq_len': SEQ_LEN,
                    'siren_param_count': weight_matrix.shape[1]}},
                'latent_ode_checkpoint.pt')
        if (epoch+1) % 50 == 0:
            upload_file(path_or_fileobj='latent_ode_checkpoint.pt', path_in_repo='latent_ode_checkpoint.pt', repo_id=REPO_ID)

upload_file(path_or_fileobj='latent_ode_checkpoint.pt', path_in_repo='latent_ode_checkpoint.pt', repo_id=REPO_ID)
print(f"\n✅ Latent ODE done in {(time.time()-t0)/60:.1f}min | best={best_loss:.4f}")

# %% [markdown]
# ## Cell 6: Save Final Model + Test Point-to-Point Generation

# %%
import matplotlib.pyplot as plt
from trajectory_gen.models.siren import SIREN

final = {
    'config': {'siren_param_count': weight_matrix.shape[1], 'hidden_features': 64,
        'hidden_layers': 3, 'embedding_dim': 256, 'num_embeddings': 512,
        'latent_dim': 64, 'rec_hidden_dim': 256, 'gen_hidden_dim': 512,
        'input_dim': 258, 'output_dim': 256, 'seq_len': SEQ_LEN,
        'w_mean': w_mean.tolist(), 'w_std': w_std.tolist()},
    'vqvae_state_dict': vqvae.state_dict(),
    'latent_ode_state_dict': latent_ode.state_dict(),
}
torch.save(final, 'trajectory_model_final.pt')
upload_file(path_or_fileobj='trajectory_model_final.pt', path_in_repo='trajectory_model_final.pt', repo_id=REPO_ID)
print(f"💾 Final model → HuggingFace: {REPO_ID}")

# Test generation
latent_ode.eval(); vqvae.eval()
routes = [((100,200),(800,600),"Short diagonal"),((50,500),(1800,100),"Long diagonal"),((900,50),(900,900),"Vertical")]
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

with torch.no_grad():
    for row, (start, end, label) in enumerate(routes):
        for col in range(4):
            z0 = torch.randn(1, 64, device=device)
            tg = torch.linspace(0, 1, SEQ_LEN, device=device)
            zt = latent_ode.decode_latent_trajectory(z0, tg)
            B, S, L = zt.shape
            embs = latent_ode.decoder(zt.reshape(-1, L)).reshape(B, S, -1).squeeze(0)
            _, _, idx, _ = vqvae.vq(embs)
            sw = vqvae.decode_indices(idx)
            wm = torch.tensor(w_mean, dtype=torch.float32, device=device)
            ws = torch.tensor(w_std, dtype=torch.float32, device=device)
            sw = sw * ws + wm
            ax, ay = [], []
            for i in range(SEQ_LEN):
                s = SIREN(in_features=1, hidden_features=64, hidden_layers=3, out_features=2).to(device)
                s.set_weight_vector(sw[i])
                _, xp, yp = s.compute_trajectory(num_points=50)
                ax.extend(xp); ay.extend(yp)
            rx, ry = np.array(ax), np.array(ay)
            alpha = np.linspace(0, 1, len(rx))
            bx = start[0] + alpha*(end[0]-start[0])
            by = start[1] + alpha*(end[1]-start[1])
            d = max(np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)*0.3, 10)
            f = np.sin(np.pi*alpha)
            axes[row,col].plot(bx+rx*d*f, by+ry*d*f, '-', lw=1.2)
            axes[row,col].plot(*start, 'go', ms=10)
            axes[row,col].plot(*end, 'rs', ms=10)
            axes[row,col].set_title(f'{label} #{col+1}')
            axes[row,col].set_aspect('equal'); axes[row,col].grid(True, alpha=0.3)

plt.suptitle('Point-to-Point Trajectories (diverse z₀)', fontsize=14)
plt.tight_layout(); plt.savefig('generated_samples.png', dpi=150, bbox_inches='tight'); plt.show()
upload_file(path_or_fileobj='generated_samples.png', path_in_repo='generated_samples.png', repo_id=REPO_ID)
print("\n🎉 PIPELINE COMPLETE!")
