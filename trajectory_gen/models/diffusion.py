import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Build sinusoidal embeddings.
    This matches the implementation in Denoising Diffusion Probabilistic Models.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb, c_emb):
        h = self.act1(self.bn1(self.conv1(x)))
        
        # Add time and condition embeddings
        time_emb = self.time_mlp(t_emb).unsqueeze(-1)
        cond_emb = self.cond_mlp(c_emb).unsqueeze(-1)
        h = h + time_emb + cond_emb
        
        h = self.bn2(self.conv2(h))
        return self.act2(h + self.shortcut(x))

class UNet1D(nn.Module):
    """
    1D U-Net for denoising sequences.
    Takes noisy trajectory, time embedding, and conditioning vector.
    """
    def __init__(self, channels=2, cond_dim=2, time_emb_dim=128, base_dim=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder (Downsampling)
        self.inc = nn.Conv1d(channels, base_dim, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_dim, base_dim * 2, time_emb_dim, time_emb_dim)
        self.pool1 = nn.Conv1d(base_dim * 2, base_dim * 2, kernel_size=4, stride=2, padding=1)
        
        self.down2 = ResidualBlock(base_dim * 2, base_dim * 4, time_emb_dim, time_emb_dim)
        self.pool2 = nn.Conv1d(base_dim * 4, base_dim * 4, kernel_size=4, stride=2, padding=1)
        
        self.down3 = ResidualBlock(base_dim * 4, base_dim * 4, time_emb_dim, time_emb_dim)
        self.pool3 = nn.Conv1d(base_dim * 4, base_dim * 4, kernel_size=4, stride=2, padding=1)
        
        # Bottleneck
        self.mid = ResidualBlock(base_dim * 4, base_dim * 4, time_emb_dim, time_emb_dim)
        
        # Decoder (Upsampling with skip connections)
        self.up1 = nn.ConvTranspose1d(base_dim * 4, base_dim * 4, kernel_size=4, stride=2, padding=1)
        self.up_res1 = ResidualBlock(base_dim * 8, base_dim * 2, time_emb_dim, time_emb_dim)
        
        self.up2 = nn.ConvTranspose1d(base_dim * 2, base_dim * 2, kernel_size=4, stride=2, padding=1)
        # u2: 128, d2: 256 -> 384
        self.up_res2 = ResidualBlock(base_dim * 6, base_dim, time_emb_dim, time_emb_dim)
        
        self.up3 = nn.ConvTranspose1d(base_dim, base_dim, kernel_size=4, stride=2, padding=1)
        # u3: 64, d1: 128 -> 192
        self.up_res3 = ResidualBlock(base_dim * 3, base_dim, time_emb_dim, time_emb_dim)
        
        self.outc = nn.Conv1d(base_dim, channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.cond_mlp(cond)
        
        x0 = self.inc(x)
        
        d1 = self.down1(x0, t_emb, c_emb)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1, t_emb, c_emb)
        p2 = self.pool2(d2)
        
        d3 = self.down3(p2, t_emb, c_emb)
        p3 = self.pool3(d3)
        
        m = self.mid(p3, t_emb, c_emb)
        
        u1 = self.up1(m)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.up_res1(u1, t_emb, c_emb)
        
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_res2(u2, t_emb, c_emb)
        
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.up_res3(u3, t_emb, c_emb)
        
        return self.outc(u3)

class TrajectoryDiffusion(nn.Module):
    """
    Denoising Diffusion Probabilistic Model for trajectories.
    """
    def __init__(self, seq_len=64, channels=2, cond_dim=2, timesteps=100):
        super().__init__()
        self.seq_len = seq_len
        self.channels = channels
        self.timesteps = timesteps
        
        self.unet = UNet1D(channels=channels, cond_dim=cond_dim, base_dim=64)
        
        # Define noise schedule (Linear schedule)
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: add noise to data at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def compute_loss(self, x_start, cond):
        """
        Compute MSE loss between true noise and predicted noise.
        x_start shape: (B, 2, 64)
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = self.unet(x_noisy, t, cond)
        
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, cond, batch_size=1):
        """
        Reverse process: Generate trajectories from pure noise.
        cond shape: (B, 2)
        """
        device = cond.device
        x = torch.randn(batch_size, self.channels, self.seq_len, device=device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            predicted_noise = self.unet(x, t, cond)
            
            alpha = (1 - self.betas[i])
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
            
        return x
