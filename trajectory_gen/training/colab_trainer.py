"""
Colab-optimized training loop with checkpoint management.

Designed for Google Colab constraints:
    - 16GB VRAM (T4 GPU)
    - 4-hour session limit
    - Checkpoint every 10 minutes to survive disconnects

Training schedule (supplement Section 6):
    - Optimizer: Adamax, lr=0.01, decay rate 0.999
    - KL annealing coefficient: 0.99
    - 3 samples from posterior for ELBO estimation
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm


class CoLabTrainer:
    """
    Training manager for Google Colab with auto-checkpointing.
    
    Handles:
        - Mixed precision (fp16) for ~2x speedup on T4
        - Gradient accumulation for larger effective batch size
        - Automatic checkpoint save/resume
        - Learning rate warmup + cosine decay
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval_min: int = 10,
        use_amp: bool = True,
        grad_accumulation_steps: int = 4,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval_min * 60  # seconds
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_accumulation_steps = grad_accumulation_steps
        
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.last_checkpoint_time = time.time()

    def save_checkpoint(self, loss: float, extra: Optional[Dict[str, Any]] = None):
        """Save training state to checkpoint."""
        state = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'loss': loss,
            'timestamp': time.strftime('%Y-%m-%d_%H:%M:%S'),
        }
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
        if extra:
            state.update(extra)
        
        path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
        torch.save(state, path)
        
        # Keep only last 3 + best
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoints) > 4:
            for old in checkpoints[:-3]:
                if 'best' not in old.name:
                    old.unlink()
        
        # Save best
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
        
        print(f"  💾 Checkpoint saved: step={self.step}, loss={loss:.6f}")

    def load_checkpoint(self) -> bool:
        """Resume from latest checkpoint if available."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return False
        
        latest = checkpoints[-1]
        state = torch.load(latest, map_location='cpu')
        
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.step = state['step']
        self.epoch = state.get('epoch', 0)
        self.best_loss = state.get('best_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])
        
        print(f"  ✅ Resumed from: step={self.step}, loss={state.get('loss', '?')}")
        return True

    def should_checkpoint(self) -> bool:
        """Check if enough time has passed for a checkpoint."""
        return (time.time() - self.last_checkpoint_time) >= self.checkpoint_interval

    def train_step(self, batch, loss_fn, device):
        """Single training step with optional mixed precision."""
        self.model.train()
        
        if isinstance(batch, (list, tuple)):
            batch = [b.to(device) for b in batch]
        else:
            batch = batch.to(device)
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                loss = loss_fn(self.model, batch)
                loss = loss / self.grad_accumulation_steps
            self.scaler.scale(loss).backward()
            
            if (self.step + 1) % self.grad_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss = loss_fn(self.model, batch)
            loss = loss / self.grad_accumulation_steps
            loss.backward()
            
            if (self.step + 1) % self.grad_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        self.step += 1
        return loss.item() * self.grad_accumulation_steps
