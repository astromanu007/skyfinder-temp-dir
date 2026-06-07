

import os
import time
import argparse
import logging
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import gmean

from datasets import SkyFinder
from model    import ResNet18Regressor
from fds      import FDS
from loss     import (weighted_mse_loss, weighted_l1_loss,
                      weighted_huber_loss, weighted_focal_l1_loss)

# ── Args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()

    # Experiment
    p.add_argument('--experiment',   type=str, default='baseline')
    p.add_argument('--data_csv',     type=str, default='../data/skyfinder/skyfinder.csv')
    p.add_argument('--out_dir',      type=str, default='../results')

    # LDS
    p.add_argument('--lds',          action='store_true', default=False)
    p.add_argument('--lds_kernel',   type=str,   default='gaussian',
                   choices=['gaussian', 'triang', 'laplace'])
    p.add_argument('--lds_ks',       type=int,   default=5)
    p.add_argument('--lds_sigma',    type=float, default=2)

    # FDS
    p.add_argument('--fds',          action='store_true', default=False)
    p.add_argument('--fds_kernel',   type=str,   default='gaussian',
                   choices=['gaussian', 'triang', 'laplace'])
    p.add_argument('--fds_ks',       type=int,   default=5)
    p.add_argument('--fds_sigma',    type=float, default=1)
    p.add_argument('--start_update', type=int,   default=0)
    p.add_argument('--start_smooth', type=int,   default=1)
    p.add_argument('--fds_mmt',      type=float, default=0.9)

    # Reweighting
    p.add_argument('--reweight',     type=str,   default='none',
                   choices=['none', 'sqrt_inv', 'inverse'])

    # Training
    p.add_argument('--epochs',       type=int,   default=20)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--img_size',     type=int,   default=224)
    p.add_argument('--num_workers',  type=int,   default=0)
    p.add_argument('--loss',         type=str,   default='l1',
                   choices=['l1', 'mse', 'huber', 'focal_l1', 'focal_mse'])
    p.add_argument('--seed',         type=int,   default=42)

    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_loss_fn(name):
    return {
        'l1':       weighted_l1_loss,
        'mse':      weighted_mse_loss,
        'huber':    weighted_huber_loss,
        'focal_l1': weighted_focal_l1_loss,
    }[name]


def shot_metrics(preds, labels, train_labels,
                 many_shot_thr=100, low_shot_thr=20):
    train_bins = np.round(np.array(train_labels)).astype(int)
    label_bins = np.round(np.array(labels)).astype(int)
    pred_arr   = np.array(preds)

    bin_counts = {}
    for b in train_bins:
        bin_counts[b] = bin_counts.get(b, 0) + 1

    many_l1, median_l1, low_l1 = [], [], []
    many_cnt = median_cnt = low_cnt = 0

    for b in np.unique(label_bins):
        cnt_train = bin_counts.get(b, 0)
        mask      = (label_bins == b)
        l1_vals   = np.abs(pred_arr[mask] - label_bins[mask])

        if cnt_train > many_shot_thr:
            many_l1.extend(l1_vals); many_cnt += mask.sum()
        elif cnt_train < low_shot_thr:
            low_l1.extend(l1_vals);  low_cnt  += mask.sum()
        else:
            median_l1.extend(l1_vals); median_cnt += mask.sum()

    def safe_mean(lst): return float(np.mean(lst)) if lst else float('nan')
    def safe_gmean(lst): return float(gmean(lst)) if lst else float('nan')

    return {
        'many':   {'l1': safe_mean(many_l1),   'gmean': safe_gmean(many_l1)},
        'median': {'l1': safe_mean(median_l1), 'gmean': safe_gmean(median_l1)},
        'low':    {'l1': safe_mean(low_l1),    'gmean': safe_gmean(low_l1)},
    }


# ── Train / Validate loops ────────────────────────────────────────────────────

def train_one_epoch(loader, model, optimizer, loss_fn, device, args, epoch):
    model.train()
    meter = AverageMeter()

    for imgs, targets, weights in tqdm(loader, desc=f"  Train epoch {epoch}", leave=False):
        imgs, targets, weights = imgs.to(device), targets.to(device), weights.to(device)

        # Forward pass — unpack safely
        output = model(imgs, targets, epoch)
        if isinstance(output, tuple):
            preds = output[0]   # first element is always the prediction
        else:
            preds = output

        loss = loss_fn(preds, targets, weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meter.update(loss.item(), imgs.size(0))

    # FDS stat update after full epoch
    if args.fds and epoch >= args.start_update:
        print(f"    [FDS] Collecting features for epoch {epoch}...")
        model.eval()
        encodings, all_labels = [], []
        with torch.no_grad():
            for imgs, targets, _ in tqdm(loader, desc="  FDS stats", leave=False):
                imgs, targets = imgs.to(device), targets.to(device)
                # In eval mode model returns just out, so call with training trick
                model.train()
                out_tuple = model(imgs, targets, epoch)
                model.eval()
                if isinstance(out_tuple, tuple):
                    _, feat = out_tuple
                else:
                    # fallback: extract features manually
                    feat = model.encoder(imgs)
                    feat = feat.view(feat.size(0), -1)
                encodings.append(feat.detach().cpu())
                all_labels.append(targets.cpu())

        enc_t = torch.cat(encodings, dim=0).to(device)
        lbl_t = torch.cat(all_labels, dim=0).to(device)
        model.fds.update_last_epoch_stats(epoch)
        model.fds.update_running_stats(enc_t, lbl_t, epoch)
        model.train()

    return meter.avg


@torch.no_grad()
def evaluate(loader, model, device, train_labels, prefix='Val'):
    model.eval()
    preds_all, labels_all = [], []
    l1_meter  = AverageMeter()
    mse_meter = AverageMeter()

    for imgs, targets, _ in tqdm(loader, desc=f"  {prefix}", leave=False):
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        # In eval mode, model always returns just the tensor (not a tuple)
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        l1_meter.update(nn.L1Loss()(out, targets).item(), imgs.size(0))
        mse_meter.update(nn.MSELoss()(out, targets).item(), imgs.size(0))

        preds_all.extend(out.cpu().numpy())
        labels_all.extend(targets.cpu().numpy())

    shots = shot_metrics(preds_all, labels_all, train_labels)
    l1_all = np.abs(np.array(preds_all) - np.array(labels_all))
    g = float(gmean(l1_all + 1e-8))

    print(f"  [{prefix}] MAE={l1_meter.avg:.3f}°C  RMSE={mse_meter.avg**0.5:.3f}°C  G-Mean={g:.3f}")
    print(f"         Many-shot MAE={shots['many']['l1']:.3f}  "
          f"Median-shot MAE={shots['median']['l1']:.3f}  "
          f"Low-shot MAE={shots['low']['l1']:.3f}")

    return {
        'mae':    l1_meter.avg,
        'rmse':   mse_meter.avg ** 0.5,
        'gmean':  g,
        'shots':  shots,
        'preds':  preds_all,
        'labels': labels_all,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)

    device = torch.device('cpu')
    print(f"[i] Device: {device}")
    print(f"[i] Experiment: {args.experiment}")

    out_dir = Path(args.out_dir) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(out_dir / 'train.log'),
            logging.StreamHandler(),
        ]
    )

    # ── Data ──────────────────────────────────────────────────────────────
    print(f"[→] Loading CSV: {args.data_csv}")
    df = pd.read_csv(args.data_csv)
    df_train = df[df['split'] == 'train']
    df_val   = df[df['split'] == 'val']
    df_test  = df[df['split'] == 'test']
    train_labels = df_train['temperature'].values

    print(f"    Train={len(df_train)}  Val={len(df_val)}  Test={len(df_test)}")

    train_ds = SkyFinder(df_train, img_size=args.img_size, split='train',
                         reweight=args.reweight,
                         lds=args.lds, lds_kernel=args.lds_kernel,
                         lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    val_ds   = SkyFinder(df_val,   img_size=args.img_size, split='val')
    test_ds  = SkyFinder(df_test,  img_size=args.img_size, split='test')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=False)

    # ── Model ─────────────────────────────────────────────────────────────
    fds_module = None
    if args.fds:
        fds_module = FDS(
            feature_dim=512,
            bucket_num=96,
            bucket_start=0,
            start_update=args.start_update,
            start_smooth=args.start_smooth,
            kernel=args.fds_kernel,
            ks=args.fds_ks,
            sigma=args.fds_sigma,
            momentum=args.fds_mmt,
        ).to(device)

    model = ResNet18Regressor(pretrained=True, fds_module=fds_module).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn   = get_loss_fn(args.loss)

    # ── Training loop ──────────────────────────────────────────────────────
    best_mae  = float('inf')
    history   = []

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn,
                                     device, args, epoch)
        scheduler.step()

        val_metrics = evaluate(val_loader, model, device, train_labels, prefix='Val')
        elapsed = time.time() - t0

        row = {
            'epoch':      epoch,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items() if k not in ('preds', 'labels', 'shots')},
            'elapsed_s':  elapsed,
        }
        history.append(row)
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_MAE={val_metrics['mae']:.3f}°C | {elapsed:.1f}s")

        if val_metrics['mae'] < best_mae:
            best_mae = val_metrics['mae']
            torch.save({'epoch': epoch, 'state': model.state_dict(),
                        'mae': best_mae}, out_dir / 'best.pth')
            print(f"    ★ New best MAE: {best_mae:.3f}°C  (saved)")

    # ── Test ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Testing best checkpoint...")
    ckpt = torch.load(out_dir / 'best.pth', map_location=device)
    model.load_state_dict(ckpt['state'])
    test_metrics = evaluate(test_loader, model, device, train_labels, prefix='Test')

    results = {
        'experiment': args.experiment,
        'args':       vars(args),
        'best_val_mae': best_mae,
        'test': {k: v for k, v in test_metrics.items() if k not in ('preds', 'labels')},
    }
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({
        'pred':  test_metrics['preds'],
        'label': test_metrics['labels'],
    }).to_csv(out_dir / 'test_predictions.csv', index=False)

    pd.DataFrame(history).to_csv(out_dir / 'history.csv', index=False)
    print(f"\n[✓] Results saved to {out_dir}")
    print(f"    Test MAE: {test_metrics['mae']:.3f}°C  RMSE: {test_metrics['rmse']:.3f}°C")


if __name__ == '__main__':
    main()
