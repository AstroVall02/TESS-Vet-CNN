import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_setup import TESSDataset, TESSTransform


def make_splits(csv_path, data_dir, aux_cols, val_frac=0.125):
    #Same test split as create_dataloaders
    df = pd.read_csv(csv_path)
    df = df[(df["Period (days)"] > 0) & (df["Duration_hrs"] > 0) & (df["Depth_ppm"] > -1)].reset_index(drop=True)
    df["log_period"] = np.log(df["Period (days)"])
    df["log_depth"] = np.log1p(df["Depth_ppm"])
    check = list(set(list(aux_cols) + ["log_period", "log_depth"]))
    df = df[np.isfinite(df[check]).all(axis=1)].reset_index(drop=True)
    have = {Path(f).stem for f in os.listdir(data_dir) if f.endswith(".npy")}
    df = df[df["TIC_ID"].apply(lambda t: str(int(t)) in have or f"TIC_{int(t)}" in have)].reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["final_label"])
    tr, va = train_test_split(tr, test_size=val_frac, random_state=42, stratify=tr["final_label"])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def _norm(X):
    return (X - X.mean(dim=2, keepdim=True)) / (X.std(dim=2, keepdim=True) + 1e-8)


@torch.no_grad()
def _predict(model, loader, device, loss_fn):
    model.eval()
    probs, ys, total = [], [], 0.0
    for X, y, aux in loader:
        X, y, aux = _norm(X.to(device)), y.float().unsqueeze(1).to(device), aux.to(device)
        logits = model(X, aux)
        total += loss_fn(logits, y).item() * len(y)
        probs.append(torch.sigmoid(logits).view(-1).cpu().numpy())
        ys.append(y.view(-1).cpu().numpy())
    return total / len(loader.dataset), np.concatenate(probs), np.concatenate(ys)


def run_protocol(csv_path, data_dir, aux_cols, model_fn, device, set_seed, tag,
                 seeds=(1, 2, 3, 4, 5), batch_size=32, epochs=100, out_dir="."):
    tr, va, te = make_splits(csv_path, data_dir, aux_cols)
    stats = {c: (tr[c].mean(), tr[c].std()) for c in aux_cols}
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    np.save(f"{out_dir}/test_tic_ids.npy", te["TIC_ID"].astype(int).values)

    ds_val = TESSDataset(va, data_dir, transform=None, aux_cols=aux_cols, aux_stats=stats)
    ds_test = TESSDataset(te, data_dir, transform=None, aux_cols=aux_cols, aux_stats=stats)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    rows = []
    for seed in seeds:
        set_seed(seed)
        ds_tr = TESSDataset(tr, data_dir, transform=TESSTransform(seed=seed), aux_cols=aux_cols, aux_stats=stats)
        g = torch.Generator(); g.manual_seed(seed)
        tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=g)

        model = model_fn(len(aux_cols)).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

        best_loss, best_w, best_ep = float("inf"), None, 0
        for ep in range(epochs):
            model.train()
            for X, y, aux in tr_loader:
                X, y, aux = _norm(X.to(device)), y.float().unsqueeze(1).to(device), aux.to(device)
                opt.zero_grad()
                loss_fn(model(X, aux), y).backward()
                opt.step()
            vloss, vp, vy = _predict(model, val_loader, device, loss_fn)      #validation
            sched.step(vloss)
            if vloss < best_loss:
                best_loss, best_ep, best_w = vloss, ep, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_w)

        _, vp, vy = _predict(model, val_loader, device, loss_fn)
        fpr, tpr, thr = roc_curve(vy, vp)
        thr_val = float(thr[np.argmax(tpr - fpr)])                            # threshold chosen on validation
        tloss, tp, ty = _predict(model, test_loader, device, loss_fn)          # test once per seed
        np.save(f"{out_dir}/cnn_probs_{tag}_seed{seed}.npy", tp)
        np.save(f"{out_dir}/cnn_valprobs_{tag}_seed{seed}.npy", vp)
        torch.save(best_w, f"{out_dir}/model_{tag}_seed_{seed}.pt")
        rows.append(dict(seed=seed, best_epoch=best_ep, val_loss=best_loss,
                         val_acc=float(100 * ((vp > 0.5) == vy).mean()), test_loss=tloss,
                         test_acc_0p5=float(100 * ((tp > 0.5) == ty).mean()),
                         thr_from_val=thr_val, test_acc_val_thr=float(100 * ((tp > thr_val) == ty).mean())))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})

    res = pd.DataFrame(rows)
    for c in ("test_acc_0p5", "test_acc_val_thr"):
        print(f"{c}: {res[c].mean():.3f} ± {res[c].std():.3f}")
    return res
