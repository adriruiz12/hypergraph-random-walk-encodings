"""
eompp.node_training
===================

Training utilities for the node-classification experiments (Benchmark and
Complementarity): tensor conversion, the shuffled-chi controls, evaluation,
and the full-batch training loop with early stopping on validation accuracy.
"""

import numpy as np
import torch
import torch.nn.functional as F

from eompp.metrics import macro_f1


# --------------------------------------------------------------------------
# Dataset dict -> torch tensors (done once per dataset).
# --------------------------------------------------------------------------
def to_tensors(data, device):
    """Turn the numpy/scipy data dict into torch tensors on `device`."""

    edge_index = torch.tensor(data["edge_index"], dtype=torch.long)

    # clique-expansion degree (distinct neighbours), for GCN/GIN and EO mean
    deg = torch.zeros(data["n_nodes"], dtype=torch.float32)
    deg = deg.index_add(0,edge_index[1],torch.ones(edge_index.size(1)))

    # incidence matrix as a torch sparse tensor (for AllDeepSets)
    B = data["incidence"].tocoo()
    idx = torch.tensor(np.vstack([B.row, B.col]), dtype=torch.long)
    val = torch.tensor(B.data, dtype=torch.float32)
    torch.sparse.check_sparse_tensor_invariants.disable()
    incidence = torch.sparse_coo_tensor(idx, val, B.shape).coalesce()
    incidence_t = incidence.transpose(0, 1).coalesce()
    he_deg = torch.tensor(np.asarray(data["incidence"].sum(axis=0)).ravel(),
                          dtype=torch.float32)
    node_deg_hg = torch.tensor(data["d_h"], dtype=torch.float32)

    batch = dict(
        x=torch.tensor(data["x"], dtype=torch.float32),
        y=torch.tensor(data["y"], dtype=torch.long),
        edge_index=edge_index,
        deg=deg,
        incidence=incidence,
        incidence_t=incidence_t,
        he_deg=he_deg,
        node_deg_hg=node_deg_hg,
    )
    for k, v in data["p"].items():                 # p_en, p_ee, p_we
        batch["p_" + k.lower()] = torch.tensor(v, dtype=torch.float32)
    for k, v in data["chi"].items():               # chi_ee, chi_we
        batch["chi_" + k.lower()] = torch.tensor(v, dtype=torch.float32)
    return {k: v.to(device) for k, v in batch.items()}

def shuffled_chi(batch, key, seed):
    """Return a copy of `batch` with the rows of `key` globally permuted.

    Used by the shuffled-chi negative controls (variants E and H); `key` is
    "chi_ee" or "chi_we".
    """

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch[key].size(0), generator=g)
    new = dict(batch)
    new[key] = batch[key][perm.to(batch[key].device)]
    return new


# --------------------------------------------------------------------------
# Evaluation.
# --------------------------------------------------------------------------
def evaluate(model, batch, mask, n_classes):
    with torch.no_grad():
        model.eval()
        pred = model(batch).argmax(dim=-1)
        y = batch["y"]
        acc = float((pred[mask] == y[mask]).float().mean())
        f1 = macro_f1(y[mask].cpu(), pred[mask].cpu(), n_classes)
        return acc, f1


# --------------------------------------------------------------------------
# Training (full-batch node classification, early stopping on val acc).
# --------------------------------------------------------------------------
def train_one(model, batch, train_mask, val_mask, n_classes, cfg):
    """Train with Adam + early stopping on validation accuracy.

    Returns the model loaded with the best-validation checkpoint.
    """

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    best_val, best_state, wait = -1.0, None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        opt.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits[train_mask], batch["y"][train_mask])
        loss.backward()
        opt.step()

        val_acc, _ = evaluate(model, batch, val_mask, n_classes)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model
