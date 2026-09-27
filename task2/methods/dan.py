# ══════════════════════════════════════════════════════════════════════════
# task2/methods/dan.py — multi-kernel RBF MMD used by the DAN method
# ══════════════════════════════════════════════════════════════════════════
import torch


# ══════════════════════════════════════════════════════════════════════════
# FILE: task2/methods/dan.py  — MMD (used by DAN)
# ══════════════════════════════════════════════════════════════════════════

def multi_kernel_rbf_mmd2(feat_s: torch.Tensor, feat_t: torch.Tensor, kernel_mults=(0.5, 1.0, 2.0),
                           eps: float = 1e-8) -> torch.Tensor:
    """Biased MMD^2 estimate with a sum-of-three-RBF-kernels, bandwidths set
    to {0.5,1,2} x the median pairwise squared distance of the CURRENT
    combined (source+target) batch, evaluated via the kernel trick (no
    explicit feature map is ever built)."""
    ns, nt = feat_s.size(0), feat_t.size(0)
    z = torch.cat([feat_s, feat_t], dim=0)  # (ns+nt, d)
    # pairwise squared distances on the combined batch
    sq_dists = torch.cdist(z, z, p=2).pow(2)  # (n, n)

    n = z.size(0)
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    median_sq_dist = sq_dists[off_diag_mask].median().clamp(min=eps)

    kernel_sum = torch.zeros_like(sq_dists)
    for mult in kernel_mults:
        bandwidth = (median_sq_dist * mult).clamp(min=eps)
        kernel_sum = kernel_sum + torch.exp(-sq_dists / bandwidth)

    k_ss = kernel_sum[:ns, :ns]
    k_tt = kernel_sum[ns:, ns:]
    k_st = kernel_sum[:ns, ns:]

    mmd2 = k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean()
    return mmd2.clamp(min=0.0)  # numerical safety: MMD^2 can't be negative

