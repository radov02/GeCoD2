"""Print the density-guided detection gate (DensityGuidedDetection.gamma)
magnitudes from saved checkpoints.
"""
import sys
import glob
import torch


def state_dict_of(ckpt):
    """weights dict from {'model': sd}, {'state_dict': sd} or a raw sd."""
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def inspect(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = state_dict_of(ckpt)
    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    # gamma key is '<maybe module.>density_guide.gamma'
    gkeys = [k for k in sd if k.endswith('density_guide.gamma')]
    name = path.rsplit('/', 1)[-1]
    if not gkeys:
        has_guide = any('density_guide' in k for k in sd)
        reason = ('density_guide present but no gamma key (unexpected)'
                  if has_guide else
                  'no density_guide module -> densg was OFF (or pre-densg checkpoint)')
        print(f"[--] {name} (epoch {epoch}): {reason}")
        return
    g = sd[gkeys[0]].detach().float().flatten()
    n = g.numel()
    abs_g = g.abs()
    # gate inits at 0, |g|<1e-3 counts as near-zero
    near0 = int((abs_g < 1e-3).sum())
    print(f"[OK] {name} (epoch {epoch})")
    print(f"       channels={n}  mean|g|={abs_g.mean():.4e}  max|g|={abs_g.max():.4e}  "
          f"L2={g.norm():.4e}")
    print(f"       channels with |g|<1e-3: {near0}/{n} ({100*near0/n:.0f}%)  "
          f"-> {'INERT gate' if abs_g.mean() < 1e-3 else 'ACTIVE gate (sign of effect needs the ON/OFF ablation)'}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    paths = []
    for a in args:
        paths.extend(sorted(glob.glob(a)) or [a])
    for p in paths:
        try:
            inspect(p)
        except Exception as e:
            print(f"[ER] {p}: {type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
