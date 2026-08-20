"""config_<dataset>.sh knobs applied as argparse defaults so a bare python run
matches the shell recipe. CLI flags still win. GECO2_RUN_CONFIG=skip opts out,
a file path points at a different config."""

import os
import re
from pathlib import Path

# env name in config_<dataset>.sh -> argparse dest, training knobs only
ENV_TO_DEST = {
    # optimizer / schedule / early stop
    'LR': 'lr',
    'DEPTH_FUSE_LR': 'depth_fuse_lr',
    'ZS_PROTO_LR': 'zs_proto_lr',
    'BACKBONE_LR': 'backbone_lr',
    'WEIGHT_DECAY': 'weight_decay',
    'DROPOUT': 'dropout',
    'MAX_GRAD_NORM': 'max_grad_norm',
    'AUX_WEIGHT': 'aux_weight',
    'TRAIN_TILING_P': 'tiling_p',
    'LR_WARMUP_EPOCHS': 'lr_warmup_epochs',
    'LR_WARMUP_START_FACTOR': 'lr_warmup_start_factor',
    'AUX_LR_WARMUP_EPOCHS': 'aux_lr_warmup_epochs',
    'SEED': 'seed',
    'NUM_WORKERS': 'num_workers',
    'BATCH_SIZE': 'batch_size',
    'PLATEAU_PATIENCE': 'plateau_patience',
    'REDUCE_LR_PATIENCE': 'reduce_lr_patience',
    'REDUCE_LR_FACTOR': 'reduce_lr_factor',
    'SPIKE_PATIENCE': 'spike_patience',
    'SPIKE_RATIO': 'spike_ratio',
    'SELECT_RMSE_WEIGHT': 'select_rmse_weight',
    'GOOD_SELECT': 'good_select',
    'GOOD_PLATEAU_PATIENCE': 'good_plateau_patience',
    'PROBE_ADAPTIVE': 'probe_adaptive',
    'PROBE_EPOCHS': 'probe_epochs',
    'PROBE_PLATEAU_PATIENCE': 'probe_plateau_patience',
    'PROBE_LR_PATIENCE': 'probe_lr_patience',
    'UNFREEZE_LAST_HIERA': 'unfreeze_last_hiera',
    'KERNEL_DIM': 'kernel_dim',
    # depth architecture
    'DEPTH_KERNEL_SIZE': 'depth_kernel_size',
    'DEPTH_FUSE_IDENTITY_INIT': 'depth_fuse_identity_init',
    'DEPTH_FEAT_NORM': 'depth_feat_norm',
    'DEPTH_FEAT_NORM_GROUPS': 'depth_feat_norm_groups',
    'DEPTH_FEAT_CHANNELS': 'depth_feat_channels',
    'DEPTH_TARGET_SIZE': 'depth_target_size',
    'DINO_INPUT_SIZE': 'dino_input_size',
    'DEPTH_ADAPT': 'depth_adapt',
    'DEPTH_CUES': 'depth_cues',
    'DEPTH_ADAPT_INIT': 'depth_adapt_init',
    'DEPTH_ADAPT_MASKED_CONV': 'depth_adapt_masked_conv',
    'DEPTH_HIRES_FUSION': 'depth_hires_fusion',
    'DEPTH_HIRES_NORM': 'depth_hires_norm',
    'SEP_HIERA_INPUT': 'sep_hiera_input',
    'SEP_HIERA_FULLRES': 'sep_hiera_fullres',
    'SEP_HIERA_PER_LEVEL_GATE': 'sep_hiera_per_level_gate',
    'FFM_NORM': 'ffm_norm',
    # density head
    'USE_DENSITY': 'use_density',
    'DENSITY_HEAD_TYPE': 'density_head_type',
    'DENSITY_ADAPTIVE_SIGMA': 'density_adaptive_sigma',
    'DENSITY_ABS_COUNT_WEIGHT': 'density_abs_count_weight',
    'DENSITY_LR': 'density_lr',
    'DENSITY_WEIGHT': 'density_weight',
    'DENSITY_LOSS_TYPE': 'density_loss_type',
    'DENSITY_GUIDED': 'density_guided',
    'DENSITY_SIGMA': 'density_sigma',
    'DENSITY_SIGMA_K': 'density_sigma_k',
    'DENSITY_SIGMA_BETA': 'density_sigma_beta',
    'DENSITY_SIGMA_MIN': 'density_sigma_min',
    'DENSITY_SIGMA_MAX': 'density_sigma_max',
}

# USE_DEPTHFEATS translates to --depth_source: 1 = decoder tap, 0 = scalar disparity
_USE_DEPTHFEATS_TO_SOURCE = {'1': 'decoder', '0': 'scalar'}

_EXPORT_RE = re.compile(r'^\s*export\s+([A-Z0-9_]+)=(.*)$')
_ARITH_RE = re.compile(r'^\$\(\(\s*(.+?)\s*\)\)$')
_VARREF_RE = re.compile(r'\$\{?([A-Z0-9_]+)\}?')
_TERNARY_RE = re.compile(r'^([^?:]+)\?([^?:]+):([^?:]+)$')
_ARITH_SAFE_RE = re.compile(r'^(?:\s|\d|[()+\-*/%]|<=|>=|==|!=|<|>|\bif\b|\belse\b)+$')


def _eval_arith(raw, env):
    """Evaluate a bash '$(( ... ))' export value for the subset the configs use
    (int literals, config-var refs, arithmetic, one ternary). None if outside that."""
    m = _ARITH_RE.match(raw)
    if not m:
        return None
    # bash arithmetic refs vars both as $NAME/${NAME} and BARE (uppercase) names
    expr = _VARREF_RE.sub(lambda v: env.get(v.group(1), ''), m.group(1))
    expr = re.sub(r'\b[A-Z][A-Z0-9_]*\b', lambda v: env.get(v.group(0), ''), expr)
    t = _TERNARY_RE.match(expr)
    if t:
        expr = f"({t.group(2)}) if ({t.group(1)}) else ({t.group(3)})"
    if not _ARITH_SAFE_RE.match(expr):
        return None
    try:
        return str(int(eval(expr, {'__builtins__': {}}, {})))  # expr already passed _ARITH_SAFE_RE
    except Exception:
        return None


def _is_rank0():
    return os.environ.get('RANK', '0') in ('', '0') and \
        os.environ.get('LOCAL_RANK', '0') in ('', '0')


def parse_config_exports(path):
    """'export NAME=value' lines of a config .sh -> dict of strings.
    '$(( ... ))' arithmetic is evaluated, other '$' values are skipped."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = _EXPORT_RE.match(line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2).strip()
        # strip a trailing inline comment (the configs always put a space before #)
        cut = raw.find(' #')
        if cut != -1:
            raw = raw[:cut].rstrip()
        if raw.startswith(('"', "'")) and raw.endswith(raw[0]) and len(raw) >= 2:
            raw = raw[1:-1]
        if '$' in raw:
            raw = _eval_arith(raw, out)
            if raw is None:
                continue
        out[name] = raw
    return out


def apply_run_config_defaults(parser, dataset, exclude=()):
    """Apply training/config_<dataset>.sh constants as parser defaults.
    exclude: dest names a caller pins itself (sep_hiera's depth_feat_channels=3)."""
    mode = os.environ.get('GECO2_RUN_CONFIG', '')
    if mode in ('skip', '0'):
        return {}
    if mode and mode not in ('1',):
        cfg_path = Path(mode)
    elif os.environ.get('CONFIG_FILE'):
        cfg_path = Path(os.environ['CONFIG_FILE'])
    else:
        cfg_path = Path(__file__).resolve().parents[1] / 'training' / f'config_{dataset}.sh'
    if not cfg_path.is_file():
        if _is_rank0():
            print(f"[run-config] no config at {cfg_path} -- keeping parser defaults "
                  f"(set GECO2_RUN_CONFIG=/path/to/config_{dataset}.sh to point at one)")
        return {}

    exports = parse_config_exports(cfg_path)
    actions = {a.dest: a for a in parser._actions}
    applied, skipped = {}, []
    for env, dest in ENV_TO_DEST.items():
        if env not in exports or dest in exclude:
            continue
        if dest not in actions:
            skipped.append(dest)
            continue
        act = actions[dest]
        raw = exports[env]
        try:
            conv = act.type(raw) if callable(act.type) else raw
        except (TypeError, ValueError):
            skipped.append(f"{dest}(bad value {raw!r})")
            continue
        applied[dest] = conv
    if 'USE_DEPTHFEATS' in exports and 'depth_source' not in exclude \
            and 'depth_source' in actions:
        src = _USE_DEPTHFEATS_TO_SOURCE.get(exports['USE_DEPTHFEATS'])
        if src:
            applied['depth_source'] = src
    # DEPTH_CHECKPOINT is read via the env (Depther.infer_depth), not argparse
    if exports.get('DEPTH_CHECKPOINT'):
        os.environ.setdefault('DEPTH_CHECKPOINT', exports['DEPTH_CHECKPOINT'])

    parser.set_defaults(**applied)
    if _is_rank0():
        kv = ', '.join(f"{k}={v}" for k, v in sorted(applied.items()))
        print(f"[run-config] {len(applied)} defaults from {cfg_path.name} "
              f"(CLI flags win): {kv}")
        if skipped:
            print(f"[run-config] skipped (no matching flag in this parser): "
                  f"{', '.join(sorted(set(skipped)))}")
    return applied
