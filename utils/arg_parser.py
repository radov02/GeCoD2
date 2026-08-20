import argparse


def apply_lr_scaling(args, rank=0):
    """Scale args.lr (and backbone_lr) in place by batch_size / reference batch size."""
    if not getattr(args, 'lr_scale_with_batch_size', False):
        return
    ref = max(1, int(args.lr_scale_reference_batch_size))
    scale = args.batch_size / ref
    if scale == 1.0:
        return
    old_lr = args.lr
    old_backbone_lr = args.backbone_lr
    args.lr = old_lr * scale
    if old_backbone_lr > 0:
        args.backbone_lr = old_backbone_lr * scale
    if rank == 0:
        print(
            f"[lr-scale] batch_size={args.batch_size}, reference={ref}, scale={scale:g} -> "
            f"lr {old_lr:.3e} -> {args.lr:.3e}, "
            f"backbone_lr {old_backbone_lr:.3e} -> {args.backbone_lr:.3e}",
            flush=True,
        )


def get_argparser():

    parser = argparse.ArgumentParser("GECO2", add_help=False)

    parser.add_argument('--model_name', default='GECO2FSCD', type=str)
    parser.add_argument('--model_name_resumed', default='GECO2FSCD', type=str)
    parser.add_argument(
        '--data_path',
        default='/storage/datasets/fsc147',
        type=str
    )
    parser.add_argument(
        '--model_path',
        default='/d/hpc/projects/FRI/pelhanj/CNT_SAM2/models/',
        type=str
    )
    parser.add_argument('--dataset', default='fsc147', type=str)
    parser.add_argument('--backbone', default='SAM', type=str)
    parser.add_argument('--reduction', default=16, type=int)
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--num_enc_layers', default=3, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--kernel_dim', default=1, type=int)  # official GeCo2 value, forward forces 1 anyway
    parser.add_argument('--num_objects', default=3, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--backbone_lr', default=0, type=float)
    parser.add_argument('--density_lr', default=1e-4, type=float,
                        help='LR for the from-scratch DensityHead optimizer group (only when --use_density>0).')
    parser.add_argument('--lr_drop', default=50, type=int)
    parser.add_argument(
        '--lr_scale_with_batch_size', action='store_true',
        help='Linearly scale --lr (and --backbone_lr if non-zero) by batch_size / --lr_scale_reference_batch_size.'
    )
    parser.add_argument(
        '--lr_scale_reference_batch_size', default=12, type=int,
        help='Batch size at which the current --lr value is considered tuned.'
    )
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--seed', default=42, type=int,
                        help='Per-run RNG seed (torch/cuda/numpy/random), offset by DDP rank.')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--max_grad_norm', default=0.1, type=float)
    parser.add_argument('--aux_weight', default=0.3, type=float)
    parser.add_argument('--tiling_p', default=0.5, type=float)
    parser.add_argument('--crop_p', default=0.0, type=float,
                        help='Probability of the zoom-in square-crop augmentation (train only); 0 disables.')
    parser.add_argument('--crop_min_px', default=384, type=int,
                        help='Min square-crop side (original px) for the --crop_p aug.')
    parser.add_argument('--crop_max_px', default=640, type=int,
                        help='Max square-crop side (original px) for the --crop_p aug.')
    parser.add_argument('--zero_shot', action='store_true',
                        help='exemplar-free counting via learned prototype tokens')
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--pre_norm', action='store_true',
                        help='unused, kept so old commands still parse')
    parser.add_argument('--encode', action='store_true')
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--cost_class", default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument("--cost_bbox", default=1, type=float, help="L1 box coefficient in the matching cost")
    parser.add_argument("--cost_giou", default=2, type=float, help="giou box coefficient in the matching cost")
    parser.add_argument("--focal_alpha", default=0.25, type=float)
    parser.add_argument("--model_name_resume_from", default='base_3_shot_softmax1', type=str)
    parser.add_argument('--init_from_pretrained', default=None, type=str,
                        help='Pretrained CNT .pth to initialize from (strict=False; bare filename resolved against --model_path).')

    # LR warmup
    parser.add_argument(
        '--lr_warmup_epochs', default=3, type=int,
        help='Linearly ramp LR over the first N epochs; 0 or 1 disables.'
    )
    parser.add_argument(
        '--lr_warmup_start_factor', default=0.1, type=float,
        help='Starting LR multiplier at epoch 1; ramps linearly to 1.0 at epoch --lr_warmup_epochs.'
    )
    parser.add_argument(
        '--aux_lr_warmup_epochs', default=0, type=int,
        help='Separate warmup length for the high-LR aux groups (depth-fusion + density); 0 = warm with the body.'
    )

    # ReduceLROnPlateau
    parser.add_argument('--reduce_lr_patience', default=5, type=int,
                        help='Epochs with no improvement in the val selection metric before LR is reduced.')
    parser.add_argument('--reduce_lr_factor', default=0.5, type=float,
                        help='Factor by which LR is reduced (new_lr = lr * factor)')
    parser.add_argument('--select_rmse_weight', default=0.0, type=float,
                        help='Blend weight w in the val selection metric val_select = w*val_RMSE + (1-w)*val_MAE; 0 = pure val MAE.')

    # Spike-based early stopping
    parser.add_argument('--spike_patience', default=2, type=int,
                        help='Stop after this many consecutive val RMSE spikes')
    parser.add_argument('--spike_ratio', default=2.0, type=float,
                        help='Val RMSE is a "spike" when it exceeds spike_ratio * best val RMSE')
    parser.add_argument('--spike_recover_max', default=2, type=int,
                        help='Warm restarts attempted before a val-RMSE spike counts toward the stop; 0 disables.')
    parser.add_argument('--spike_lr_decay', default=0.5, type=float,
                        help='LR multiplier applied at each divergence recovery.')
    parser.add_argument('--plateau_patience', default=0, type=int,
                        help='Stop after this many consecutive epochs with no val_select or val NAE improvement; 0 disables.')
    parser.add_argument('--good_select', default=0.0, type=float,
                        help='val_select target that switches the plateau stop to --good_plateau_patience; 0 disables.')
    parser.add_argument('--good_plateau_patience', default=2, type=int,
                        help='Plateau patience used once --good_select is reached (capped at --plateau_patience).')

    # probe-then-finetune
    parser.add_argument('--probe_epochs', default=0, type=int,
                        help='For the first N epochs only the depth-fusion (and density) params receive updates; requires --use_depth>0, 0 = off.')
    parser.add_argument('--depth_fuse_lr', default=0.0, type=float,
                        help='Separate LR for the depth-fusion params; 0 = use --lr.')
    parser.add_argument('--zs_proto_lr', default=0.0, type=float,
                        help='Separate LR for the zero-shot prototype tokens; 0 = use --lr.')
    parser.add_argument('--probe_adaptive', default=0, type=int,
                        help='1 = probe runs until val_select stalls for --probe_plateau_patience epochs instead of a fixed --probe_epochs.')
    parser.add_argument('--probe_only', default=0, type=int,
                        help='1 = stop training when the probe ends instead of unfreezing the body.')
    parser.add_argument('--probe_plateau_patience', default=5, type=int,
                        help='Adaptive probe: consecutive non-improving val_select epochs that end the probe (hard cap epochs//2).')
    parser.add_argument('--probe_lr_patience', default=2, type=int,
                        help='Adaptive probe: stalled epochs after which the depth-fusion group LR is cut by --reduce_lr_factor; 0 = no in-probe cuts.')
    parser.add_argument('--depth_zero_ablation', default=0, type=int,
                        help='Inference only: 1 = re-apply the identity init to the depth-fusion params and evaluate (_zerodepth suffix).')

    # depth fusion (these have to match between training and inference)
    parser.add_argument('--use_depth', default=0, type=int,
                        help='Depth-fusion mode: 0=off, 1=conv-add, 2=conv-concat, 3=input-level concat, 4=separate Hiera, 5=ffm (BiSeNet FFM).')
    parser.add_argument('--depth_feat_channels', default=16, type=int,
                        help='Output channels of the depth adapter conv. '
                             'Must be 3 when use_depth=4 (feeds a Hiera).')
    parser.add_argument('--depth_kernel_size', default=3, type=int,
                        help='kernel size of the depth-fusion convs in modes 1/2/5')
    parser.add_argument('--depth_fuse_identity_init', default=0, type=int,
                        help='1 = initialise the depth fusion as identity (zero depth contribution at init); 0 = normal random init.')
    parser.add_argument('--depth_feat_norm', default='group', type=str, choices=['none', 'group'],
                        help="norm on the adapted depth features: 'group' = masked GroupNorm, 'none'")
    parser.add_argument('--depth_feat_norm_groups', default=0, type=int,
                        help='num_groups for --depth_feat_norm=group; 0 = auto.')
    parser.add_argument('--depth_target_size', default=0, type=int,
                        help='resolution (px) depth is predicted and fused at, 0 = auto (image_size*4//reduction)')
    parser.add_argument('--depth_source', default='decoder', type=str, choices=['scalar', 'decoder'],
                        help="depth features for modes 1/2: 'scalar' = 1-ch disparity, 'decoder' = DPT path_1")
    parser.add_argument('--depth_adapt', default='conv', type=str, choices=['linear', 'conv'],
                        help="depth adapter for modes 1/2/5: 'linear' = 1x1 conv, 'conv' = 3x3 conv stack")
    parser.add_argument('--depth_cues', default='learned', type=str, choices=['learned', 'fixed'],
                        help="conv-adapter input: 'learned' = raw depth, 'fixed' = adds Sobel/Laplacian channels")
    parser.add_argument('--depth_adapt_init', default='orthogonal', type=str,
                        choices=['orthogonal', 'default'],
                        help="Weight init of the conv depth adapter: 'orthogonal' or the plain PyTorch 'default'.")
    parser.add_argument('--depth_adapt_masked_conv', default=1, type=int,
                        help='1 = masked conv in the depth adapter (windows renormalized by valid-pixel fraction); 0 = plain Conv2d.')
    parser.add_argument('--sep_hiera_input', default='cues', type=str, choices=['cues', 'replicate'],
                        help="Input to the mode-4 depth Hiera: 'cues' = [disparity, Sobel magnitude, Laplacian], 'replicate' = disparity in all 3 channels.")
    parser.add_argument('--sep_hiera_fullres', default=1, type=int,
                        help='mode 4: 1 = build depth cues at the Hiera image_size, 0 = at --depth_target_size then upsample')
    parser.add_argument('--sep_hiera_per_level_gate', default=1, type=int,
                        help='mode 4 with identity init: 1 = depth gate per fused pyramid level, 0 = one shared scalar gate')
    parser.add_argument('--ffm_norm', default='group', type=str, choices=['group', 'batch'],
                        help="norm inside the mode-5 FeatureFusionModule: 'group' or 'batch'")
    parser.add_argument('--dino_input_size', default=0, type=int,
                        help='resolution (px) the DAv2 DINOv2 encoder runs at, 0 = max(518, depth_target_size)')
    parser.add_argument('--depth_hires_fusion', default=0, type=int,
                        help='1 = also inject full-res depth at the input in modes 1/2/5 (identity-init 1x1 conv)')
    parser.add_argument('--depth_hires_norm', default=1, type=int,
                        help='1 = masked per-image GroupNorm on the full-res depth in the hires path')
    parser.add_argument('--vis_every', default=0, type=int,
                        help='Every N epochs save a training depth-cue visualization for one fixed val sample; 0 = off.')

    # pre-computed depth-map cache
    parser.add_argument('--depthmaps_dir', default='', type=str,
                        help='Directory of the pre-computed depth-map cache (<id>.jpg, grayscale, near=bright); empty = in-model prediction.')
    # PCA-k cached decoder features
    parser.add_argument('--depthfeats_dir', default='', type=str,
                        help='Directory of the PCA-k cached decoder features (<id>.npy + pca_basis.npz); empty = off.')
    parser.add_argument('--use_available_depthfeats', action='store_true',
                        help='With --depthfeats_dir: require the feature cache to exist and never generate.')
    parser.add_argument('--decoder_feat_channels_PCA', default=16, type=int,
                        help='channels of the decoder-feature cache: >0 = PCA components, -1 = raw path_1 (must match at inference)')
    parser.add_argument('--use_available_depthmaps', action='store_true',
                        help='With --depthmaps_dir: require the maps to already exist and never generate.')

    # bounded backbone unfreeze
    parser.add_argument('--unfreeze_last_hiera', default=0, type=int,
                        help='N>0 = train the last N stages of the RGB Hiera trunk at --backbone_lr; 0 = fully frozen.')

    # density-regression head
    parser.add_argument('--use_density', default=0, type=int,
                        help='1 = add a DensityHead and count by integrating its map (detection branch kept for AP); 0 = baseline.')
    parser.add_argument('--density_weight', default=1.0, type=float,
                        help='Weight of the density loss added to the detection losses; 0 = pure detection.')
    parser.add_argument('--density_head_type', default='simple', type=str,
                        choices=['simple', 'fpn'],
                        help="'simple' = shallow 3-conv DensityHead at 512x512; 'fpn' = deeper multi-scale DensityDecoder at 1024x1024.")
    parser.add_argument('--density_guided', default=0, type=int,
                        help='1 = use the predicted density map as a spatial prior on the detection features; requires --use_density>0.')
    parser.add_argument('--density_detach', default=0, type=int,
                        help='1 = feed the density head a detached trunk feature so the density loss cannot perturb detection; tag _ddet.')
    parser.add_argument('--density_loss_type', default='dmcount', type=str,
                        choices=['dmcount', 'mse'],
                        help="'dmcount' = DM-Count loss (OT + count + TV); 'mse' = debug fallback (count L1 + per-pixel MSE).")
    parser.add_argument('--density_count_weight', default=1.0, type=float,
                        help='Weight of the relative count term |sum(pred)-sum(gt)|/max(sum(gt),1) inside DensityLoss.')
    parser.add_argument('--density_abs_count_weight', default=0.5, type=float,
                        help='Weight of a second count term |sum(pred)-sum(gt)|/sqrt(max(sum(gt),1)) inside DensityLoss; 0 disables.')
    parser.add_argument('--density_ot_weight', default=0.1, type=float,
                        help='Weight of the Sinkhorn OT term inside DensityLoss (dmcount only).')
    parser.add_argument('--density_tv_weight', default=0.01, type=float,
                        help='Weight of the total-variation term inside DensityLoss (dmcount only).')
    parser.add_argument('--density_pix_weight', default=1.0, type=float,
                        help='Weight of the per-pixel MSE term (mse fallback only).')
    parser.add_argument('--ot_reg', default=0.1, type=float,
                        help='Entropic regularisation (epsilon) of the Sinkhorn OT.')
    parser.add_argument('--ot_num_iter', default=100, type=int,
                        help='Number of Sinkhorn iterations for the OT term.')
    # GT density sigma
    parser.add_argument('--density_sigma', default=8.0, type=float,
                        help='Gaussian sigma (px, original-image resolution) for the GT density map when --density_adaptive_sigma=0.')
    parser.add_argument('--density_adaptive_sigma', default=0, type=int,
                        help='1 = geometry-adaptive per-point sigma from k-NN point spacing (clamped); 0 = fixed --density_sigma.')
    parser.add_argument('--density_sigma_k', default=3, type=int,
                        help='k for the k-NN point spacing in adaptive sigma.')
    parser.add_argument('--density_sigma_beta', default=0.3, type=float,
                        help='Scale on the mean k-NN distance in adaptive sigma.')
    parser.add_argument('--density_sigma_min', default=2.0, type=float,
                        help='Lower clamp (px) for adaptive sigma.')
    parser.add_argument('--density_sigma_max', default=15.0, type=float,
                        help='Upper clamp (px) for adaptive sigma.')
    parser.add_argument('--ot_downsample', default=32, type=int,
                        help='Grid side for the OT cost matrix; density is avg-pooled to this NxN before Sinkhorn.')

    # FSCD147 train-time filter
    parser.add_argument('--max_objects', type=int, default=None,
                        help='FSCD147 only: drop training images with >= this many GT objects from the train split; val/test never filtered.')

    # stratified reporting
    parser.add_argument('--report_max_gt', type=int, nargs='+', default=[],
                        help='Inference only: GT-count thresholds for extra stratified MAE/MSE*/NAE rows in the results txt.')

    return parser
