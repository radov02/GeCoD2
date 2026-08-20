# FSCD-147 version of the ffm IOCfish training script.
import argparse
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import ops

from utils.env import load_env
load_env()

from models.counter import build_model, is_depth_fusion_param
from models.matcher import build_matcher
from utils.arg_parser import apply_lr_scaling, get_argparser
from utils.bbox_metrics import per_image_counters
from utils.box_ops import BOX_V_ABS_THRESHOLD
from utils.data import FSC147DATASET
from utils.data import pad_collate, pad_collate_test
from utils.losses import SetCriterion
from utils.losses import DensityLoss
from utils.scheduler import WarmupThenPlateau
from utils.finalize import write_summary, install_cancel_handler
from tools.timestamp import now_str

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def train(args):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )
    print("init dist: ", dist.is_initialized(), rank, world_size, gpu)

    seed = int(getattr(args, "seed", 42)) + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    if rank == 0:
        print(f"[rank0] seed={args.seed} (+rank offset for data/aug)", flush=True)

    # frozen Hiera backbone, DDP needs find_unused_parameters
    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu,
        find_unused_parameters=True,
    )

    apply_lr_scaling(args, rank=rank)
    backbone_params = dict()
    density_params = dict()
    depth_fuse_params = dict()
    zs_proto_params = dict()
    non_backbone_params = dict()
    for n, p in model.named_parameters():
        if 'backbone' in n:
            backbone_params[n] = p
        elif is_depth_fusion_param(n):
            depth_fuse_params[n] = p
        elif args.use_density > 0 and ('density_head' in n or 'density_guide' in n):
            density_params[n] = p
        elif 'zs_prototypes' in n:
            zs_proto_params[n] = p
        else:
            non_backbone_params[n] = p

    param_groups = [
        {'params': non_backbone_params.values()},
        {'params': backbone_params.values(), 'lr': args.backbone_lr},
    ]
    aux_group_indices = []  # groups that get the aux warmup
    if depth_fuse_params:
        aux_group_indices.append(len(param_groups))
        param_groups.append({'params': depth_fuse_params.values(),
                             'lr': args.depth_fuse_lr or args.lr})
    if density_params:
        aux_group_indices.append(len(param_groups))
        param_groups.append({'params': density_params.values(), 'lr': args.density_lr})
    if zs_proto_params:
        param_groups.append({'params': zs_proto_params.values(),
                             'lr': args.zs_proto_lr or args.lr})
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if rank == 0 and depth_fuse_params:
        print(f"[setup] depth-fusion optimizer group: {len(depth_fuse_params)} params "
              f"@ lr={(args.depth_fuse_lr or args.lr):.2e} (body @ lr={args.lr:.2e}, "
              f"probe_epochs={args.probe_epochs})", flush=True)
    if rank == 0 and density_params:
        print(f"[setup] density-head optimizer group: {len(density_params)} params "
              f"@ lr={args.density_lr:.2e} (other non-backbone @ lr={args.lr:.2e})",
              flush=True)
    if rank == 0 and zs_proto_params:
        print(f"[setup] zero-shot prototype optimizer group: {len(zs_proto_params)} params "
              f"@ lr={(args.zs_proto_lr or args.lr):.2e} (body @ lr={args.lr:.2e})",
              flush=True)
    # linear warmup, then ReduceLROnPlateau on val_select
    scheduler = WarmupThenPlateau(
        optimizer,
        warmup_epochs=args.lr_warmup_epochs,
        start_factor=args.lr_warmup_start_factor,
        factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience,
        mode='min',
        aux_group_indices=aux_group_indices,
        aux_warmup_epochs=args.aux_lr_warmup_epochs,
    )
    start_epoch = 0
    best = 10000000000000
    if rank == 0 and not args.init_from_pretrained and not args.resume_training:
        print("[rank0] WARNING: no --init_from_pretrained and no --resume_training -- "
              "the model trains FROM SCRATCH. If a pretrained init was intended, "
              "check PRETRAINED_INIT in the submitting shell.", flush=True)
    if args.init_from_pretrained:
        # warm start, params missing from the checkpoint keep their init
        ckpt_path = args.init_from_pretrained
        if not os.path.isabs(ckpt_path) and not os.path.isfile(ckpt_path):
            candidate = os.path.join(args.model_path, ckpt_path)
            if os.path.isfile(candidate):
                ckpt_path = candidate
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        if any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        missing, unexpected = model.module.load_state_dict(state, strict=False)
        if rank == 0:
            print(
                f"[rank0] init from pretrained: {ckpt_path} "
                f"(missing={len(missing)}, unexpected={len(unexpected)})",
                flush=True,
            )
            if missing:
                print(f"[rank0]   sample missing keys:    {missing[:5]}", flush=True)
            if unexpected:
                print(f"[rank0]   sample unexpected keys: {unexpected[:5]}", flush=True)
    elif args.resume_training:
        # resume ckpt keys already carry the DDP prefix
        checkpoint = torch.load(os.path.join(args.model_path, f'{args.model_name_resume_from}.pth'))
        model.load_state_dict(checkpoint['model'], strict=False)
    if args.use_depth > 0:
        # snapshot after weight loading for the deviation log
        model.module.snapshot_depth_fusion_init()
    if args.zero_shot:
        # snapshot for the per-epoch zs-prototype deviation log
        model.module.snapshot_zs_prototypes_init()
    best_metrics = None
    matcher = build_matcher(args)
    criterion = SetCriterion(0, matcher, {"loss_giou": args.giou_loss_coef}, ["bboxes", "ce"],
                             focal_alpha=args.focal_alpha)
    criterion.to(device)
    # density head loss, DM-Count OT by default
    density_criterion = DensityLoss(args).to(device) if args.use_density else None

    # GT density-map sigma settings
    density_kwargs = dict(
        density_sigma=args.density_sigma,
        density_adaptive_sigma=bool(args.density_adaptive_sigma),
        density_sigma_k=args.density_sigma_k,
        density_sigma_beta=args.density_sigma_beta,
        density_sigma_min=args.density_sigma_min,
        density_sigma_max=args.density_sigma_max,
    )
    train_dataset = FSC147DATASET(
        args.data_path,
        args.image_size,
        split='train',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        crop_p=args.crop_p,
        crop_min_px=args.crop_min_px,
        crop_max_px=args.crop_max_px,
        zero_shot=args.zero_shot,
        training=True,
        max_objects=args.max_objects,
        **density_kwargs,
    )
    val_dataset = FSC147DATASET(
        args.data_path,
        args.image_size,
        split='val',
        num_objects=args.num_objects,
        tiling_p=args.tiling_p,
        # eval mode adds scaling_factor+padwh so val can filter pad-region boxes
        training=False,
        **density_kwargs,
    )

    # depth-map cache: built on first run, read back as a 4th image channel
    depthmaps_dir_eff = args.depthmaps_dir if (getattr(args, 'use_depth', 0) > 0 and args.depthmaps_dir) else ''
    if depthmaps_dir_eff:
        from utils.depth_recipe import prepare_depthmaps
        prepare_depthmaps(
            depthmaps_dir_eff,
            {**train_dataset.id_to_imgpath(), **val_dataset.id_to_imgpath()},
            args.use_available_depthmaps, device=device, rank=rank,
            log=(print if rank == 0 else (lambda *a, **k: None)),
        )
        dist.barrier()
        train_dataset.depthmaps_dir = depthmaps_dir_eff
        val_dataset.depthmaps_dir = depthmaps_dir_eff
        if getattr(args, 'depthfeats_dir', ''):
            from utils.depth_recipe import (prepare_depthfeats, resolve_dino_input_size,
                                             resolve_depthfeats_spec)
            _dfk, _dfpca = resolve_depthfeats_spec(args.decoder_feat_channels_PCA)
            prepare_depthfeats(
                args.depthfeats_dir,
                {**train_dataset.id_to_imgpath(), **val_dataset.id_to_imgpath()},
                args.use_available_depthfeats, k=_dfk, pca=_dfpca,
                input_size=resolve_dino_input_size(args),
                fit_ids=list(train_dataset.id_to_imgpath()),
                device=device, rank=rank,
                log=(print if rank == 0 else (lambda *a, **k: None)),
            )
            dist.barrier()
            train_dataset.depthfeats_dir = args.depthfeats_dir
            val_dataset.depthfeats_dir = args.depthfeats_dir
    elif getattr(args, 'depthfeats_dir', '') and getattr(args, 'use_depth', 0) > 0:
        raise SystemExit(
            "[depthfeats][FATAL] --depthfeats_dir requires --depthmaps_dir")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        collate_fn=pad_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_dataset,
        sampler=DistributedSampler(train_dataset),
        batch_size=args.batch_size,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=DistributedSampler(val_dataset),
        batch_size=8,
        drop_last=False,
        # val samples are 7-tuples, pad_collate only unpacks 5, use pad_collate_test
        **{**loader_kwargs, 'collate_fn': pad_collate_test},
    )

    print(rank)
    model_path_abs = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}_{args.epochs}.pth'))
    txt_path = os.path.abspath(os.path.join(args.model_path, f'{args.model_name}_metrics.txt'))
    run = None
    if rank == 0:
        run = wandb.init(
            entity="radovicevic-erik1-",
            project="GECO2-D",
            name=args.model_name,
            job_type="train",
            config=vars(args),
        )
        print(f"Wandb run: {run.url}", flush=True)
        with open(txt_path, 'w') as f:
            f.write(f"Model path: {model_path_abs}\n")
            f.write(f"Created: {now_str()}\n")
            f.write(f"Wandb run: {run.url}\n")
            f.write(f"\n{'Epoch':>6s}  {'TrainLoss':>10s}  {'ValLoss':>8s}  {'TrainMainLoss':>14s}  {'ValMainLoss':>12s}  {'TrainMAE':>9s}  {'ValMAE':>7s}  {'ValRMSE':>8s}  {'ValNAE':>7s}  {'ValIoU':>7s}  {'ValGIoU':>8s}  {'ValF1':>6s}  {'LR':>9s}  {'EpochTime':>10s}  Best\n")
            f.write('-' * 121 + '\n')

    # spike early-stop state, rank 0 decides and broadcasts
    spike_count = 0
    best_spike_rmse = float('inf')
    recover_count = 0  # capped by --spike_recover_max
    # plateau stop: no val_select or val NAE improvement for --plateau_patience epochs
    plateau_count = 0
    best_plateau_select = float('inf')
    best_plateau_nae = float('inf')
    stop_flag = torch.zeros(1, dtype=torch.int32, device=device)
    recover_flag = torch.zeros(1, dtype=torch.int32, device=device)
    # adaptive probe: body frozen until val_select stalls, rank 0 decides
    in_probe = bool(args.probe_adaptive == 1 and depth_fuse_params)
    probe_best_select = float('inf')
    probe_stall = 0
    probe_lr_stall = 0
    probe_end_flag = torch.zeros(1, dtype=torch.int32, device=device)
    probe_cut_flag = torch.zeros(1, dtype=torch.int32, device=device)

    if rank == 0:
        print(
            f"[rank0] starting training: {args.epochs} epochs, "
            f"train={len(train_dataset)} val={len(val_dataset)}",
            flush=True,
        )
    cancel_state = {
        'txt_path': txt_path, 'run': run, 'model_path_abs': model_path_abs,
        'epoch': start_epoch, 'best_metrics': best_metrics, 'done': False,
    }
    install_cancel_handler(rank, cancel_state)

    if (args.probe_epochs > 0 or args.probe_adaptive == 1) and not depth_fuse_params:
        if rank == 0:
            print(f"[rank0] probe ignored: no depth-fusion params "
                  f"(use_depth={args.use_depth}).", flush=True)
        args.probe_epochs = 0
        args.probe_adaptive = 0
        in_probe = False
    # probe freeze list, zs_prototypes stay trainable
    probe_frozen = [p for n, p in {**non_backbone_params, **backbone_params}.items()
                    if 'zs_prototypes' not in n]
    if rank == 0 and in_probe:
        print(f"[rank0] ADAPTIVE PROBE: train ONLY depth-fusion ({len(depth_fuse_params)} "
              f"params) + any zero-shot prototype tokens; body unfreezes when val_select stalls "
              f"for {args.probe_plateau_patience} epochs (min {args.probe_epochs}, hard cap "
              f"{args.epochs // 2}); depth-fusion LR x{args.reduce_lr_factor} every "
              f"{args.probe_lr_patience} stalled epochs.", flush=True)
    elif rank == 0 and args.probe_epochs > 0:
        print(f"[rank0] FIXED PROBE: epochs 1..{args.probe_epochs} train ONLY the "
              f"depth-fusion group ({len(depth_fuse_params)} params) + any zero-shot "
              f"prototype tokens; body unfreezes at epoch {args.probe_epochs + 1}.", flush=True)
    _probe_ran = False  # for the --probe_only stop below
    for epoch in range(start_epoch + 1, args.epochs + 1):
        cancel_state['epoch'] = epoch
        probe_active = in_probe or (epoch <= args.probe_epochs)
        if probe_active:
            _probe_ran = True
        elif args.probe_only == 1 and _probe_ran:
            if rank == 0:
                print(f"[rank0] PROBE_ONLY: probe ended after epoch {epoch - 1}; "
                      f"stopping training with the body still frozen.", flush=True)
            break
        start = perf_counter()
        train_loss = torch.tensor(0.0).to(device)
        val_loss = torch.tensor(0.0).to(device)
        val_ae = torch.tensor(0.0).to(device)
        val_rmse = torch.tensor(0.0).to(device)
        val_nae = torch.tensor(0.0).to(device)
        train_ae = torch.tensor(0.0).to(device)
        train_main_loss = torch.tensor(0.0).to(device)
        val_main_loss = torch.tensor(0.0).to(device)
        train_density_loss = torch.tensor(0.0).to(device)
        # box-quality counters, greedy IoU matching at 0.5
        val_tp_iou_sum = torch.tensor(0.0).to(device)
        val_tp_giou_sum = torch.tensor(0.0).to(device)
        val_tp = torch.tensor(0.0).to(device)
        val_fp = torch.tensor(0.0).to(device)
        val_fn = torch.tensor(0.0).to(device)

        train_loader.sampler.set_epoch(epoch)
        if rank == 0:
            print(f"[rank0] epoch {epoch}/{args.epochs} -- training", flush=True)
        model.train()
        criterion.train()
        phase_start = perf_counter()
        n_train_batches = len(train_loader)
        train_log_interval = max(1, n_train_batches // 20)
        for batch_idx, (img, bboxes, density_map, img_name, gt_bboxes) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            bboxes = bboxes.to(device, non_blocking=True)
            gt_bboxes = gt_bboxes.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)
                outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux = aux

                losses = []
                num_objects_pred = []

                nms_bboxes = []
                for idx in range(img.shape[0]):
                    target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / args.image_size
                    l = criterion(outputs[idx], [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}], centerness[idx], ref_points[idx])
                                # outputs     , targets                                                                         , centerness     , ref_points

                    l1 = criterion(outputs_aux[idx],
                                   [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                   centerness_aux[idx], ref_points_aux[idx])
                    alpha = 0
                    if min((target_bboxes[:, 3] - target_bboxes[:, 1]).mean() * args.image_size,
                           (target_bboxes[:, 2] - target_bboxes[:, 0]).mean() * args.image_size) < 25:
                        alpha = args.aux_weight  # aux loss weight, 0.3 in the paper
                    # same box_v threshold as inference_whole (0.11*max with abs floor)
                    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)],
                                   outputs[idx]['box_v'][outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)], 0.5)

                    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)])[keep]
                    nms_bboxes.append(boxes)
                    num_objects_pred.append(len(boxes))
                    main_l = l['loss_giou'] + l["loss_ce"] + l["loss_bbox"]
                    losses.append(main_l)
                    losses.append(l1['loss_giou'] * alpha + l1["loss_ce"] * alpha + l["loss_bbox"] * alpha)
                    train_main_loss += main_l.detach()
                    if args.use_density:
                        _dloss = args.density_weight * density_criterion(outputs[idx]['pred_density'], density_map[idx])
                        losses.append(_dloss)
                        train_density_loss += _dloss.detach()
                num_objects_gt = density_map.flatten(1).sum(dim=1)
                loss = sum(losses)

            loss.backward()

            if probe_active:
                # null body/backbone grads before clipping so only the probe groups step
                for p in probe_frozen:
                    p.grad = None
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += loss
            train_ae += torch.abs(num_objects_gt - torch.tensor(num_objects_pred)).sum()
            if rank == 0 and (batch_idx + 1) % train_log_interval == 0:
                elapsed = perf_counter() - phase_start
                eta = elapsed * (n_train_batches - batch_idx - 1) / (batch_idx + 1)
                print(
                    f"[rank0] epoch {epoch}/{args.epochs} train "
                    f"[{batch_idx+1}/{n_train_batches}] | "
                    f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s",
                    flush=True,
                )

        criterion.eval()
        model.eval()
        if rank == 0:
            print(f"[rank0] epoch {epoch}/{args.epochs} -- validating", flush=True)
        with torch.no_grad():
            phase_start = perf_counter()
            n_val_batches = len(val_loader)
            val_log_interval = max(1, n_val_batches // 10)
            for batch_idx, (img, bboxes, density_map, img_name, gt_bboxes, scaling_factor, padwh) in enumerate(val_loader):
                img = img.to(device, non_blocking=True)
                bboxes = bboxes.to(device, non_blocking=True)
                gt_bboxes = gt_bboxes.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):  # val in fp32, matches inference_whole
                    outputs, ref_points, centerness, outputs_coord, aux = model(img, bboxes)
                outputs_aux, ref_points_aux, centerness_aux, outputs_coord_aux = aux
                losses = []

                num_objects_pred = []
                nms_bboxes = []

                for idx in range(img.shape[0]):
                    target_bboxes = gt_bboxes[idx][torch.logical_not((gt_bboxes[idx] == 0).all(dim=1))] / args.image_size

                    l1 = criterion(outputs_aux[idx],
                                   [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                   centerness_aux[idx], ref_points_aux[idx])

                    l = criterion(outputs[idx],
                                  [{"boxes": target_bboxes, "labels": torch.tensor([0] * target_bboxes.shape[0])}],
                                  centerness[idx], ref_points[idx])
                    keep = ops.nms(outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)],
                                   outputs[idx]['box_v'][outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)], 0.5)

                    boxes = (outputs[idx]['pred_boxes'][outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)])[keep]
                    boxes = torch.clamp(boxes, 0, 1)
                    # like inference_whole, drop boxes centered in the letterbox padding
                    _maxw = (img.shape[-1] - padwh[idx][0]).to(boxes.device)
                    _maxh = (img.shape[-2] - padwh[idx][1]).to(boxes.device)
                    _ctr = (boxes[:, :2] + boxes[:, 2:]) / 2
                    _valid = (_ctr[:, 0] * img.shape[-2] < _maxw) & (_ctr[:, 1] * img.shape[-1] < _maxh)
                    boxes = boxes[_valid]
                    nms_bboxes.append(boxes)
                    num_objects_pred.append(len(boxes))
                    # greedy match at IoU 0.5, both box sets in the same [0,1] frame
                    _bm_mask = outputs[idx]['box_v'] > torch.clamp(outputs[idx]['box_v'].max() * 0.11, min=BOX_V_ABS_THRESHOLD)
                    _bm_scores = outputs[idx]['box_v'][_bm_mask][keep][_valid]
                    _bm = per_image_counters(boxes, target_bboxes, _bm_scores)
                    val_tp_iou_sum += _bm['tp_iou_sum']
                    val_tp_giou_sum += _bm['tp_giou_sum']
                    val_tp += _bm['n_tp']
                    val_fp += _bm['n_fp']
                    val_fn += _bm['n_fn']
                    # same loss terms as in training
                    main_l = l['loss_giou'] + l["loss_ce"] + l["loss_bbox"]
                    losses.append(main_l)
                    losses.append(l1['loss_giou'] + l1["loss_ce"] + l1["loss_bbox"])
                    val_main_loss += main_l
                num_objects_gt = density_map.flatten(1).sum(dim=1)
                loss = sum(losses)

                num_objects_pred = torch.tensor(num_objects_pred)

                val_loss += loss
                val_ae += torch.abs(
                    num_objects_gt - num_objects_pred
                ).sum()
                val_rmse += torch.pow(
                    num_objects_gt - num_objects_pred, 2
                ).sum()
                # NAE = |gt - pred| / gt
                val_nae += (
                    torch.abs(num_objects_gt - num_objects_pred)
                    / torch.clamp(num_objects_gt, min=1)
                ).sum()
                if rank == 0 and (batch_idx + 1) % val_log_interval == 0:
                    elapsed = perf_counter() - phase_start
                    eta = elapsed * (n_val_batches - batch_idx - 1) / (batch_idx + 1)
                    print(
                        f"[rank0] val "
                        f"[{batch_idx+1}/{n_val_batches}] | "
                        f"{elapsed:.1f}s elapsed | ETA {eta:.1f}s",
                        flush=True,
                    )


        dist.all_reduce(train_loss)
        dist.all_reduce(val_loss)
        dist.all_reduce(val_rmse)
        dist.all_reduce(val_nae)
        dist.all_reduce(val_ae)
        dist.all_reduce(train_ae)
        dist.all_reduce(train_main_loss)
        dist.all_reduce(val_main_loss)
        dist.all_reduce(train_density_loss)
        dist.all_reduce(val_tp_iou_sum)
        dist.all_reduce(val_tp_giou_sum)
        dist.all_reduce(val_tp)
        dist.all_reduce(val_fp)
        dist.all_reduce(val_fn)

        # selection metric: RMSE/MAE blend of the all-reduced sums
        val_rmse_norm = torch.sqrt(val_rmse / len(val_dataset)).item()
        val_mae_norm = val_ae.item() / len(val_dataset)
        val_select = args.select_rmse_weight * val_rmse_norm + (1.0 - args.select_rmse_weight) * val_mae_norm
        # probe epochs skip the plateau scheduler (body frozen)
        scheduler.step(val_select, plateau_enabled=(not probe_active))
        current_lr = optimizer.param_groups[0]['lr']

        stop_flag.zero_()

        if rank == 0:

            end = perf_counter()
            epoch_time = end - start
            best_epoch = False

            val_mae_val  = val_ae.item()  / len(val_dataset)
            val_rmse_val = val_rmse_norm
            val_nae_val  = val_nae.item() / len(val_dataset)
            train_mae_val = train_ae.item() / len(train_dataset)
            train_main_loss_val = train_main_loss.item() / len(train_dataset)
            val_main_loss_val   = val_main_loss.item()   / len(val_dataset)
            train_density_loss_val = train_density_loss.item() / len(train_dataset)
            val_tp_n = val_tp.item()
            val_fp_n = val_fp.item()
            val_fn_n = val_fn.item()
            val_iou_val  = val_tp_iou_sum.item() / val_tp_n if val_tp_n > 0 else float('nan')
            val_giou_val = val_tp_giou_sum.item() / val_tp_n if val_tp_n > 0 else float('nan')
            val_prec = val_tp_n / (val_tp_n + val_fp_n) if (val_tp_n + val_fp_n) > 0 else float('nan')
            val_rec  = val_tp_n / (val_tp_n + val_fn_n) if (val_tp_n + val_fn_n) > 0 else float('nan')
            if val_prec != val_prec or val_rec != val_rec:
                val_f1 = float('nan')
            else:
                val_f1 = (2 * val_prec * val_rec / (val_prec + val_rec)) if (val_prec + val_rec) > 0 else 0.0

            # depth-fusion L2 deviation from the loaded snapshot
            depth_dev = model.module.depth_fusion_deviation() if args.use_depth > 0 else {}
            if depth_dev:
                dev_msg = (
                    f"  ~~~ depth-fusion deviation @epoch {epoch}: "
                    f"total_L2={depth_dev['total']:.4e}  ("
                    + ", ".join(f"{k}={v:.3e}" for k, v in sorted(depth_dev.items())
                                if k != 'total')
                    + ")\n"
                )
                print(dev_msg.strip())
                with open(txt_path, 'a') as f:
                    f.write(dev_msg)

            # zs-prototype movement from init
            zs_dev = model.module.zs_prototypes_deviation() if args.zero_shot else {}
            if zs_dev:
                zs_msg = (
                    f"  ~~~ zs-prototypes deviation @epoch {epoch}: "
                    f"total_L2={zs_dev['total']:.4e}  ("
                    + ", ".join(f"{k}={v:.3e}" for k, v in sorted(zs_dev.items())
                                if k != 'total')
                    + ")\n"
                )
                print(zs_msg.strip())
                with open(txt_path, 'a') as f:
                    f.write(zs_msg)

            # densg gate magnitude
            guide_stats = model.module.density_guide_stats() if getattr(args, 'density_guided', 0) else {}
            if guide_stats:
                guide_msg = (
                    f"  ~~~ density-guide gate @epoch {epoch}: "
                    f"gamma|mean|={guide_stats['gamma_abs_mean']:.4e}  "
                    f"gamma|max|={guide_stats['gamma_abs_max']:.4e}"
                    + (f"  rel_energy={guide_stats['rel_energy']:.4e}"
                       if 'rel_energy' in guide_stats else "")
                    + "\n"
                )
                print(guide_msg.strip())
                with open(txt_path, 'a') as f:
                    f.write(guide_msg)

            # conv-depth-adapter stats
            adapter_health = model.module.depth_adapter_health() if args.use_depth > 0 else {}
            if adapter_health:
                ah_msg = (
                    f"  ~~~ conv-depth-adapter @epoch {epoch}: "
                    f"eff_rank={adapter_health['eff_rank']:.1f}/{adapter_health['n_channels']:.0f}  "
                    f"|act|={adapter_health['act_absmean']:.3e}  std={adapter_health['act_std']:.3e}  "
                    f"dead={adapter_health['dead_frac']:.3f}  naninf={adapter_health['naninf']:.0f}\n"
                )
                print(ah_msg.strip())
                with open(txt_path, 'a') as f:
                    f.write(ah_msg)

            # render depth/edge cues for one val sample every --vis_every epochs
            if getattr(args, 'vis_every', 0) and args.use_depth > 0 and epoch % args.vis_every == 0:
                try:
                    _cue_dir = os.path.join(args.model_path, f'{args.model_name}_train_cue_vis')
                    _cue_img = val_dataset[0][0].to(device)
                    model.module.save_train_cue_visual(_cue_img, os.path.join(_cue_dir, f'epoch{epoch:04d}.png'))
                except Exception as _cue_e:
                    print(f"[rank0] cue viz failed @epoch {epoch}: {_cue_e}", flush=True)

            # spike detection
            if val_rmse_val < best_spike_rmse:
                best_spike_rmse = val_rmse_val
                spike_count = 0
            elif val_rmse_val > args.spike_ratio * best_spike_rmse:
                if recover_count < args.spike_recover_max:
                    # warm restart: reload best ckpt and cut LR instead of counting a spike
                    recover_count += 1
                    new_lr = optimizer.param_groups[0]['lr'] * args.spike_lr_decay
                    recover_flag.fill_(1)
                    recover_msg = (
                        f"  >>> RECOVER #{recover_count}/{args.spike_recover_max} at epoch {epoch}: "
                        f"Val RMSE={val_rmse_val:.2f} > {args.spike_ratio}x best "
                        f"({best_spike_rmse:.2f}). Restoring best ckpt, LR -> {new_lr:.2e}.\n"
                    )
                    print(recover_msg.strip())
                    with open(txt_path, 'a') as f:
                        f.write(recover_msg)
                else:
                    spike_count += 1
                    spike_msg = (
                        f"  *** SPIKE #{spike_count} at epoch {epoch}: "
                        f"Val RMSE={val_rmse_val:.2f} > {args.spike_ratio}x best "
                        f"({best_spike_rmse:.2f}) and recovery budget exhausted. "
                        f"Stopping in {args.spike_patience - spike_count} more spike(s).\n"
                    )
                    print(spike_msg.strip())
                    with open(txt_path, 'a') as f:
                        f.write(spike_msg)
                    if spike_count >= args.spike_patience:
                        stop_flag.fill_(1)
            else:
                spike_count = 0

            # plateau stop, shorter patience once --good_select is reached
            if args.plateau_patience > 0:
                improved = False
                if val_select < best_plateau_select:
                    best_plateau_select = val_select
                    improved = True
                if val_nae_val < best_plateau_nae:
                    best_plateau_nae = val_nae_val
                    improved = True
                target_reached = args.good_select > 0 and best_plateau_select <= args.good_select
                effective_patience = (min(args.plateau_patience, args.good_plateau_patience)
                                      if target_reached else args.plateau_patience)
                if improved:
                    plateau_count = 0
                elif probe_active:
                    pass  # probe epochs don't count
                else:
                    plateau_count += 1
                    plateau_msg = (
                        f"  --- PLATEAU {plateau_count}/{effective_patience} at epoch {epoch}: "
                        f"neither Val select ({val_select:.2f}, best {best_plateau_select:.2f}) nor "
                        f"Val NAE ({val_nae_val:.3f}, best {best_plateau_nae:.3f}) improved."
                        + (f" [good_select {args.good_select:.2f} reached -> short patience]"
                           if target_reached else "")
                        + "\n"
                    )
                    print(plateau_msg.strip())
                    with open(txt_path, 'a') as f:
                        f.write(plateau_msg)
                    if plateau_count >= effective_patience:
                        stop_flag.fill_(1)

            # probe end condition and in-probe LR cuts, rank 0 decides
            probe_cut_flag.zero_()
            probe_end_flag.zero_()
            if in_probe and epoch > args.probe_epochs:
                if val_select < probe_best_select - 1e-9:
                    probe_best_select = val_select
                    probe_stall = 0
                    probe_lr_stall = 0
                else:
                    probe_stall += 1
                    probe_lr_stall += 1
                    if args.probe_lr_patience > 0 and probe_lr_stall >= args.probe_lr_patience:
                        probe_cut_flag.fill_(1)
                        probe_lr_stall = 0  # restart the LR-cut window after a cut
                # end the probe on stall patience or the hard epochs//2 cap
                if probe_stall >= args.probe_plateau_patience or epoch >= args.epochs // 2:
                    probe_end_flag.fill_(1)
                probe_msg = (
                    f"  ::: PROBE epoch {epoch}: val_select={val_select:.3f} "
                    f"best={probe_best_select:.3f} stall={probe_stall}/{args.probe_plateau_patience}"
                    + ("  -> CUT depth-fusion LR" if probe_cut_flag.item() else "")
                    + ("  -> END PROBE (body unfreezes next epoch)" if probe_end_flag.item() else "")
                    + "\n"
                )
                print(probe_msg.strip())
                with open(txt_path, 'a') as f:
                    f.write(probe_msg)

            if val_select < best:
                best = val_select
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict()
                }
                torch.save(
                    checkpoint,
                    os.path.join(args.model_path, f'{args.model_name}_{args.epochs}.pth')
                )
                print(
                    f"[rank0] epoch {epoch} -- saved best checkpoint "
                    f"(val_select={val_select:.2f}, val_rmse={val_rmse_val:.2f}, val_mae={val_mae_val:.2f}) -> {model_path_abs}",
                    flush=True,
                )

                best_epoch = True
                best_metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss.item(),
                    'val_loss': val_loss.item(),
                    'train_main_loss': train_main_loss_val,
                    'val_main_loss': val_main_loss_val,
                    'train_mae': train_mae_val,
                    'val_mae': val_mae_val,
                    'val_rmse': val_rmse_val,
                    'val_nae': val_nae_val,
                    'val_iou': val_iou_val,
                    'val_giou': val_giou_val,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                    'val_f1': val_f1,
                    'epoch_time': epoch_time,
                }
                cancel_state['best_metrics'] = best_metrics

            print(
                f"-----------------------------------------------",
                f"Epoch: {epoch}",
                f"LR: {current_lr:.2e}",
                f"Train loss: {train_loss.item():.3f}",
                f"Val loss: {val_loss.item():.3f}",
                f"Train main loss: {train_main_loss_val:.3f}",
                f"Val main loss: {val_main_loss_val:.3f}",
                f"Train dens loss: {train_density_loss_val:.4f}",
                f"Train MAE: {train_mae_val:.3f}",
                f"Val MAE: {val_mae_val:.3f}",
                f"Val RMSE: {val_rmse_val:.2f}",
                f"Val NAE: {val_nae_val:.3f}",
                f"Val IoU@.5: {val_iou_val:.3f}",
                f"Val GIoU@.5: {val_giou_val:.3f}",
                f"Val F1@.5: {val_f1:.3f}",
                f"Epoch time: {epoch_time:.3f} seconds",
                'best' if best_epoch else ''
            )
            with open(txt_path, 'a') as f:
                f.write(
                    f"{epoch:>6d}  {train_loss.item():>10.3f}  {val_loss.item():>8.3f}  "
                    f"{train_main_loss_val:>14.3f}  {val_main_loss_val:>12.3f}  "
                    f"{train_mae_val:>9.3f}  "
                    f"{val_mae_val:>7.3f}  "
                    f"{val_rmse_val:>8.2f}  "
                    f"{val_nae_val:>7.3f}  "
                    f"{val_iou_val:>7.3f}  "
                    f"{val_giou_val:>8.3f}  "
                    f"{val_f1:>6.3f}  "
                    f"{current_lr:>9.2e}  "
                    f"{epoch_time:>10.1f}  "
                    f"{'*' if best_epoch else ''}\n"
                )
            run.log({
                "epoch": epoch,
                "lr": current_lr,
                "train/loss": train_loss.item(),
                "val/loss": val_loss.item(),
                "train/main_loss": train_main_loss_val,
                "val/main_loss": val_main_loss_val,
                "train/dens_loss": train_density_loss_val,
                "train/mae": train_mae_val,
                "val/mae": val_mae_val,
                "val/rmse": val_rmse_val,
                "val/nae": val_nae_val,
                "val/iou": val_iou_val,
                "val/giou": val_giou_val,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1": val_f1,
                "epoch_time_s": epoch_time,
                "best": int(best_epoch),
                "spike_count": spike_count,
                **({f"depth/fusion_dev_{k}": v for k, v in depth_dev.items()} if depth_dev else {}),
                **({f"density_guide/{k}": v for k, v in guide_stats.items()} if guide_stats else {}),
                **({f"depth_adapter/{k}": v for k, v in adapter_health.items()} if adapter_health else {}),
            }, step=epoch)

        # broadcast rank-0 decisions
        dist.broadcast(stop_flag, src=0)
        dist.broadcast(recover_flag, src=0)
        dist.broadcast(probe_cut_flag, src=0)
        dist.broadcast(probe_end_flag, src=0)
        if probe_cut_flag.item() == 1 and recover_flag.item() != 1 and aux_group_indices:
            # cut only the depth-fusion group's peak LR, unless a recovery already did
            scheduler.scale_peak_lr(args.reduce_lr_factor, group_indices=[aux_group_indices[0]])
            if rank == 0:
                print(f"[rank0] PROBE LR cut: depth-fusion group lr -> "
                      f"{optimizer.param_groups[aux_group_indices[0]]['lr']:.2e}", flush=True)
        if probe_end_flag.item() == 1:
            in_probe = False  # body unfreezes next epoch
            if rank == 0:
                print(f"[rank0] PROBE ended at epoch {epoch}: body unfreezes at epoch "
                      f"{epoch + 1} (best probe val_select={probe_best_select:.3f}).", flush=True)
        if recover_flag.item() == 1:
            # recovery: reload best weights, cut LR, clear optimizer state
            dist.barrier()
            ckpt = torch.load(model_path_abs, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            scheduler.scale_peak_lr(args.spike_lr_decay)
            optimizer.state.clear()
            recover_flag.zero_()
            if rank == 0:
                print(f"[rank0] recovery applied: restored epoch {ckpt.get('epoch', '?')}, "
                      f"LR -> {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if stop_flag.item() == 1:
            if rank == 0:
                print(f"Early stopping triggered (spike_count={spike_count}, plateau_count={plateau_count}).")
            break
    dist.destroy_process_group()
    if rank == 0:
        print("[rank0] training complete", flush=True)

    if rank == 0:
        # stop the cancel handler from writing the footer again
        cancel_state['done'] = True
        write_summary(txt_path, run, best_metrics, model_path_abs)


if __name__ == '__main__':
    _script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser('GECO2-FSCD147', parents=[get_argparser()])

    # FSCD147 defaults, CLI flags still win
    # few-shot only, 3 exemplar boxes per image
    parser.set_defaults(
        model_name='GECO2_FSCD147',
        model_name_resumed='GECO2_FSCD147',
        dataset='FSCD147',
        data_path=str(_script_dir / 'fsc147'),
        model_path=str(_script_dir / 'models'),
        # architecture, must match the checkpoint
        backbone='SAM',
        reduction=16,
        image_size=1024,
        num_enc_layers=3,
        emb_dim=256,
        num_heads=8,
        kernel_dim=3,
        num_objects=3,
        # Training hyper-parameters
        epochs=200,
        lr=1e-4,
        backbone_lr=0.0,
        lr_drop=50, # unused
        reduce_lr_patience=3,
        reduce_lr_factor=0.5,
        spike_patience=2,
        spike_ratio=2.0,
        weight_decay=5e-5,
        batch_size=8,
        dropout=0.1,
        num_workers=8,
        max_grad_norm=0.1,
        tiling_p=0.5,
        # Loss coefficients
        giou_loss_coef=2,
        cost_class=2,
        cost_bbox=1,
        cost_giou=2,
        focal_alpha=0.25,
        aux_weight=0.3,
        # Depth fusion, mode 5 = BiSeNet FFM per level
        use_depth=5,
        depth_feat_channels=16, # config default
    )

    # pull defaults from training/config_FSCD147.sh
    from utils.run_config import apply_run_config_defaults
    apply_run_config_defaults(parser, dataset='FSCD147')

    args = parser.parse_args()
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    print(args)
    train(args)
