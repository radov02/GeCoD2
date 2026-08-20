"""summary footer + SIGTERM/SIGINT cancel handler for the train scripts."""

import os
import signal


def write_summary(txt_path, run, best_metrics, model_path_abs,
                  cancelled=False, stopped_epoch=None, dens_best=None):
    """Append the summary footer and mirror it into the wandb run (rank 0 only).
    best_metrics may be None if cancelled before any best checkpoint."""
    if best_metrics is None:
        if cancelled:
            with open(txt_path, 'a') as f:
                f.write(
                    f"\n*** RUN CANCELLED (signal received during epoch "
                    f"{stopped_epoch}) -- no best checkpoint was saved yet. ***\n"
                )
            print("[rank0] run cancelled before any best checkpoint", flush=True)
        if run is not None:
            if cancelled:
                run.summary["cancelled"] = True
                run.summary["cancelled_at_epoch"] = stopped_epoch
            print(f"Wandb run: {run.url}", flush=True)
            run.finish()
        return

    with open(txt_path, 'a') as f:
        if cancelled:
            f.write(
                f"\n*** RUN CANCELLED (signal received during epoch "
                f"{stopped_epoch}); reporting best checkpoint so far. ***\n"
            )
        f.write(f"\nBest model - epoch {best_metrics['epoch']}:\n")
        f.write(f"  Train loss:      {best_metrics['train_loss']:.3f}\n")
        f.write(f"  Val loss:        {best_metrics['val_loss']:.3f}\n")
        f.write(f"  Train main loss: {best_metrics['train_main_loss']:.3f}\n")
        f.write(f"  Val main loss:   {best_metrics['val_main_loss']:.3f}\n")
        f.write(f"  Train MAE:       {best_metrics['train_mae']:.3f}\n")
        f.write(f"  Val MAE:         {best_metrics['val_mae']:.3f}\n")
        f.write(f"  Val RMSE:        {best_metrics['val_rmse']:.2f}\n")
        f.write(f"  Val NAE:         {best_metrics['val_nae']:.3f}\n")
        _vdm = best_metrics.get('val_dens_mae')
        if _vdm is not None and _vdm == _vdm:  # not NaN
            f.write(f"  Val dens MAE:    {_vdm:.3f}\n")
        f.write(f"  Val IoU@.5:      {best_metrics['val_iou']:.4f}\n")
        f.write(f"  Val GIoU@.5:     {best_metrics['val_giou']:.4f}\n")
        f.write(f"  Val Precision@.5: {best_metrics['val_precision']:.4f}\n")
        f.write(f"  Val Recall@.5:    {best_metrics['val_recall']:.4f}\n")
        f.write(f"  Val F1@.5:        {best_metrics['val_f1']:.4f}\n")
        f.write(f"  Epoch time:      {best_metrics['epoch_time']:.1f}s\n")
        if dens_best is not None:
            f.write(f"\nBest-DENSITY checkpoint - epoch {dens_best['epoch']}: "
                    f"Val dens MAE {dens_best['val_dens_mae']:.3f} "
                    f"(secondary, saved as *_bestdens_<epochs>.pth; "
                    f"box-MAE selection unchanged)\n")
        f.write(f"\nCheckpoint saved at: {model_path_abs}\n")
    print(f"Metrics saved to:    {txt_path}")
    print(f"Checkpoint saved at: {model_path_abs}")

    if run is not None:
        run.summary["best/epoch"] = best_metrics['epoch']
        run.summary["best/train_loss"] = best_metrics['train_loss']
        run.summary["best/val_loss"] = best_metrics['val_loss']
        run.summary["best/train_main_loss"] = best_metrics['train_main_loss']
        run.summary["best/val_main_loss"] = best_metrics['val_main_loss']
        run.summary["best/train_mae"] = best_metrics['train_mae']
        run.summary["best/val_mae"] = best_metrics['val_mae']
        run.summary["best/val_rmse"] = best_metrics['val_rmse']
        run.summary["best/val_nae"] = best_metrics['val_nae']
        run.summary["best/val_iou"] = best_metrics['val_iou']
        run.summary["best/val_giou"] = best_metrics['val_giou']
        run.summary["best/val_precision"] = best_metrics['val_precision']
        run.summary["best/val_recall"] = best_metrics['val_recall']
        run.summary["best/val_f1"] = best_metrics['val_f1']
        _vdm = best_metrics.get('val_dens_mae')
        if _vdm is not None and _vdm == _vdm:
            run.summary["best/val_dens_mae"] = _vdm
        if dens_best is not None:
            run.summary["dens_best/epoch"] = dens_best['epoch']
            run.summary["dens_best/val_dens_mae"] = dens_best['val_dens_mae']
        run.summary["best/checkpoint_path"] = model_path_abs
        if cancelled:
            run.summary["cancelled"] = True
            run.summary["cancelled_at_epoch"] = stopped_epoch
        print(f"Wandb run: {run.url}", flush=True)
        run.finish()


def install_cancel_handler(rank, cancel_state):
    """SIGTERM/SIGINT handlers that write the summary footer once on cancel.
    cancel_state: mutable dict the loop keeps current ('done' stops double-writes)."""
    def _handler(signum, frame):
        if cancel_state.get('done'):
            os._exit(0)
        cancel_state['done'] = True
        if rank == 0:
            try:
                write_summary(
                    cancel_state['txt_path'],
                    cancel_state.get('run'),
                    cancel_state.get('best_metrics'),
                    cancel_state['model_path_abs'],
                    cancelled=True,
                    stopped_epoch=cancel_state.get('epoch'),
                    dens_best=cancel_state.get('dens_best'),
                )
            except Exception as exc:  # still exit on failure
                print(f"[rank0] cancel finalize failed: {exc}", flush=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
