import torch


class WarmupThenPlateau:
    """Linear LR warmup for the first N epochs, then ReduceLROnPlateau.
    aux_group_indices + aux_warmup_epochs > 0 give those groups a longer ramp."""

    def __init__(self, optimizer, warmup_epochs, start_factor,
                 factor, patience, mode='min',
                 aux_group_indices=None, aux_warmup_epochs=0):
        self.optimizer = optimizer
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.start_factor = float(start_factor)
        aux_idx = set(int(i) for i in (aux_group_indices or []))
        aux_warmup = max(0, int(aux_warmup_epochs))
        n_groups = len(optimizer.param_groups)
        self._group_warmup = [
            (aux_warmup if (i in aux_idx and aux_warmup > 0) else self.warmup_epochs)
            for i in range(n_groups)
        ]
        self._max_warmup = max(self._group_warmup) if self._group_warmup else self.warmup_epochs
        self._peak_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience,
        )
        self._warmup_best = None
        self._epoch = 0  # number of step() calls so far
        # epoch 1's scale has to happen here, step() only runs after each epoch
        if self._max_warmup > 1:
            self._apply_warmup(1)

    def _warmup_scale(self, epoch_1based, warmup_len):
        if warmup_len <= 1:
            return 1.0
        # clamp so groups past their own warmup stay at 1.0
        t = (epoch_1based - 1) / (warmup_len - 1)
        t = min(max(t, 0.0), 1.0)
        return self.start_factor + (1.0 - self.start_factor) * t

    def _apply_warmup(self, next_epoch):
        for pg, peak, wlen in zip(self.optimizer.param_groups,
                                  self._peak_lrs, self._group_warmup):
            pg['lr'] = peak * self._warmup_scale(next_epoch, wlen)

    def _restore_peak(self):
        for pg, peak in zip(self.optimizer.param_groups, self._peak_lrs):
            pg['lr'] = peak

    def step(self, val_metric, plateau_enabled=True):
        """Advance one epoch. plateau_enabled=False (probe epochs) still ramps
        the warmup but keeps val_metric away from ReduceLROnPlateau."""
        self._epoch += 1
        # warmup epochs don't step the plateau, but track their best val for it
        if plateau_enabled and self._epoch <= self._max_warmup:
            if self._warmup_best is None or self._plateau.is_better(
                    val_metric, self._warmup_best):
                self._warmup_best = val_metric
        next_epoch = self._epoch + 1  # the epoch that trains next
        if next_epoch <= self._max_warmup:
            self._apply_warmup(next_epoch)
        else:
            if next_epoch == self._max_warmup + 1:
                self._restore_peak()
                if self._warmup_best is not None and self._plateau.is_better(
                        self._warmup_best, self._plateau.best):
                    self._plateau.best = self._warmup_best
            # the last warmup epoch still doesn't feed the plateau
            if self._epoch > self._max_warmup and plateau_enabled:
                self._plateau.step(val_metric)

    def scale_peak_lr(self, factor, group_indices=None):
        """Scale peak and live LR by factor, from the current values so earlier
        plateau cuts survive. group_indices=None hits every group and also
        resets the plateau patience."""
        factor = float(factor)
        idx = None if group_indices is None else set(int(i) for i in group_indices)
        for i, pg in enumerate(self.optimizer.param_groups):
            if idx is None or i in idx:
                self._peak_lrs[i] = self._peak_lrs[i] * factor
                pg['lr'] = pg['lr'] * factor
        if group_indices is None:
            self._plateau.best = self._plateau.mode_worse
            self._plateau.num_bad_epochs = 0
            self._plateau.cooldown_counter = 0

    @property
    def in_warmup(self):
        return self._epoch < self._max_warmup

    def state_dict(self):
        return {
            'epoch': self._epoch,
            'peak_lrs': self._peak_lrs,
            'warmup_best': self._warmup_best,
            'plateau': self._plateau.state_dict(),
        }

    def load_state_dict(self, state):
        self._epoch = state['epoch']
        self._peak_lrs = state['peak_lrs']
        self._warmup_best = state.get('warmup_best', None)
        self._plateau.load_state_dict(state['plateau'])
