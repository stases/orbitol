"""Narrow fp32 AdamW used for training, and implementation-preserving resume.

CUDA arithmetic targets PyTorch 2.8's non-capturable foreach implementation.
Only Orbitol's betas=(0.9, 0.999), eps=1e-8, and zero weight decay are supported.
"""
from __future__ import annotations

import functools
import torch


def _configured_state(optimizer, state_dict):
    """Copy group metadata before PyTorch chooses the counter device on load."""
    if len(optimizer.param_groups) != len(state_dict["param_groups"]):
        raise ValueError("loaded state dict has a different number of parameter groups")
    groups = []
    for current, saved in zip(optimizer.param_groups, state_dict["param_groups"]):
        groups.append({**saved, **{key: current[key] for key in
                                  ("foreach", "fused", "capturable")}})
    # Noncapturable PyTorch load leaves step on its SAVED device. Normalize to
    # CPU explicitly; PyTorch moves it to the parameter device for fused mode.
    # Clone also prevents independent optimizers sharing a mutable CPU counter.
    states = {key: ({**value, "step": value["step"].detach().cpu().clone()}
                    if "step" in value else dict(value))
              for key, value in state_dict["state"].items()}
    return {**state_dict, "state": states, "param_groups": groups}


def load_adamw_state(optimizer, state_dict):
    """Load moments/LR while keeping the destination AdamW execution flags."""
    optimizer.load_state_dict(_configured_state(optimizer, state_dict))


@functools.lru_cache(maxsize=None)
def _kernel():
    # CPU tests and CPU checkpoint readers do not require a Triton installation.
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def update(P, G, M, V, N: tl.constexpr, rate, correction, BLOCK: tl.constexpr):
        x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = x < N
        p = tl.load(P + x, mask, 0)
        g = tl.load(G + x, mask, 0)
        m = tl.load(M + x, mask, 0)
        v = tl.load(V + x, mask, 0)
        # Explicit FMA only within the corresponding eager CUDA operator.
        m = tl.fma(0.1, g - m, m)
        v = tl.fma(0.001, g * g, v * 0.999)
        denom = libdevice.div_rn(libdevice.sqrt_rn(v), correction) + 1.0e-8
        p = tl.fma(rate, libdevice.div_rn(m, denom), p)
        tl.store(P + x, p, mask)
        tl.store(M + x, m, mask)
        tl.store(V + x, v, mask)

    return update


@functools.lru_cache(maxsize=None)
def _multi_tensor_kernel():
    """Batch the exact arithmetic; EMA keeps mul then fused add rounding."""
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    @triton.jit
    def update(Meta, Blocks, Rates, Step, decay, weight, EMA: tl.constexpr,
               SCHEDULED: tl.constexpr, BLOCK: tl.constexpr):
        block = tl.program_id(0)
        index = tl.load(Blocks + 2 * block)
        offset = tl.load(Blocks + 2 * block + 1)
        stride: tl.constexpr = 3 if EMA else 5
        p_ptr = tl.load(Meta + index * stride).to(tl.pointer_type(tl.float32))
        other_ptr = tl.load(Meta + index * stride + 1).to(tl.pointer_type(tl.float32))
        n = tl.load(Meta + index * stride + stride - 1)
        x = offset + tl.arange(0, BLOCK)
        mask = x < n
        p = tl.load(p_ptr + x, mask, 0)
        other = tl.load(other_ptr + x, mask, 0)
        if EMA:
            # Separate multiply rounding, then the foreach add's explicit FMA.
            p = tl.fma(weight, other, p * decay)
        else:
            m_ptr = tl.load(Meta + index * stride + 2).to(tl.pointer_type(tl.float32))
            v_ptr = tl.load(Meta + index * stride + 3).to(tl.pointer_type(tl.float32))
            m = tl.load(m_ptr + x, mask, 0)
            v = tl.load(v_ptr + x, mask, 0)
            rate_index = tl.load(Step) if SCHEDULED else index
            rate = tl.load(Rates + rate_index * 2)
            correction = tl.load(Rates + rate_index * 2 + 1)
            m = tl.fma(0.1, other - m, m)
            v = tl.fma(0.001, other * other, v * 0.999)
            denom = libdevice.div_rn(libdevice.sqrt_rn(v), correction) + 1.0e-8
            p = tl.fma(rate, libdevice.div_rn(m, denom), p)
            tl.store(m_ptr + x, m, mask)
            tl.store(v_ptr + x, v, mask)
        tl.store(p_ptr + x, p, mask)

    return update


def _upload(values, *, dtype, device):
    """Fresh pinned storage stays owned by the allocator until its copy ends."""
    return torch.tensor(values, dtype=dtype, device="cpu", pin_memory=True).to(
        device, non_blocking=True)


class _MultiTensorUpdate:
    """One eager CUDA launch with persistent pointer staging and geometry.

    Two pinned buffers alternate. An event protects each buffer until its H2D
    copy completes; the device copy and update are ordered on the current stream.
    Gradients may belong to CUDA-graph trees and must not be retained here.
    Unscheduled standalone callers retain the legacy per-parameter rate upload.
    These runtime caches never enter a checkpoint.
    """

    def __init__(self):
        self._update = _multi_tensor_kernel()
        self._key = None
        self._storage = None
        self._meta = None

    def _refresh_metadata(self, metadata, device):
        shape = (len(metadata), len(metadata[0]))
        if self._meta is None or self._meta.shape != shape or self._meta.device != device:
            self._meta = torch.empty(shape, dtype=torch.int64, device=device)
            self._staging = [torch.empty(shape, dtype=torch.int64, device="cpu", pin_memory=True)
                             for _ in range(2)]
            self._views = [buffer.numpy() for buffer in self._staging]
            self._events = [torch.cuda.Event() for _ in range(2)]
            self._slot = 0
        event = self._events[self._slot]
        if not event.query():
            event.synchronize()
        self._views[self._slot][:] = metadata
        self._meta.copy_(self._staging[self._slot], non_blocking=True)
        event.record(torch.cuda.current_stream(device))
        self._slot = 1 - self._slot

    def __call__(self, tensors, *, rates=None, step=None, decay=None):
        device = tensors[0][0].device
        sizes = tuple(row[0].numel() for row in tensors)
        key = (device, sizes)
        if key != self._key:
            blocks = [(i, offset) for i, n in enumerate(sizes) for offset in range(0, n, 1024)]
            self._blocks = _upload(blocks, dtype=torch.int64, device=device)
            self._key = key
        if not len(self._blocks):
            return
        storage = tuple(tuple(tensor.data_ptr() for tensor in row) for row in tensors)
        if storage != self._storage or key != getattr(self, "_meta_key", None):
            metadata = [[*pointers, n] for pointers, n in zip(storage, sizes)]
            self._refresh_metadata(metadata, device)
            self._storage, self._meta_key = storage, key
        mutated_key = tuple(tuple(id(row[i]) for i in ((0,) if decay is not None else (0, 2, 3)))
                            for row in tensors)
        if mutated_key != getattr(self, "_mutated_key", None):
            self._mutated = ([row[0] for row in tensors] if decay is not None
                             else [t for row in tensors for t in (row[0], row[2], row[3])])
            self._mutated_key = mutated_key
        meta = self._meta
        rate = (rates if step is not None else
                _upload(rates, dtype=torch.float32, device=device) if rates is not None else meta)
        with torch.cuda.device(device):
            self._update[(len(self._blocks),)](
                meta, self._blocks, rate, step if step is not None else meta,
                decay if decay is not None else 0.,
                1. - decay if decay is not None else 0., EMA=decay is not None,
                SCHEDULED=step is not None, BLOCK=1024, enable_fp_fusion=False)
        torch.autograd.graph.increment_version(self._mutated)


class CachedClip:
    """PyTorch's L2 foreach arithmetic, with cached lists on one device.

    The same-device norm.to(device) and coefficient.to(device) calls in PyTorch
    are identities. Mixed dtype/device and sparse inputs retain its fallback.
    """

    def __init__(self, parameters):
        self.parameters = tuple(parameters)
        self._grads = []
        self._key = None

    @torch.no_grad()
    def __call__(self, max_norm):
        key = tuple((p.grad.device, p.grad.dtype, p.grad.layout)
                    if p.grad is not None else None for p in self.parameters)
        if key != self._key:
            kinds = {kind for kind in key if kind is not None}
            self._foreach = (len(kinds) == 1 and all(layout == torch.strided
                             and device.type in ("cpu", "cuda") for device, _, layout in kinds))
            self._key = key
        self._grads.extend(p.grad for p in self.parameters if p.grad is not None)
        try:
            if not self._grads:
                return torch.tensor(0.)
            if not self._foreach:
                return torch.nn.utils.clip_grad_norm_(self.parameters, max_norm, error_if_nonfinite=False)
            norms = torch._foreach_norm(self._grads, 2.)
            norm = torch.linalg.vector_norm(torch.stack(norms), 2.)
            coefficient = torch.clamp(float(max_norm) / (norm + 1e-6), max=1.)
            torch._foreach_mul_(self._grads, coefficient)
            return norm
        finally:
            # Do not keep graph-pool gradient outputs alive into the next forward.
            self._grads.clear()


class ExactAdamW(torch.optim.AdamW):
    """Training AdamW: a one-pass CUDA update that rounds like foreach AdamW; ordinary foreach on CPU.

    Checkpoints use AdamW's standard state and CPU step counters. Construction
    selects the implementation; loading cannot silently change that selection.
    configure_schedule enables cached Python-rounded rate tables indexed by an
    int64 device counter. Updates, scheduler.step() and finish_schedule_step()
    stay eager. zero_grad retains PyTorch semantics, including set_to_none=True,
    so compiled backward's graph-pool gradients can be released between steps.
    GradScaler's overflow read remains synchronous so skipped updates leave the
    CPU counters unchanged; switching to device powers would change rounding.
    """

    def __init__(self, params, lr=1e-4, weight_decay=0.0, *, fast_update=False):
        if weight_decay != 0:
            raise ValueError("ExactAdamW supports only zero weight decay")
        if not isinstance(lr, (float, int)):
            raise ValueError("ExactAdamW requires a Python scalar learning rate")
        super().__init__(params, lr=lr, weight_decay=0., foreach=True, fused=False)
        self._update = None
        self.fast_update = fast_update
        self._batched_updates = {}
        self._host_steps = {}
        self._schedule = None
        self._schedule_dirty = False
        self._rate_tables = {}

    def configure_schedule(self, scheduler, max_steps):
        """Rebuild runtime tables for this schedule, including after resume.

        The first LR is the optimizer's saved LR: LambdaLR.load_state_dict does
        not recompute it. Later rows use the current lambda and base LR, matching
        ordinary scheduler.step even if the schedule/horizon changed on resume.
        No runtime tables, pointers or device counters are checkpoint fields.
        """
        if not self.fast_update or scheduler.optimizer is not self:
            raise ValueError("configure_schedule requires this fast_update optimizer")
        if max_steps < scheduler.last_epoch:
            raise ValueError("max_steps precedes the resumed scheduler step")
        self._schedule, self._max_steps = scheduler, max_steps
        self._next_step = scheduler.last_epoch
        device = next(p for group in self.param_groups for p in group["params"]).device
        self._device_step = torch.tensor(self._next_step, dtype=torch.int64, device=device)
        self._host_steps = {p: int(state["step"]) for p, state in self.state.items() if state}
        self._rate_tables.clear()
        self._batched_updates.clear()
        self._schedule_dirty = False
        self._initial_epoch = self._next_step
        self._initial_lrs = [group["lr"] for group in self.param_groups]
        for i in range(len(self.param_groups)):
            self._rate_table(i, 0)

    def _rate_table(self, group_index, offset):
        key = (group_index, offset)
        if key not in self._rate_tables:
            base_lr = self._schedule.base_lrs[group_index]
            multiplier = self._schedule.lr_lambdas[group_index]
            values = []
            for k in range(self._max_steps):
                lr = (self._initial_lrs[group_index] if k == self._initial_epoch
                      else base_lr * multiplier(k))
                count = k + 1 - offset
                values.append((-(lr / (1 - 0.9 ** count)), (1 - 0.999 ** count) ** 0.5)
                              if count > 0 else (0., 1.))
            device = self._device_step.device
            self._rate_tables[key] = (_upload(values, dtype=torch.float32, device=device)
                                      if device.type == "cuda" else
                                      torch.tensor(values, dtype=torch.float32).reshape(-1, 2))
        return self._rate_tables[key]

    def _ensure_schedule(self):
        if self._schedule_dirty:
            self.configure_schedule(self._schedule, self._max_steps)

    def finish_schedule_step(self):
        """Check host counters without reading the device; handle scaler skips.

        A successful update already advanced the device counter. GradScaler may
        skip an update; only that exceptional eager path needs a counter fill.
        Its next update uses bias corrections for the actual number of updates.
        """
        self._ensure_schedule()  # A scaler can skip the first update after loading.
        assert self._schedule.last_epoch in (self._next_step, self._next_step + 1)
        if self._schedule.last_epoch != self._next_step:
            self._next_step = self._schedule.last_epoch
            self._device_step.fill_(self._next_step)

    def load_state_dict(self, state_dict):
        configured = _configured_state(self, state_dict)
        for group in configured["param_groups"]:
            self._validate_group(group)
        super().load_state_dict(configured)
        self._batched_updates.clear()
        self._host_steps.clear()
        self._schedule_dirty = self._schedule is not None

    @staticmethod
    def _validate_group(group):
        if (group["betas"] != (0.9, 0.999) or group["eps"] != 1e-8
                or group["weight_decay"] != 0 or group["amsgrad"]
                or group["maximize"] or group["differentiable"]
                or group["capturable"] or group["fused"]
                or not group["foreach"] or not isinstance(group["lr"], (float, int))):
            raise ValueError("ExactAdamW only supports Orbitol's scalar-LR AdamW settings")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ExactAdamW does not support closures")
        for group in self.param_groups:
            self._validate_group(group)
        self._ensure_schedule()
        if self._schedule is not None:
            assert self._next_step == self._schedule.last_epoch, "scheduler and optimizer steps disagree"
            if self._next_step >= self._max_steps:
                raise ValueError("rate table exhausted; configure_schedule with the new max_steps")
        params = [p for group in self.param_groups for p in group["params"]]
        devices = {p.device.type for p in params}
        if devices == {"cpu"}:
            result = super().step()
            if self._schedule is not None:
                self._device_step.add_(1)
                self._next_step += 1
            return result
        if devices != {"cuda"} or len({p.device for p in params}) != 1:
            raise ValueError("ExactAdamW requires all CPU or one CUDA device")
        if self._update is None and not self.fast_update:
            self._update = _kernel()
        for group_index, group in enumerate(self.param_groups):
            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if (p.dtype != torch.float32 or not p.is_contiguous()
                        or p.grad.is_sparse or not p.grad.is_contiguous()):
                    raise ValueError("ExactAdamW requires contiguous dense fp32 parameters and gradients")
                state = self.state[p]
                if not state:
                    state.update(step=torch.tensor(0., dtype=torch.float32, device="cpu"),
                                 exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
                if self.fast_update and p not in self._host_steps:
                    self._host_steps[p] = int(state["step"])
            if not active:
                for key in list(self._batched_updates):
                    if key[0] == group_index:
                        del self._batched_updates[key]
                continue
            if self.fast_update:
                # Normally one cohort at offset zero. Missing gradients and scaler
                # skips need the same LR but a different Python bias correction.
                cohorts = {}
                for p in active:
                    offset = self._next_step - self._host_steps[p] if self._schedule is not None else 0
                    cohorts.setdefault(offset, []).append(p)
                    self._host_steps[p] += 1
                live = set()
                for offset, cohort in cohorts.items():
                    # Bias offsets can change after a scaler skip without any
                    # storage change. Keep the pointer/block tables in that case.
                    key = (group_index, tuple(id(p) for p in cohort))
                    live.add(key)
                    if key not in self._batched_updates:
                        batch = self._batched_updates[key] = _MultiTensorUpdate()
                        batch._steps = [self.state[p]["step"] for p in cohort]
                    batch = self._batched_updates[key]
                    tensors = [(p, p.grad, self.state[p]["exp_avg"], self.state[p]["exp_avg_sq"])
                               for p in cohort]
                    torch._foreach_add_(batch._steps, 1)
                    if self._schedule is not None:
                        batch(tensors, rates=self._rate_table(group_index, offset), step=self._device_step)
                    else:
                        rates = [(-(group["lr"] / (1 - 0.9 ** self._host_steps[p])),
                                  (1 - 0.999 ** self._host_steps[p]) ** 0.5) for p in cohort]
                        batch(tensors, rates=rates)
                # Bound runtime caches when gradient participation changes.
                for key in list(self._batched_updates):
                    if key[0] == group_index and key not in live:
                        del self._batched_updates[key]
                for key in list(self._rate_tables):
                    if key[0] == group_index and key[1] not in cohorts:
                        del self._rate_tables[key]
                continue
            torch._foreach_add_([self.state[p]["step"] for p in active], 1)
            mutated = []
            with torch.cuda.device(active[0].device):
                for p in active:
                    state = self.state[p]
                    step = state["step"].item()
                    if p.numel():
                        self._update[((p.numel() + 1023) // 1024,)](
                            p, p.grad, state["exp_avg"], state["exp_avg_sq"], p.numel(),
                            -(group["lr"] / (1 - 0.9 ** step)), (1 - 0.999 ** step) ** 0.5,
                            BLOCK=1024, enable_fp_fusion=False)
                    mutated.extend((p, state["exp_avg"], state["exp_avg_sq"]))
            # Raw Triton stores do not increment PyTorch's mutation counter.
            torch.autograd.graph.increment_version(mutated)
        if self._schedule is not None:
            # Ordered after all groups; incrementing in one update block would
            # race other blocks reading the counter.
            self._device_step.add_(1)
            self._next_step += 1
