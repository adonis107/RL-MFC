import os
import time

import torch


def synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def synchronized_time(device):
    synchronize_device(device)
    return time.perf_counter()


def progress_interval():
    """Iterations between progress lines. 0 silences them."""
    try:
        return max(0, int(os.environ.get("MFC_PROGRESS_INTERVAL", "1000")))
    except ValueError:
        return 1000


def report_progress(step, n_train, history):
    """Print one progress line, so a running job can be followed with tail.

    Training otherwise writes nothing until it returns, which leaves no way to
    tell a slow job from a stuck one, or to know how far along a multi-hour run
    is. The elapsed time is taken from the history the loop already records, so
    this adds no measurement of its own.
    """
    done = step + 1
    interval = progress_interval()
    if not interval or (done % interval and done != n_train):
        return

    elapsed = sum(history.get("train_step_seconds", ())) + sum(history.get("validation_seconds", ()))
    rate = elapsed / max(done, 1)
    objective = history.get("objective") or [float("nan")]
    print(
        f"[{done}/{n_train}] {100.0 * done / max(n_train, 1):5.1f}%  "
        f"elapsed {elapsed / 60:7.1f} min  left {rate * (n_train - done) / 60:7.1f} min  "
        f"{rate:6.3f} s/it  objective {objective[-1]:.6g}",
        flush=True,
    )
