"""One logging setup for every step.

The console gets short lines (`12:04:31 resolve  INFO  message`) at INFO and above. A file in
out/logs/ gets every DEBUG line of the run. tqdm bars and log lines share the terminal through
logging_redirect_tqdm, so a log line never breaks a bar.
"""

import contextlib
import logging
import shutil
import time
from collections.abc import Iterable, Iterator
from datetime import datetime
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mise_ml.config import DATA, ML_ROOT, OUT

ROOT = "mise_ml"
LOGS = OUT / "logs"


class ShortNameFormatter(logging.Formatter):
    """Shows `mise_ml.resolve` as `resolve`, and WARNING as WARN so the columns line up."""

    LEVELS = {"WARNING": "WARN", "CRITICAL": "CRIT"}

    def format(self, record: logging.LogRecord) -> str:
        record.short = record.name.rsplit(".", 1)[-1]
        record.level = self.LEVELS.get(record.levelname, record.levelname)
        return super().format(record)


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name.rsplit('.', 1)[-1]}")


def num(n: int | float) -> str:
    """A count with thousands separators: 14993 -> 14,993."""
    return f"{n:,}"


def elapsed(start: float) -> str:
    s = int(time.perf_counter() - start)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s % 3600 // 60:02d}m"


def progress(iterable: Iterable[Any] | None = None, **kwargs: Any) -> tqdm:
    """A tqdm bar with the same look everywhere. It clears itself when done."""
    kwargs.setdefault("leave", False)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("disable", None)  # no bars when stderr is not a terminal
    return tqdm(iterable, **kwargs)


def log_environment(log: logging.Logger) -> None:
    import torch
    import transformers

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        gpu = f"{props.name}, {free / 2**30:.1f} of {total / 2**30:.1f} GiB free"
    else:
        gpu = "none (CUDA not available)"
    disk = shutil.disk_usage(DATA if DATA.exists() else ML_ROOT)
    log.info(
        "environment: GPU %s; torch %s, transformers %s; %.0f GiB free on the data disk",
        gpu,
        torch.__version__,
        transformers.__version__,
        disk.free / 2**30,
    )


@contextlib.contextmanager
def session(step: str) -> Iterator[logging.Logger]:
    """Set up logging for one command. All steps of one `all` run share one file."""
    root = logging.getLogger(ROOT)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(
        ShortNameFormatter("%(asctime)s %(short)-8s %(level)-5s %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{datetime.now():%Y%m%d-%H%M%S}-{step}.log"
    file = logging.FileHandler(path, encoding="utf-8")
    file.setLevel(logging.DEBUG)
    file.setFormatter(
        ShortNameFormatter("%(asctime)s %(short)-8s %(level)-5s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(file)
    # Library warnings (for example Hugging Face cache notes) go to the file, not the console.
    logging.captureWarnings(True)
    warnings_log = logging.getLogger("py.warnings")
    warnings_log.handlers = [file]
    warnings_log.propagate = False

    log = get("cli")
    log.debug("log file %s", path)
    with logging_redirect_tqdm(loggers=[root]):
        # The redirect swaps in its own console handler, which keeps no level.
        for handler in root.handlers:
            if not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.INFO)
        log_environment(log)
        try:
            yield log
        except BaseException as e:
            if not isinstance(e, SystemExit | KeyboardInterrupt):
                log.exception("failed")
            raise
        finally:
            log.info("full log: %s", path.relative_to(ML_ROOT).as_posix())
            file.close()
