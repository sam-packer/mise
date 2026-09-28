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
LIBRARIES = ("transformers", "huggingface_hub")
LOGS = OUT / "logs"


class ShortNameFormatter(logging.Formatter):
    """Shows `mise_ml.resolve` as `resolve`, and WARNING as WARN so the columns line up."""

    LEVELS = {"WARNING": "WARN", "CRITICAL": "CRIT"}
    LIBRARY_NAMES = {"transformers": "hf", "huggingface_hub": "hub", "py": "warnings"}

    def format(self, record: logging.LogRecord) -> str:
        top = record.name.split(".", 1)[0]
        record.short = self.LIBRARY_NAMES.get(top) or record.name.rsplit(".", 1)[-1]
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


def route_libraries(console: logging.Handler, file: logging.Handler) -> None:
    """Send transformers and huggingface_hub logs through the same handlers.

    transformers installs its own stderr handler, which would bypass the tqdm redirect and
    the log file. Their INFO lines go to the file only; WARNING and above also reach the
    console.
    """
    import transformers.utils.logging as tf_logging

    tf_logging.disable_default_handler()
    library_console = logging.StreamHandler()
    library_console.setLevel(logging.WARNING)
    library_console.setFormatter(console.formatter)
    for name in LIBRARIES:
        lib = logging.getLogger(name)
        lib.handlers = [library_console, file]
        lib.setLevel(logging.INFO)
        lib.propagate = False


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
    route_libraries(console, file)

    log = get("cli")
    log.debug("log file %s", path)
    libraries = [logging.getLogger(name) for name in LIBRARIES]
    with logging_redirect_tqdm(loggers=[root, *libraries]):
        # The redirect swaps in its own console handlers, which keep no level.
        for logger, level in [(root, logging.INFO)] + [(lib, logging.WARNING) for lib in libraries]:
            for handler in logger.handlers:
                if not isinstance(handler, logging.FileHandler):
                    handler.setLevel(level)
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
