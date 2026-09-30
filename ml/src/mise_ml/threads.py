"""Run catalog work in thread pools with interruptible waits on Windows.

Poll every half second because blocking pool waits prevent the main thread from receiving signals.
Share a stop event so log.session can end the process without waiting for network calls.
"""

import threading
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from typing import Any

STOP = threading.Event()


class Stopped(Exception):
    """The run is stopping; raised in threads other than the main thread."""


def run_all(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    """fn over items in a thread pool; the results are in item order."""

    def guarded(item: Any) -> Any:
        if STOP.is_set():
            raise Stopped
        return fn(item)

    pool = ThreadPoolExecutor(max_workers=max(1, workers))
    futures: list[Future] = [pool.submit(guarded, item) for item in items]
    try:
        pending = set(futures)
        while pending:
            if STOP.is_set():
                raise Stopped
            done, pending = wait(pending, timeout=0.5, return_when=FIRST_EXCEPTION)
            for f in done:
                if f.exception() is not None:
                    raise f.exception()
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            STOP.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=False)
    return [f.result() for f in futures]
