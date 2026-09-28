"""Thread pools that stop on Ctrl+C, on Windows too.

On Windows, a main thread that blocks in a pool join or in `pool.map` never sees Ctrl+C.
Here the waiting thread polls every half second. Ctrl+C in the main thread sets STOP, so
pools in other threads stop too, and log.session ends the process at once. Finished work
is on disk already: every append and every file write is safe against a kill.
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
