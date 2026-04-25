from time import perf_counter
from functools import wraps
from typing import Any, Callable

F = Callable[..., Any]


def timeit(enabled: bool) -> F:
    """
    Decorator factory to measure and print the execution time of a function.

    When enabled, this decorator wraps a function and reports the time taken
    for each call using ``time.perf_counter``. The timing is printed even if
    the wrapped function raises an exception.

    Parameters
    ----------
    enabled : bool
        If True, timing is applied to the decorated function.
        If False, the original function is returned unchanged (no overhead).

    Returns
    -------
    Callable
        A decorator that either wraps the function with timing logic or
        returns the original function unchanged.

    Notes
    -----
    - The decorator prints execution time in seconds and minutes.
    - Function metadata (name, docstring, etc.) is preserved via ``functools.wraps``.
    - Timing is performed using ``time.perf_counter`` for high-resolution measurement.

    Examples
    --------
    >>> @timeit(enabled=True)
    ... def slow_function():
    ...     import time
    ...     time.sleep(1)
    ...
    >>> slow_function()
    slow_function time: 1.000s (0.017 min)
    """

    def decorator(func: F) -> F:
        if not enabled:
            return func

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            t0 = perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                t1 = perf_counter()
                print(f"{func.__name__} time: {t1 - t0}s ({(t1 - t0) / 60:.3f} min)")

        return wrapper

    return decorator


if __name__ == "__main__":

    @timeit(enabled=True)
    def sleep(n):
        from time import sleep

        sleep(n)
        print(f"Slept for {n}s")
        return 3

    print(sleep(1))
