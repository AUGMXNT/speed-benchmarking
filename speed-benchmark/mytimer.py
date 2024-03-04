from   loguru import logger
import statistics
import time

class TimeIt:
    runs = {}

    def __init__(self, name=None):
        self.name = name

    async def __aenter__(self):
        self.start_time = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        elapsed_time *= 1000
        logger.info(f"{self.name} took {elapsed_time:.1f} ms")
        TimeIt.runs.setdefault(self.name, []).append(elapsed_time)

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        elapsed_time *= 1000
        logger.info(f"{self.name} took {elapsed_time:.1f} ms")
        TimeIt.runs.setdefault(self.name, []).append(elapsed_time)

    def __call__(self, func):
        @functools.wraps(func)  # Preserve the signature of the original function
        async def wrapped_func(*args, **kwargs):
            async with TimeIt(self.name or func.__name__):
                return await func(*args, **kwargs)
        return wrapped_func

    @staticmethod
    def get_mean(name):
        return statistics.mean(TimeIt.runs.get(name, []))

    @staticmethod
    def get_median(name):
        return statistics.median(TimeIt.runs.get(name, []))
