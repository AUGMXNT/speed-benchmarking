import time
from   loguru import logger

class TimeIt:
    def __init__(self, name=None):
        self.name = name

    async def __aenter__(self):
        self.start_time = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        elapsed_time *= 1000
        logger.info(f"{self.name} took {elapsed_time:.1f} ms")

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        elapsed_time *= 1000
        logger.debug(f"{self.name} took {elapsed_time:.1f} ms")

    def __call__(self, func):
        @functools.wraps(func)  # Preserve the signature of the original function
        async def wrapped_func(*args, **kwargs):
            async with TimeIt(self.name or func.__name__):
                return await func(*args, **kwargs)
        return wrapped_func
