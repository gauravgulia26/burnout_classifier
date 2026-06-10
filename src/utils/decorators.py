import sys
from functools import wraps

from src.core.exception import CustomException


def handle_exceptions(func):
    """Object Must Have its own logger for this Decorator to work"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)

        except Exception as e:

            err = CustomException(
                error_message=e,
                error_detail=sys,
            )

            self = args[0]

            if hasattr(self, "logger"):
                self.logger.exception(err)

            raise err

    return wrapper
