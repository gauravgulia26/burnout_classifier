import sys

from rich.traceback import install


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        install()
        super().__init__(error_message)

        _, _, exc_tb = error_detail.exc_info()

        self.file_name = exc_tb.tb_frame.f_code.co_filename
        self.line_number = exc_tb.tb_lineno
        self.error_message = error_message

    def __str__(self):

        return (
            f"Error occurred in {self.file_name} at line {self.line_number}: {self.error_message}"
        )
