"""This module implements logging tools."""

import logging


def set_logger(log_path: str) -> None:
    """Configures the logger to save information in a certain format to
    the specified output.

    Parameters
    ----------
    log_path : str
        A path where the logs are going to be saved.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
