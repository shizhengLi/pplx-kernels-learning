import logging


def setup_logging(prefix: str = "") -> None:
    logging.basicConfig(
        level="DEBUG",
        format=prefix
        + "[%(asctime)s] [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
        datefmt="%H:%M:%S",
    )
