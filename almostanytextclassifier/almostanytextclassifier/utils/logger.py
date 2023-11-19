import sys
import logging


class Logger:
    @staticmethod
    def get_logger(logger_name: str = "[AlmostAnyTextClassifier]"):
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    filename="acitivity.log",
                    encoding="utf-8",
                    delay=True,
                    mode="w+",
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logger = logging.getLogger(logger_name)
        return logger
