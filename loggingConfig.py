#! usr/env/bin python3

import logging


def configureLogging(loggingLevel, loggerName: str) -> logging.Logger:
    # Create a logger instance with the specified name
    logger = logging.getLogger(loggerName)
    logger.setLevel(loggingLevel)

    # Create a console handler for displaying log messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(loggingLevel)

    # Create a formatter and add it to the console handler
    formatter = logging.Formatter("(%(asctime)s) %(name)s %(levelname)s: %(message)s", "%d.%m. %H:%M:%S")
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    L = configureLogging(logging.INFO, __name__)
    L.info("uwuw")