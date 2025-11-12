# logger.py
import logging
import sys
from datetime import datetime

class LoggerFactory:
    reset_color = '\x1b[0m'
    timestamp_color = '\x1b[35m'
    location_color = '\x1b[36m'
    message_color = '\x1b[30m'
    value_color = '\x1b[38;5;141m'

    level_colors = {
        'DEBUG': '\x1b[34m',
        'INFO': '\x1b[32m',
        'WARNING': '\x1b[33m',
        'ERROR': '\x1b[31m',
        'CRITICAL': '\x1b[41m'
    }

    @staticmethod
    def get_logger(name: str = None):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(LoggerFactory.CustomFormatter())
            logger.addHandler(handler)

        return logger

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            level_color = LoggerFactory.level_colors.get(record.levelname, '')
            timestamp = datetime.fromtimestamp(record.created).strftime('%d-%m-%Y %H:%M:%S')
            colored_level = f"{level_color}[{record.levelname}]{LoggerFactory.reset_color}"
            colored_timestamp = f"{LoggerFactory.timestamp_color}[{timestamp}]{LoggerFactory.reset_color}"
            colored_location = f"{LoggerFactory.location_color}{record.filename}:{record.lineno}{LoggerFactory.reset_color}"
            colored_message = f"{LoggerFactory.message_color}{record.getMessage()}{LoggerFactory.reset_color}"

            return f"{colored_level} {colored_timestamp} {colored_location} : {colored_message}\n" + \
                   "-" * 100