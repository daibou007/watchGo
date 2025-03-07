import os
import logging
from datetime import datetime


class Logger:
    def __init__(self, config):
        self.config = config
        self.setup_logger()

    def setup_logger(self):
        """设置日志记录器"""
        log_dir = os.path.join(self.config.base_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir, f"watchgo_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # 设置日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 设置日志级别
        level = getattr(logging, self.config.logging.level.upper())

        # 配置根日志记录器
        logger = logging.getLogger("WatchGo")
        logger.setLevel(level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.logger = logger

    def get_logger(self):
        """获取日志记录器"""
        return self.logger

    def error(self, message):
        """记录错误日志"""
        self.logger.error(message)

    def warning(self, message):
        """记录警告日志"""
        self.logger.warning(message)

    def info(self, message):
        """记录信息日志"""
        self.logger.info(message)

    def debug(self, message):
        """记录调试日志"""
        self.logger.debug(message)

    def critical(self, message):
        """记录严重错误日志"""
        self.logger.critical(message)
