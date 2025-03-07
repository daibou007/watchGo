import os
import atexit
import sys
import signal
from logger import Logger


class LockManager:
    def __init__(self, config, lock_file="/tmp/watchgo.lock"):
        self.lock_file = lock_file
        self.pid = os.getpid()
        self.config = config
        self.logger = Logger(self.config)

        # 注册清理函数
        atexit.register(self.cleanup)
        self.logger.debug(f"注册清理函数，进程ID: {self.pid}")

        # 注册信号处理器
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        self.logger.debug("注册信号处理器完成")

    def _signal_handler(self, signum, frame):
        """处理进程信号"""
        self.logger.info(f"接收到信号: {signum}，开始清理...")
        self.cleanup()
        sys.exit(0)

    def acquire(self):
        """获取锁"""
        self.logger.debug(f"尝试获取锁文件: {self.lock_file}")
        if os.path.exists(self.lock_file):
            # 检查已存在的进程是否还在运行
            try:
                with open(self.lock_file, "r") as f:
                    old_pid = int(f.read().strip())
                self.logger.debug(f"发现已存在的锁文件，进程ID: {old_pid}")
                if self._is_process_running(old_pid):
                    self.logger.warning("WatchGo 已经在运行中")
                    print("WatchGo 已经在运行中")
                    sys.exit(1)
                else:
                    self.logger.info(f"进程 {old_pid} 已不存在，删除旧的锁文件")
                    os.remove(self.lock_file)
            except (ValueError, FileNotFoundError) as e:
                self.logger.warning(f"处理已存在的锁文件时出错: {str(e)}")
                pass

        # 创建新的锁文件
        with open(self.lock_file, "w") as f:
            f.write(str(self.pid))
        self.logger.info(f"成功创建锁文件，进程ID: {self.pid}")

    def _is_process_running(self, pid):
        """检查进程是否在运行"""
        try:
            os.kill(pid, 0)
            self.logger.debug(f"进程 {pid} 正在运行")
            return True
        except OSError:
            self.logger.debug(f"进程 {pid} 未运行")
            return False

    def cleanup(self):
        """清理锁文件"""
        try:
            if os.path.exists(self.lock_file):
                with open(self.lock_file, "r") as f:
                    file_pid = int(f.read().strip())
                if file_pid == self.pid:
                    self.logger.info(f"清理锁文件: {self.lock_file}")
                    os.remove(self.lock_file)
                else:
                    self.logger.warning(f"锁文件PID不匹配，当前: {self.pid}，文件: {file_pid}")
        except (ValueError, FileNotFoundError) as e:
            self.logger.error(f"清理锁文件时出错: {str(e)}")
            pass