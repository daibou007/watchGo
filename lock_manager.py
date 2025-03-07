import os
import atexit
import sys

class LockManager:
    def __init__(self, lock_file="/tmp/watchgo.lock"):
        self.lock_file = lock_file
        atexit.register(self.cleanup)
        
    def acquire(self):
        """获取锁"""
        if os.path.exists(self.lock_file):
            print("WatchGo 已经在运行中")
            sys.exit(1)
        else:
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))
                
    def cleanup(self):
        """清理锁文件"""
        if os.path.exists(self.lock_file):
            os.remove(self.lock_file)