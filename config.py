import os
import json
import logging


class Config:
    def __init__(self, config_file="config.json"):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.base_path, config_file)

        # 设置基础日志配置
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("Config")

        self.load_config()

    def load_config(self):
        """加载配置"""
        self.logger.info("开始加载配置文件")
        default_config = {
            "calibration_file": "calibration.npy",
            # ... 其他默认配置保持不变 ...
        }

        try:
            if os.path.exists(self.config_file):
                self.logger.debug(f"找到配置文件: {self.config_file}")
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.logger.debug("成功读取配置文件")
                    self._update_config(default_config, config)
                    self.logger.info("成功更新配置")
            else:
                self.logger.warning(f"配置文件不存在，创建默认配置: {self.config_file}")
                self.save_config(default_config)

            self._set_attributes(default_config)
            self.logger.info("配置加载完成")

        except Exception as e:
            self.logger.error(f"加载配置失败: {str(e)}")
            self.logger.warning("使用默认配置")
            self._set_attributes(default_config)

    def _update_config(self, default, update):
        """递归更新配置"""
        self.logger.debug("开始递归更新配置")
        for key, value in update.items():
            if key in default and isinstance(default[key], dict):
                self.logger.debug(f"更新子配置: {key}")
                self._update_config(default[key], value)
            else:
                self.logger.debug(f"更新配置项: {key}")
                default[key] = value
        self.logger.debug("配置更新完成")

    def _set_attributes(self, config):
        """设置类属性"""
        self.logger.debug("开始设置类属性")
        for key, value in config.items():
            try:
                if isinstance(value, dict):
                    self.logger.debug(f"设置字典属性: {key}")
                    setattr(self, key, type("Config", (), value))
                else:
                    self.logger.debug(f"设置普通属性: {key}")
                    setattr(self, key, value)
            except Exception as e:
                self.logger.error(f"设置属性 {key} 失败: {str(e)}")
        self.logger.debug("类属性设置完成")

    def save_config(self, config=None):
        """保存配置"""
        self.logger.info("开始保存配置")
        if config is None:
            self.logger.debug("使用当前配置")
            config = self._get_current_config()

        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.logger.info(f"配置已保存到: {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"保存配置失败: {str(e)}")
            return False

    def _get_current_config(self):
        """获取当前配置"""
        self.logger.debug("开始获取当前配置")
        config = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_") and key != "config_file":
                try:
                    if isinstance(value, type):
                        self.logger.debug(f"获取类属性: {key}")
                        config[key] = value.__dict__
                    else:
                        self.logger.debug(f"获取普通属性: {key}")
                        config[key] = value
                except Exception as e:
                    self.logger.error(f"获取属性 {key} 失败: {str(e)}")
        self.logger.debug("当前配置获取完成")
        return config
