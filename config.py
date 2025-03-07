import os
import json
import logging

class Config:
    def __init__(self, config_file='config.json'):
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(self.base_path, config_file)
        self.load_config()
        
    def load_config(self):
        """加载配置"""
        default_config = {
            'camera': {
                'device_id': 0,
                'width': 1280,
                'height': 720,
                'fps': 30
            },
            'board': {
                'size': 19,
                'corner_detection': {
                    'min_distance': 20,
                    'quality_level': 0.01,
                    'block_size': 3
                }
            },
            'analysis': {
                'territory_threshold': 0.6,
                'dead_stone_threshold': 1
            },
            'output': {
                'save_frames': True,
                'save_analysis': True
            },
            'logging': {
                'level': 'INFO',
                'file': 'watchgo.log'
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self._update_config(default_config, config)
            else:
                self.save_config(default_config)
                
            self._set_attributes(default_config)
            
        except Exception as e:
            print(f"加载配置失败: {str(e)}")
            self._set_attributes(default_config)
            
    def _update_config(self, default, update):
        """递归更新配置"""
        for key, value in update.items():
            if key in default and isinstance(default[key], dict):
                self._update_config(default[key], value)
            else:
                default[key] = value
                
    def _set_attributes(self, config):
        """设置类属性"""
        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, type('Config', (), value))
            else:
                setattr(self, key, value)
                
    def save_config(self, config=None):
        """保存配置"""
        if config is None:
            config = self._get_current_config()
            
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {str(e)}")
            return False
            
    def _get_current_config(self):
        """获取当前配置"""
        config = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'config_file':
                if isinstance(value, type):
                    config[key] = value.__dict__
                else:
                    config[key] = value
        return config