import yaml
import logging
from pathlib import Path
from typing import Dict, Optional

class ConfigLoader:
    _instance = None  # 单例实例
    _config: Optional[Dict] = None  # 允许 _config 为 Dict 或 None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()  # 首次实例化时加载配置
            cls._instance._setup_logging()  # 初始化日志配置
        return cls._instance

    def _load_config(self):
        """加载YAML配置（仅执行一次）"""
        config_path = Path(__file__).parent.parent.parent / 'config.yml'  # 定位到项目根目录的config.yml
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def _setup_logging(self):
        """初始化日志配置"""
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 获取根日志器并添加处理器
        root_logger = logging.getLogger()
        root_logger.setLevel(self._config['log_level'])
        root_logger.addHandler(console_handler)

    @property
    def config(self) -> Dict:
        """提供只读访问（确保配置已加载）"""
        if self._config is None:
            raise RuntimeError("配置未加载，请检查 config.yml 文件是否存在或格式是否正确")
        return self._config  # 此时类型检查器确认返回值为 Dict

# 全局单例实例（其他模块直接导入这个实例）
config_loader = ConfigLoader()