from flask import Flask
from app.utils.config_loader import config_loader  # 导入工具实例

def create_app():
    app = Flask(__name__)
    
    # 从工具类获取已缓存的配置
    app.config.update(config_loader.config)
    
    # 注册路由蓝图（示例）
    from app.routes.bnsj import bn_sj_bp  # 修正：使用正确的蓝图名称
    app.register_blueprint(bn_sj_bp)  # 修正：注册正确的蓝图实例

    return app