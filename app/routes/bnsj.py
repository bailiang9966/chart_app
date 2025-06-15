from flask import Blueprint,render_template, request,jsonify
from ..services import bnk

# 创建蓝图实例（名称、模块名）
bn_sj_bp = Blueprint('bn_sj', __name__)

# 定义路由：/bn_sj 指向 bn_sj.html 模板
@bn_sj_bp.route('/bn_sj')
def bn_sj_page():
    return render_template('bn_sj.html')  # 需确保 templates 目录存在 bn_sj.html 文件

@bn_sj_bp.route('/sj10')
def sj10_page():
    return render_template('sj_10.html')  


@bn_sj_bp.route('/bn_sj/his_kline')
def get_his_kline():
    token = request.args.get('token')
    minute = int(request.args.get('minute'))
    start_ts = int(request.args.get('start_ts'))
    end_ts = int(request.args.get('end_ts'))
    limit = int(request.args.get('limit'))

    # 查询k线数据
    print(token,minute,limit,start_ts,end_ts)
    params = bnk.get_params(token,minute,limit,start_ts,end_ts)
    print(params)
    df = bnk.get_his_df(params)
    # TODO 计算指标数据

    result = df.to_dict('records')

    return jsonify(result)