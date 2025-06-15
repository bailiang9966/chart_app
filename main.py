from flask import Flask, request
from flask_socketio import SocketIO, emit, disconnect
import time
from app.services.bnws import BNWS
from app.services.bnrest import BNREST
import threading
import json
from app import create_app  # 导入 create_app 方法


app = create_app()
socketio = SocketIO(app)
# 启动币安的 websocket
bnws = BNWS()
# bnr = BNREST()
# 保存用户订阅信息，结构为 session_id -> (token, minutes)
subscriptions = {}

@socketio.on('subscribe_kline')
def handle_subscribe_kline(data):
    token = data['token']
    minutes = data['minutes']
    session_id = request.sid
    subscriptions[session_id] = (token, minutes)

    
    if bnws.ws_running :
        kline_data = bnws.get_kline_data(data_type='all')
    else:
        emit('ws_err', "ws连接状态异常", to=session_id)

    # if bnr.his_ready:
    #     kline_data = bnr.get_kline_data(data_type='all')
    if kline_data:
        for session_id, (token, minutes) in subscriptions.items():
            # 获取该token的所有分钟数据（安全访问避免KeyError）
            token_data = kline_data.get(token, {})
            # 过滤出用户订阅的分钟周期数据
            filtered_data = {m: token_data.get(m) for m in minutes if token_data.get(m)}
            if filtered_data:  # 仅当有有效数据时发送
                # 包含token字段供前端识别
                socketio.emit('kline_init', {
                    token:filtered_data
                }, to=session_id)

@socketio.on('unsubscribe_kline')
def handle_unsubscribe_kline():
    session_id = request.sid
    if session_id in subscriptions:
        del subscriptions[session_id]

def send_kline_data():
    while True:
        if bnws.ws_running:
            for token in bnws.tokens:
                bnws.calculate_indicators(token)
            kline_data = bnws.get_kline_data(data_type='last')
        else:
            break
        # bnr.get_his_kline()
        # kline_data = bnr.get_kline_data(data_type='last')
        if kline_data:
            for session_id, (token, minutes) in subscriptions.items():
                # 获取该token的所有分钟数据（安全访问避免KeyError）
                token_data = kline_data.get(token, {})
                # 过滤出用户订阅的分钟周期数据
                filtered_data = {m: token_data.get(m) for m in minutes if token_data.get(m)}
                if filtered_data:  # 仅当有有效数据时发送
                    # 包含token字段供前端识别
                    socketio.emit('kline_refresh', {
                        token:filtered_data
                    }, to=session_id)
            time.sleep(1)   

# 监听断开连接事件
@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in subscriptions:
        del subscriptions[session_id]

# 使用 socketio 的 start_background_task 方法启动线程
socketio.start_background_task(send_kline_data)

if __name__ == '__main__':
    socketio.run(app, debug=True)