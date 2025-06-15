from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from .bnk import get_his_df,get_params,get_interval
import pandas as pd
import json
import logging
import copy
from  app.services.ti import TI


class BNWS():
    """
    初始化自动连接币安的websocket 并订阅BTC和ETH的1分钟级别的实时k
    历史数据计算初始的k线和指标 并根据币种和周期缓存起来
    后续当ws返回1分钟级别k数据的时候 根据时间戳判断是否是新增的bar 更新缓存的各个周期k和指标
    历史数据中 币安不提供10分钟数据 从5分钟的数据聚合获得
    ws获取数据后更新高时间周期数据时 都从低时间周期聚合 
    """

    RETURN_COLS_DROP = ['uf_volume','spot_volume']
    def __init__(self):
        self.ws_running = False
        self.his_ready = False
        self.tokens = ['BTC','ETH']
        self.periods = [1,3,5,10,15,30,60,240]
        self.source_periods = {}
        self.kline_data = {
            token: {period: pd.DataFrame() for period in self.periods}
            for token in self.tokens
        }
        self.get_source_periods()
        logging.debug(f"start binance ws...")

        
        
        #初始获取历史数据
        logging.info(f"get_his_kline...")
        self.get_his_kline()
        #启动ws
        self.start_ws()
        logging.debug(f"init ok...")
        pass
    def get_source_periods(self):
        """
        计算聚合高周期k线数据的时候需要用哪个作为源周期
        """
        for i in range(len(self.periods)):
            current_num = self.periods[i]
            found_divisor_in_period = None
            # 从当前索引的前一个位置向前查找
            for j in range(i - 1, -1, -1):
                previous_num = self.periods[j]
                if current_num % previous_num == 0:
                    found_divisor_in_period = previous_num
                    break # 找到第一个后就停止查找

            self.source_periods.update({current_num:found_divisor_in_period})
        pass
    def start_ws_spot(self):
        self.ws_spot = SpotWebsocketStreamClient(
            on_message=self.ws_spot_msg_handler,
            on_close=self.ws_close_handler)
        #订阅1分钟级别k线数据
        for token in self.tokens:
            self.ws_spot.kline(
                symbol=token.upper() + 'USDT',
                interval='1m'
            )

    def start_ws_uf(self):
        self.ws_uf = UMFuturesWebsocketClient(
            on_message=self.ws_uf_msg_handler,
            on_close=self.ws_close_handler)
        #订阅1分钟级别k线数据
        for token in self.tokens:
            self.ws_uf.kline(
                symbol=token.upper() + 'USDT',
                interval='1m'
            )


    def start_ws(self):
        if not self.ws_running:
            self.start_ws_spot()
            self.start_ws_uf()   
            self.ws_running = True

    def ws_spot_msg_handler(self,_, message):
        """
        如果ws完全启动且历史数据已经查询完毕 则开始处理ws返回的数据
        """
        if not self.ws_running or self.his_ready:
            return
        
        result = json.loads(message)
        if 'e' in result and result['e'] == 'kline':
            k = result['k']
            if not 'USDT' in k['s']:
                return
            token = k['s'][:-4]

            k_data = {
                'ts':k['t'],
                'spot_volume':float(k['v']),
                'spot_delta':float(k['V'])*2 - float(k['v'])
            }
            self.update_kline(token,k_data)

    def ws_uf_msg_handler(self,_, message):
        if not self.ws_running or not self.his_ready:
            return
        result = json.loads(message)
        if 'e' in result and result['e'] == 'kline':
            k = result['k']
            if not 'USDT' in k['s']:
                return
            token = k['s'][:-4]
            
            k_data = {
                'ts':k['t'],
                'open':float(k['o']),
                'high':float(k['h']),
                'low':float(k['l']),
                'close':float(k['c']),
                'uf_volume':float(k['v']),
                'uf_delta':float(k['V'])*2-float(k['v']),
            }
            
            self.update_kline(token,k_data)
    
    def update_kline(self,token,k_data):
        """ 
        将k_data中的属性 根据ts更新或者新增到1分钟级别的df中
        然后聚合出高周期的df
        """
        # 1.更新1分钟级别df数据
        df = self.kline_data[token][1]
        if df.empty:
            logging.debug("1分钟k数据为空")
            return
        
        
        
        k_data["ts"] = pd.to_datetime(k_data["ts"], unit="ms")+ pd.Timedelta(hours=8) 
        last_ts = df.index[-1]

        if k_data['ts'] == last_ts:
            
            for col in k_data:
                if col != 'ts':
                    df.loc[last_ts, col] = k_data[col]
        elif 'open' in k_data:
            df.loc[k_data['ts']] = [k_data.get(col, 0) for col in df.columns]

        
        # 2.修改高时间周期的df
        for i,curr_period in enumerate(self.periods[1:]):
            df = self.kline_data[token][curr_period]
            if df.empty:
                logging.debug(f"{curr_period}分钟k数据空")
                continue
            last_ts = df.index[-1]
            next_ts = last_ts + pd.Timedelta(minutes=curr_period)
            #聚合的周期选择

            source_period = self.source_periods[curr_period]
            source_df = self.kline_data[token][source_period]
            source_last_ts = source_df.index[-1]

            start_ts = next_ts if next_ts <= source_last_ts else last_ts
            sub_df = self.kline_data[token][source_period][self.kline_data[token][source_period].index>=start_ts]
            # print(sub_df.columns)
            tmp_df = sub_df.resample(f"{curr_period}min").agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'uf_volume': 'sum',
                'spot_volume':'sum',
                'spot_delta':'sum',
                'uf_delta':'sum'

            })
            #将tem_df数据更新到self.kline_data[token][source_period] 
            self.kline_data[token][curr_period].loc[start_ts, tmp_df.columns] = tmp_df.iloc[0];

        for period in self.kline_data[token]:
                df = self.kline_data[token][period]
                if len(df) > 500:
                    # 保留最后500行，删除前面的数据
                    self.kline_data[token][period] = df.iloc[-500:]
            # 
            
    
    def ws_close_handler(self, sm):
        self.ws_running = False
        logging.error('ws closed')               
    
    def get_his_kline(self):
        from concurrent.futures import ThreadPoolExecutor
        
        def process_token_period(token, period):
            if period != 10:
                params = get_params(token, minute=period)
                df = get_his_df(params)
            else:
                params = get_params(token, minute=5, limit=1000)
                df = get_his_df(params)
                df = df.resample(f"{period}min").agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'uf_volume': 'sum',
                    'spot_volume':'sum',
                    'spot_delta':'sum',
                    'uf_delta':'sum'
                })
            self.kline_data[token][period] = df

        # 使用线程池并发处理任务（根据实际情况调整max_workers）
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for token in self.tokens:
                for period in self.periods:
                    # 提交任务到线程池
                    futures.append(executor.submit(process_token_period, token, period))
            # 等待所有任务完成
            for future in futures:
                future.result()  # 捕获可能的异常

        # 所有数据加载完成后计算指标
        for token in self.tokens:
            self.calculate_indicators(token)
        self.his_ready = True
        pass
    def calculate_indicators(self,token):
        """
        统一计算指标
        """
        for p in self.periods:
            df = self.kline_data[token][p]
            if df.empty:
                continue
            df['volume'] = df['uf_volume'] + df['spot_volume']
            df['delta']=df['spot_delta']+df['uf_delta']

            pd.options.mode.chained_assignment = None
            ti = TI(df)
            df['EMA20'] = ti.EMA(window=20)
            df['EMA50'] = ti.EMA(window=50)
            df['MACD'], df['SIGNAL'], df['HISTOGRAM'] = ti.MACD()
            df['KDJ_K'], df['KDJ_D'], df['KDJ_J'] = ti.KDJ(n=69)
            df['PLUS_DI'], df['MINUS_DI'],_, _= ti.DMI()
            df['WR'] = ti.WR(window=39)
        pass
    
    def get_kline_data(self,data_type='all'):
        """
        如果data_type为all 则返回全量数据
        如果data_type为last 则返回最新的k数据最后1条数据 如果只返回一条怕前面一条的数据不完整
        """
        # result = copy.deepcopy(self.kline_data)
        result = {}
        for token in self.tokens:
            result[token] = {}
            for period in self.periods:
                # df= copy.deepcopy(self.kline_data[token][period])
                # df = self.kline_data[token][period].copy()  # 创建副本
                df = self.kline_data[token][period].drop(columns=self.RETURN_COLS_DROP)
                if data_type == 'last':
                    df = df.tail(1)
                df_reset = df.fillna(0).reset_index()
                
                if data_type == 'last':
                    json_array = df_reset.to_json(orient="records")
                    json_list = json.loads(json_array)
                    # 确保列表非空（根据逻辑此时应有1条数据）
                    result_str = json.dumps(json_list[0]) if json_list else "{}"
                    result[token][period] = result_str
                else:
                    result[token][period] = df_reset.to_json(orient="records")

        return result