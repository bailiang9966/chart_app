import pandas as pd
import json
import logging
import copy
from  app.services.ti import TI
from .bnk import get_his_df,get_params,get_interval

class BNREST():


    RETURN_COLS_DROP = ['uf_volume','spot_volume']
    def __init__(self):
        self.his_ready=False
        self.tokens = ['BTC','ETH']
        self.periods = [1,3,5,10,15,30]
        self.kline_data = {
            token: {period: pd.DataFrame() for period in self.periods}
            for token in self.tokens
        }
        self.get_his_kline()
    
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

            ti = TI(df)
            df['EMA20'] = ti.EMA(window=20)
            df['EMA50'] = ti.EMA(window=50)
            df['MACD'], df['SIGNAL'], df['HISTOGRAM'] = ti.MACD()
            df['KDJ_K'], df['KDJ_D'], df['KDJ_J'] = ti.KDJ()
            df['PLUS_DI'], df['MINUS_DI'],_, _= ti.DMI()
            df['WR'] = ti.WR(window=39)
        pass
    
    def get_kline_data(self,data_type='all'):
        """
        如果data_type为all 则返回全量数据
        如果data_type为last 则返回最新的k数据最后2条数据 如果只返回一条怕前面一条的数据不完整
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
