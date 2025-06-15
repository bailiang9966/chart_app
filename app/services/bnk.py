import requests
import pandas as pd
import logging

UF_KLINE_URL= 'https://fapi.binance.com/fapi/v1/klines'
SPOT_KLINE_URL= 'https://api.binance.com/api/v3/klines'

def get_interval(minute:int):
    if minute < 60:
        return f'{minute}m'
    elif minute <1440:
        # 对于60分钟，显示为1h；对于240分钟，显示为4h
        return f'{int(minute/60)}h'
    elif minute == 1440:
        return '1d'
    else:
        raise ValueError(f"周期分钟数错误:{minute}")
def get_params(token,minute,limit=500,start_ts=0,end_ts=0):
    symbol = token.upper() + 'USDT'
    interval = get_interval(minute)
    params = {
        'symbol': symbol,  # 交易对
        'interval': interval,  # 时间间隔
    }
    
    if start_ts==0 and end_ts==0:
            params.update({
                'limit': limit,  # 数据量
            })
            pass
    else:
        if start_ts==0 :
            start_ts = end_ts  - limit *minute*60000
        elif end_ts == 0:
            end_ts = start_ts + limit *minute*60000 
        params.update({
            'endTime':end_ts,
            'startTime':start_ts,
        })
    return params

def get_his_df(params:dict):
 

    df_cols = ['ts', 'open', 'high', 'low', 'close', 'volume', 'ts_end', 'quote_asset_volume', 'number_of_trades', 'buy_vol', 'taker_buy_quote_asset_volume', 'ignore']

    spot_data = requests.get(SPOT_KLINE_URL, params=params).json()
    uf_data = requests.get(UF_KLINE_URL, params=params).json()

    uf_df = pd.DataFrame(uf_data, columns=df_cols)
    spot_df = pd.DataFrame(spot_data, columns=df_cols)

    if len(uf_df)==0 or len(spot_df)==0:
        logging.info(params)
        raise Exception(f"查询数据为空 {len(uf_df)} : {len(spot_df)}")
    if len(uf_df)!= len(spot_df):
        raise Exception(f"查询数据量不一致 {len(uf_df)} : {len(spot_df)}")

    uf_df.drop(columns=['ts_end', 'quote_asset_volume', 'number_of_trades', 'taker_buy_quote_asset_volume', 'ignore'], inplace=True)
    

    float_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_vol']
    uf_df[float_cols] = uf_df[float_cols].astype(float)
    spot_df[float_cols] = spot_df[float_cols].astype(float)

    spot_df['delta'] = 2*spot_df['buy_vol']-spot_df['volume']


    uf_df['delta'] = 2*uf_df['buy_vol']-uf_df['volume']

    spot_df = spot_df[['ts','delta','volume']]

    spot_df.rename(columns={ 'volume':'spot_volume','delta': 'spot_delta'}, inplace=True)
    uf_df.rename(columns={'volume':'uf_volume','delta': 'uf_delta',}, inplace=True)
    df = pd.merge(uf_df, spot_df, on='ts', how='inner')
    # df['volume'] = df['uf_volume'] + df['spot_volume']
    # df['delta']=df['spot_delta']+df['uf_delta']

    # df = df[['ts', 'open', 'high', 'low', 'close','volume','delta','spot_volume','uf_volume','spot_delta','uf_delta']]
    df = df[['ts', 'open', 'high', 'low', 'close','uf_volume','uf_delta','spot_volume','spot_delta']]

    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df['ts'] = df['ts'] + pd.Timedelta(hours=8) 
    # df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.tz_localize(None) 
    df.set_index('ts', inplace=True)
    # print(df.columns)
    return df

if __name__ == '__main__':
    # start_ts = 
    params = get_params('btc',5)
    df = get_his_df(params)
    print(df.tail())


    # # 判断是否上涨
    # df['is_up'] = df['close'] > df['close'].shift(1)

    # # 创建分组标识
    # df['group'] = (~df['is_up']).cumsum()

    # # 计算每个连续上涨组的长度
    # up_counts = df[df['is_up']].groupby('group').size()

    # # 最大连续上涨期数
    # max_up_streak = up_counts.max()

    # print("最大连续上涨期数:", max_up_streak)
    # print("各连续上涨期数分布:")
    # print(up_counts)