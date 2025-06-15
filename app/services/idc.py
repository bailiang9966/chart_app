# from IPython import get_ipython
# from IPython.display import display
# %%
import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict, Callable, Optional


class IndicatorCalculator:
    """
    高效指标计算模块，支持灵活添加各类常用指标
    """
    
    # 指标函数默认配置
    DEFAULT_CONFIG = {
        'SMA': {
            'columns': ['SMA'],
            'params': {'column': 'close', 'period': 14} # 修改 kwargs 为 params
        },
        'EMA': {
            'columns': ['EMA'],
            'params': {'column': 'close', 'period': 14} # 修改 kwargs 为 params
        },
        'DEMA': {
            'columns': ['DEMA'],
            'params': {'column': 'close', 'period': 14} # 修改 kwargs 为 params
        },
        'RSI': {
            'columns': ['RSI'],
            'params': {'column': 'close', 'period': 14} # 修改 kwargs 为 params
        },
        'MACD': {
            'columns': ['MACD', 'SIGNAL', 'HISTOGRAM'],
            'params': {'column': 'close', 'fast_period': 12, 
                      'slow_period': 26, 'signal_period': 9} # 修改 kwargs 为 params
        },
        'BOLLINGER_BANDS': {
            'columns': ['BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'],
            'params': {'column': 'close', 'period': 20, 'std_dev': 2.0} # 修改 kwargs 为 params
        },
        'STOCHASTIC': {
            'columns': ['STOCH_K', 'STOCH_D'],
            'params': {'high_column': 'high', 'low_column': 'low', 
                      'close_column': 'close', 'k_period': 14, 'd_period': 3} # 修改 kwargs 为 params
        },
        'KDJ': {
            'columns': ['KDJ_K', 'KDJ_D', 'KDJ_J'],
            'params': {'high_column': 'high', 'low_column': 'low', 
                      'close_column': 'close', 'k_period': 9, 'd_period': 3, 'j_period': 3} # 修改 kwargs 为 params
        },
        'DMI': {
            'columns': ['PLUS_DI', 'MINUS_DI', 'ADX'],
            'params': {'high_column': 'high', 'low_column': 'low', 
                      'close_column': 'close', 'period': 14} # 修改 kwargs 为 params
        },
        'WR': {
            'columns': ['WR'],
            'params': {'high_column': 'high', 'low_column': 'low', 
                      'close_column': 'close', 'period': 14} # 修改 kwargs 为 params
        }
    }
    
    def __init__(self, df: pd.DataFrame, inplace: bool = False):
        """
        参数:
            df: 输入DataFrame
            inplace: 是否直接在原DataFrame上操作（避免复制，提高性能）
        """
        self.df = df if inplace else df.copy()
    
    def add_indicators(self, indicators_list: List[Dict[str, Dict]]) -> 'IndicatorCalculator':
      """
      批量添加指标，使用列表参数，列表内每个字典以指标名做key。
      支持同一个指标多次添加不同参数版本（通过列表元素区分）。
      
      参数:
          indicators_list: 指标配置列表，每个元素是一个字典，以指标名做key。
                          例如: [{'SMA': {'params': {'period': 5}}}, # 修改 kwargs 为 params
                                {'EMA': {'params': {'period': 10}}}, # 修改 kwargs 为 params
                                {'EMA': {'columns': ['EMA_20'], 'params': {'period': 20}}}, # 修改 kwargs 为 params
                                {'MACD': {}}]
      """
      for indicator_config_dict in indicators_list:
          # 由于字典只有一个key，直接获取key和value
          if not indicator_config_dict or len(indicator_config_dict) != 1:
              print("Warning: Invalid indicator configuration in list. Each dictionary should have exactly one key (the indicator name). Skipping.")
              continue
              
          func_name = list(indicator_config_dict.keys())[0]
          config = indicator_config_dict[func_name]
          
          # 获取指标计算函数
          func = getattr(self, func_name, None)
          if func is None:
              print(f"Warning: Indicator function '{func_name}' not found. Skipping.")
              continue
          
          # 获取默认配置
          default_config = self.DEFAULT_CONFIG.get(func_name, {})
          default_columns = default_config.get('columns', [func_name])
          # 修改获取默认参数的键名
          default_params = default_config.get('params', {}) 
          
          # 合并用户配置与默认配置
          # 允许用户在配置中通过 'columns' 指定列名
          user_columns = config.get('columns', default_columns)
          # 修改获取用户参数的键名
          user_params = {**default_params, **config.get('params', {})} 
          
          # 确保传递当前 DataFrame
          user_params['df'] = self.df
          
          # 计算指标
          # 修改传递参数时的字典
          results = func(**user_params) 
          
          # 如果结果不是元组，转为元组
          if not isinstance(results, tuple):
              results = (results,)
          
          # 添加到 DataFrame
          # 检查列名数量是否匹配结果数量
          if len(user_columns) != len(results):
              print(f"Warning: Column names count ({len(user_columns)}) does not match results count ({len(results)}) for indicator '{func_name}'. Using default columns.")
              user_columns = default_columns
              # 如果默认列名也不匹配，则跳过或报错，这里选择跳过
              if len(user_columns) != len(results):
                  print(f"Error: Default column names count ({len(user_columns)}) does not match results count ({len(results)}) for indicator '{func_name}'. Skipping.")
                  continue

          for col, result in zip(user_columns, results):
              self.df[col] = result
          
      return self
    
    def get_df(self) -> pd.DataFrame:
        """获取处理后的 DataFrame"""
        return self.df
    
    # 基础指标计算方法
    @staticmethod
    def SMA(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """简单移动平均线"""
        return df[column].rolling(window=period).mean()
    
    @staticmethod
    def EMA(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """指数移动平均线"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def DEMA(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """双指数移动平均线 (Double Exponential Moving Average)"""
        ema = df[column].ewm(span=period, adjust=False).mean()
        return 2 * ema - ema.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def RSI(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """相对强弱指数"""
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def MACD(df: pd.DataFrame, column: str = 'close', 
             fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
            ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        移动平均收敛发散指标
        
        返回:
            tuple: (MACD线, 信号线, 柱状图)
        """
        ema_fast = df[column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def BOLLINGER_BANDS(df: pd.DataFrame, column: str = 'close', 
                        period: int = 20, std_dev: float = 2.0
                       ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        布林带
        
        返回:
            tuple: (上轨, 中轨, 下轨)
        """
        middle = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def STOCHASTIC(df: pd.DataFrame, high_column: str = 'high', low_column: str = 'low', 
                  close_column: str = 'close', k_period: int = 14, d_period: int = 3
                 ) -> Tuple[pd.Series, pd.Series]:
        """
        随机指标
        
        返回:
            tuple: (K线, D线)
        """
        lowest_low = df[low_column].rolling(window=k_period).min()
        highest_high = df[high_column].rolling(window=k_period).max()
        k = 100 * ((df[close_column] - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    @staticmethod
    def KDJ(df: pd.DataFrame, high_column: str = 'high', low_column: str = 'low', 
           close_column: str = 'close', k_period: int = 9, d_period: int = 3, j_period: int = 3
          ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        KDJ指标 - 修正版本，采用EMA平滑计算
        
        返回:
            tuple: (K线, D线, J线)
        """
        lowest_low = df[low_column].rolling(window=k_period).min()
        highest_high = df[high_column].rolling(window=k_period).max()
        
        # 计算 RSV 值，处理分母为零的情况
        rsv_numerator = df[close_column] - lowest_low
        rsv_denominator = highest_high - lowest_low
        # 避免除以零，将分母为零的地方设为NaN，或者根据实际情况处理
        rsv = 100 * rsv_numerator / rsv_denominator
        rsv = rsv.replace([np.inf, -np.inf], np.nan) # 处理无穷大情况
        
        # 使用 EMA 计算 K 值
        # alpha = 1/k_period 或者 2/(k_period+1) 取决于具体实现，这里使用 1/k_period
        # 初始 K 值为 50
        k = rsv.ewm(alpha=1/d_period, adjust=False).mean() # 这里实际上是平滑 rsv，alpha 通常是 1/D周期

        # 使用 EMA 计算 D 值
        # 初始 D 值为 50
        d = k.ewm(alpha=1/d_period, adjust=False).mean()
        
        # 计算 J 值
        j = 3 * k - 2 * d
        
        # 初始 K 和 D 值通常设为 50，我们可以填充前期的 NaN 值
        # 需要注意rolling窗口的大小，前k_period-1个值会是NaN
        # 我们可以选择填充或者保留NaN
        # 这里我们选择保留NaN，等待足够的数据点
        
        return k, d, j
    
    @staticmethod
    def DMI(df: pd.DataFrame, high_column: str = 'high', low_column: str = 'low', 
           close_column: str = 'close', period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        方向移动指标
        
        返回:
            tuple: (+DI, -DI, ADX)
        """
        plus_dm = df[high_column].diff()
        minus_dm = df[low_column].diff()
        
        plus_dm_condition = (plus_dm > minus_dm) & (plus_dm > 0)
        plus_dm = plus_dm.where(plus_dm_condition, 0)
        
        minus_dm_condition = (minus_dm > plus_dm) & (minus_dm > 0)
        minus_dm = minus_dm.where(minus_dm_condition, 0)
        minus_dm = abs(minus_dm)
        
        high_low = df[high_column] - df[low_column]
        high_close = np.abs(df[high_column] - df[close_column].shift())
        low_close = np.abs(df[low_column] - df[close_column].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / 
                         tr.ewm(alpha=1/period, adjust=False).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / 
                          tr.ewm(alpha=1/period, adjust=False).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return plus_di, minus_di, adx
    
    @staticmethod
    def WR(df: pd.DataFrame, high_column: str = 'high', low_column: str = 'low', 
          close_column: str = 'close', period: int = 14) -> pd.Series:
        """威廉指标"""
        highest_high = df[high_column].rolling(window=period).max()
        lowest_low = df[low_column].rolling(window=period).min()
        return -100 * (highest_high - df[close_column]) / (highest_high - lowest_low)

# # 以下是您的测试代码，需要修改add_indicators的调用方式以匹配新的'params'键
# data_points = 500

# # 模拟收盘价，从100开始，随机波动
# close_prices = 100 + np.random.randn(data_points).cumsum()
# # 确保收盘价不为负
# close_prices[close_prices < 1] = 1

# # 模拟最高价、最低价和成交量
# # 最高价略高于收盘价，最低价略低于收盘价
# high_prices = close_prices * (1 + np.random.rand(data_points) * 0.02) # 随机增加0%到2%
# low_prices = close_prices * (1 - np.random.rand(data_points) * 0.02) # 随机减少0%到2%
# # 确保 low <= close <= high
# high_prices = np.maximum(high_prices, close_prices)
# low_prices = np.minimum(low_prices, close_prices)

# # 模拟成交量，随机波动
# volume = 1000 + np.random.randn(data_points).cumsum() * 100 # 基础成交量加上随机波动
# # 确保成交量不为负
# volume[volume < 100] = 100 # 设定最小成交量

# # 创建DataFrame
# df = pd.DataFrame({
#     'close': close_prices,
#     'high': high_prices,
#     'low': low_prices,
#     'volume': volume
# })

# print("Generated DataFrame with 500 rows:")
# display(df.head())
# display(df.tail())

# # 实例化计算器
# calc = IndicatorCalculator(df)

# # 批量添加指标，使用列表配置，列表内每个字典以指标名做key
# # 修改这里的配置，将 'kwargs' 改为 'params'
# calc.add_indicators([
#     {'SMA': {'params': {'period': 5}}}, # 计算 SMA 5
#     {'EMA': {'params': {'period': 10}}}, # 计算 EMA 10
#     {'EMA': {'columns': ['EMA_20'], 'params': {'period': 20}}}, # 计算 EMA 20，指定列名
#     {'MACD': {}}, # 计算 MACD
#     {'KDJ': {}},
#     {'DMI': {}},
#     {'WR': {}}
# ])

# # 获取结果
# result_df = calc.get_df()
# print(result_df.columns)
# display(result_df)