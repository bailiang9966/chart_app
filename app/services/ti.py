import pandas as pd
import numpy as np

class TI:
    """
    一个用于计算各种技术交易指标的类。
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def HHLL(self, high_column: str = 'high', low_column: str = 'low', window: int = 14) -> tuple[pd.Series, pd.Series]:
        """
        计算指定窗口期内的最低低点 (Lowest Low) 和最高高点 (Highest High)。

        Args:
            high_column: 高价所在的列名。
            low_column: 低价所在的列名。
            window: 计算的窗口期。

        Returns:
            一个包含最低低点和最高高点的元组。
        """
        lowest_low = self.df[low_column].rolling(window=window).min()
        highest_high = self.df[high_column].rolling(window=window).max()
        return lowest_low, highest_high

    def EMA(self, column: str = 'close', window: int = 14) -> pd.Series:
        """
        计算指定列的指数移动平均线 (Exponential Moving Average)。

        Args:
            column: 需要计算EMA的列名。
            window: 计算的窗口期。

        Returns:
            包含EMA值的Series。
        """
        return self.df[column].ewm(span=window, adjust=False).mean()

    def SMA(self, column: str = 'close', window: int = 14) -> pd.Series:
        """
        计算指定列的简单移动平均线 (Simple Moving Average)。

        Args:
            column: 需要计算SMA的列名。
            window: 计算的窗口期。

        Returns:
            包含SMA值的Series。
        """
        return self.df[column].rolling(window=window).mean()

    def DEMA(self, column: str = 'close', window: int = 14) -> pd.Series:
        """
        计算指定列的双重指数移动平均线 (Double Exponential Moving Average)。

        Args:
            column: 需要计算DEMA的列名。
            window: 计算的窗口期。

        Returns:
            包含DEMA值的Series。
        """
        ema1 = self.EMA(column, window)
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        dema = 2 * ema1 - ema2
        return dema


    def RSI(self, column: str = 'close', window: int = 14) -> pd.Series:
        """
        计算指定列的相对强弱指数 (Relative Strength Index)。

        Args:
            column: 需要计算RSI的列名。
            window: 计算的窗口期。

        Returns:
            包含RSI值的Series。
        """
        delta = self.df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def MACD(self, close_column: str = 'close', fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算移动平均收敛/发散指标 (Moving Average Convergence Divergence)。

        Args:
            close_column: 收盘价所在的列名。
            fast_period: 快速EMA的周期。
            slow_period: 慢速EMA的周期。
            signal_period: 信号线的EMA周期。

        Returns:
            一个包含MACD线、信号线和MACD柱状图的元组。
        """
        ema_fast = self.df[close_column].ewm(span=fast_period, adjust=False).mean()
        ema_slow = self.df[close_column].ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_line, signal_line, macd_histogram

    def BB(self, column: str = 'close', window: int = 20, num_std: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算布林带 (Bollinger Bands)。

        Args:
            column: 需要计算布林带的列名。
            window: 计算均值和标准差的窗口期。
            num_std: 标准差的倍数。

        Returns:
            一个包含布林带中轨、上轨和下轨的元组。
        """
        sma = self.SMA(column=column, window=window)
        rolling_std = self.df[column].rolling(window=window).std()
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        return sma, upper_band, lower_band

    def STOCHASTIC_OSCILLATOR(self, high_column: str = 'high', low_column: str = 'low', close_column: str = 'close', k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
        """
        计算随机震荡指标 (Stochastic Oscillator)。

        Args:
            high_column: 高价所在的列名。
            low_column: 低价所在的列名。
            close_column: 收盘价所在的列名。
            k_window: 计算%K线的窗口期。
            d_window: 计算%D线的窗口期。

        Returns:
            一个包含%K线和%D线的元组。
        """
        lowest_low, highest_high = self.HHLL(high_column=high_column, low_column=low_column, window=k_window)
        k_line = 100 * ((self.df[close_column] - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(window=d_window).mean()
        return k_line, d_line

    def WR(self, high_column: str = 'high', low_column: str = 'low', close_column: str = 'close', window: int = 14) -> pd.Series:
        """
        计算威廉指标 (Williams %R)。

        Args:
            high_column: 高价所在的列名。
            low_column: 低价所在的列名。
            close_column: 收盘价所在的列名。
            window: 计算的窗口期。

        Returns:
            包含威廉指标值的Series。
        """
        lowest_low, highest_high = self.HHLL(high_column=high_column, low_column=low_column, window=window)
        wr = -100 * ((highest_high - self.df[close_column]) / (highest_high - lowest_low))
        return wr

    def KDJ(self, high_column: str = 'high', low_column: str = 'low', close_column: str = 'close', n: int = 9, m1: int = 3, m2: int = 3) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算随机指标 (KDJ)。

        Args:
            high_column: 高价所在的列名。
            low_column: 低价所在的列名。
            close_column: 收盘价所在的列名。
            n: 计算RSV的窗口期。
            m1: 计算K线的窗口期。
            m2: 计算D线的窗口期。

        Returns:
            一个包含K线、D线和J线的元组。
        """
        lowest_low, highest_high = self.HHLL(high_column=high_column, low_column=low_column, window=n)
        rsv = 100 * ((self.df[close_column] - lowest_low) / (highest_high - lowest_low))

        k_line = rsv.ewm(span=m1, adjust=False).mean()
        d_line = k_line.ewm(span=m2, adjust=False).mean()

        j_line = 3 * k_line - 2 * d_line
        return k_line, d_line, j_line

    def DONCHIAN_CHANNEL(self, high_column: str = 'high', low_column: str = 'low', window: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算唐奇安通道 (Donchian Channel)。

        Args:
            high_column: 高价所在的列名。
            low_column: 低价所在的列名。
            window: 计算的窗口期。

        Returns:
            一个包含唐奇安通道下轨、中轨和上轨的元组。
        """
        lower_band, upper_band = self.HHLL(high_column=high_column, low_column=low_column, window=window)
        middle_band = (upper_band + lower_band) / 2
        return lower_band, middle_band, upper_band

    def DMI(self, high_column: str = 'high', low_column: str = 'low', close_column: str = 'close', n: int = 14, m: int = 14) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        计算动向指数 (Directional Movement Index) 和平均动向指数 (Average Directional Index)。

        Args:
            high_column: 高价所在的列名。
            low_column: 低价所在的列名。
            close_column: 收盘价所在的列名。
            n: 计算DI和TR的窗口期。
            m: 计算ADX的窗口期。

        Returns:
            一个包含+DI、-DI、DX和ADX的元组。
        """
        high = self.df[high_column]
        low = self.df[low_column]
        close = self.df[close_column]

        plus_dm = (high - high.shift(1)).apply(lambda x: max(x, 0) if x > 0 else 0)
        minus_dm = (low.shift(1) - low).apply(lambda x: max(x, 0) if x > 0 else 0)

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        smooth_plus_dm = plus_dm.ewm(span=n, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(span=n, adjust=False).mean()
        smooth_tr = tr.ewm(span=n, adjust=False).mean()

        plus_di = (smooth_plus_dm / smooth_tr) * 100
        minus_di = (smooth_minus_dm / smooth_tr) * 100

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        dx = dx.replace([float('inf'), -float('inf')], 0).fillna(0)

        adx = dx.ewm(span=m, adjust=False).mean()

        return plus_di, minus_di, dx, adx

    def CROSSOVER(self, series1: pd.Series, series2: pd.Series) -> pd.Series:
        """
        判断两个Series是否发生向上交叉。

        Args:
            series1: 第一个Series。
            series2: 第二个Series。

        Returns:
            一个布尔类型的Series，表示是否发生向上交叉。
        """
        crossover = (series1.shift(1) <= series2.shift(1)) & (series1 > series2)
        return crossover

    def CONT_BOOL_COUNT(self, bool_column: str) -> pd.Series:
        """
        计算连续的True或False的计数。

        Args:
            bool_column: 布尔类型的列名。

        Returns:
            一个整数类型的Series，表示连续True或False的计数。
            正数表示连续True的次数，负数表示连续False的次数。
        """
        count_col = pd.Series(index=self.df.index, dtype=int)

        if self.df[bool_column].iloc[0]:
            count_col.iloc[0] = 1
        else:
            count_col.iloc[0] = -1

        shifts = self.df[bool_column] != self.df[bool_column].shift(1)
        shift_indices = shifts[shifts].index

        start_index = self.df.index[0]
        start_iloc = self.df.index.get_loc(start_index)

        for end_index in shift_indices:
            end_iloc = self.df.index.get_loc(end_index)
            if self.df[bool_column].loc[end_index]:
                count_col.iloc[start_iloc:end_iloc + 1] = range(1, (end_iloc - start_iloc) + 2)
            else:
                count_col.iloc[start_iloc:end_iloc + 1] = range(-1, -(end_iloc - start_iloc) - 2, -1)
            start_index = end_index
            start_iloc = end_iloc # Update start_iloc for the next iteration

        if start_index < self.df.index[-1]:
            start_iloc = self.df.index.get_loc(start_index)
            end_iloc = len(self.df.index) - 1 # Last index position

            if self.df[bool_column].iloc[-1]:
                count_col.iloc[start_iloc:] = range(1, (end_iloc - start_iloc) + 2)
            else:
                count_col.iloc[start_iloc:] = range(-1, -(end_iloc - start_iloc) - 2, -1)

        return count_col






    def OVER_BS(self, indicator_series: pd.Series, thresholds: list) -> pd.Series:
        """
        根据指标值和传入的超卖超买阈值列表判断超买超卖，并用 -1, 0, 1 表示。

        Args:
            indicator_series: 需要判断的指标Series (例如RSI或WR)。
            thresholds: 包含两个元素的列表，第一个元素是超卖阈值，第二个元素是超买阈值。例如 [30, 70]。

        Returns:
            一个Series，其中 1 表示超买，-1 表示超卖，0 表示正常。
        """
        if len(thresholds) != 2:
            raise ValueError("thresholds 必须是一个包含两个元素的列表：[超卖阈值, 超买阈值]")

        oversold_threshold = thresholds[0]
        overbought_threshold = thresholds[1]

        conditions = [
            indicator_series >= overbought_threshold,
            indicator_series <= oversold_threshold
        ]
        choices = [1, -1]
        return np.select(conditions, choices, default=0)
    def COMPARE_BS(self, series1: pd.Series, compare_with) -> pd.Series:
        """
        比较一个Series与另一个Series或一个数值，并返回 -1, 0, 1。

        如果 compare_with 是一个 Series:
            如果 series1 > compare_with，则返回 1。
            如果 series1 < -compare_with，则返回 -1。
            否则返回 0 (包括 compare_with 为 NaN 的情况)。

        如果 compare_with 是一个数值:
            如果 series1 > compare_with，则返回 1。
            如果 series1 < -compare_with，则返回 -1。
            否则返回 0 (包括 series1 为 NaN 的情况)。

        Args:
            series1: 需要比较的 Series。
            compare_with: 用于比较的 Series 或数值。

        Returns:
            一个Series，其中 1 表示满足第一个条件，-1 表示满足第二个条件，0 表示其它情况。
        """
        if not isinstance(series1, pd.Series):
            raise TypeError("series1 必须是一个 pandas Series。")

        if isinstance(compare_with, pd.Series):
            # 比较两个 Series
            if len(series1) != len(compare_with):
                 raise ValueError("series1 和 compare_with Series 长度必须相同。")

            conditions = [
                (series1 > compare_with) & (~compare_with.isna()), # series1 > compare_with 且 compare_with 不是 NaN
                (series1 < -compare_with) & (~compare_with.isna()) # series1 < -compare_with 且 compare_with 不是 NaN
            ]
            choices = [1, -1]
            return np.select(conditions, choices, default=0)

        elif isinstance(compare_with, (int, float)):
            # 比较 Series 和数值
            conditions = [
                series1 > compare_with,  # series1 > compare_with (数值)
                series1 < -compare_with # series1 < -compare_with (数值)
            ]
            choices = [1, -1]
            # 处理 series1 为 NaN 的情况，np.select 默认会将 NaN 条件的结果设置为 default
            return np.select(conditions, choices, default=0)

        else:
            raise TypeError("compare_with 必须是一个 pandas Series 或数值。")

    def STOCHRSI(self, column: str = 'close', rsi_window: int = 14, stoch_window: int = 14, k_window: int = 3, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
        """
        计算随机相对强弱指数 (Stochastic RSI)。

        Args:
            column: 需要计算StochRSI的列名（通常是收盘价）。
            rsi_window: 计算RSI的窗口期。
            stoch_window: 计算StochRSI的窗口期（在该窗口期内计算最高RSI和最低RSI）。
            k_window: 计算StochRSI %K 线的 SMA 窗口期。
            d_window: 计算StochRSI %D 线的 SMA 窗口期。

        Returns:
            一个包含StochRSI %K线和%D线的元组。
        """
        # 1. 计算 RSI
        rsi_series = self.RSI(column=column, window=rsi_window)

        # 2. 计算指定窗口期内的最高 RSI 和最低 RSI
        lowest_rsi = rsi_series.rolling(window=stoch_window).min()
        highest_rsi = rsi_series.rolling(window=stoch_window).max()

        # 3. 计算 StochRSI 的 %K
        # 避免除以零的情况
        stochrsi_k = 100 * ((rsi_series - lowest_rsi) / (highest_rsi - lowest_rsi))
        # 填充 NaN 值，避免 inf/nan
        stochrsi_k = stochrsi_k.fillna(0).replace([np.inf, -np.inf], 0)


        # 4. 计算 StochRSI 的 %K 线的 SMA (TradingView 默认 K 线也是平滑的)
        stochrsi_k_smooth = stochrsi_k.rolling(window=k_window).mean()

        # 5. 计算 StochRSI 的 %D 线 (TradingView 默认 D 线是平滑后的 K 线的 SMA)
        stochrsi_d = stochrsi_k_smooth.rolling(window=d_window).mean()


        return stochrsi_k_smooth, stochrsi_d
    def SLOPE(self, column: str = 'close', window: int = 14) -> pd.Series:
        """
        计算指定列在指定窗口期内的斜率。

        Args:
            column: 需要计算斜率的列名。
            window: 计算斜率的窗口期。

        Returns:
            包含斜率值的Series。
        """
        y = self.df[column]
        n = window
        x_sum = (n - 1) * n // 2  # Sum of x_i from 0 to n-1
        x_sq_sum = (n - 1) * n * (2 * n - 1) // 6  # Sum of x_i^2 from 0 to n-1

        # 计算滚动求和
        sum_y = y.rolling(window=n).sum()
        sum_xy = (y * pd.Series(range(n)).rolling(window=n).sum()).rolling(window=n).sum() # 这个地方需要修正

        # 修正 sum_xy 的计算，需要对每个窗口内的 x_i 进行加权求和
        # 考虑使用 dot product 或者更优化的方法
        # 以下是一种可能的实现方式，但可能不是最优的，需要进一步测试和优化
        sum_xy = y.rolling(window=n).apply(lambda w: (w * np.arange(len(w))).sum(), raw=True)



        # 计算斜率
        numerator = n * sum_xy - x_sum * sum_y
        denominator = n * x_sq_sum - x_sum**2

        # 避免除以零
        slope = numerator / denominator
        slope = slope.replace([np.inf, -np.inf], np.nan) # 将无穷大替换为 NaN

        return slope

# if __name__ == '__main__':

#     end_date = "2020-01-01"
#     ts_start = date_to_ms(end_date)
#     base_tf = 5
#     df = query_kline(f"kline_BTC_{base_tf}",limit=10000)
#     start = time.time()
#     hid_cols = ['open','high','low','close']
#     # 创建 TI 对象
#     ti = TI(df)
#     df['volume'] = df['spot_volume']+df['uf_volume']
#     df['delta'] = df['spot_delta']+df['uf_delta']
#     hid_cols.extend(['uf_buy_vol', 'uf_sell_vol', 'spot_buy_vol', 'spot_sell_vol','volume','delta','uf_volume','spot_volume','uf_delta','spot_delta'])


#     start_time = time.time()
#     df['vol_ratio'] = df['volume'] / ti.EMA(window=50)
#     end_time = time.time()
#     print(f"计算 vol_ratio (EMA) 耗时: {end_time - start_time:.4f} 秒")

#     start_time = time.time()
#     df['cvd'] = df['delta'].cumsum()
#     df['uf_cvd'] = df['uf_delta'].cumsum()
#     df['spot_cvd'] = df['spot_delta'].cumsum()
#     end_time = time.time()
#     print(f"计算 CVD 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['cvd', 'uf_cvd', 'spot_cvd'])


#     start_time = time.time()
#     df['delta_abs'] = df['delta'].abs()
#     df['uf_delta_abs'] = df['uf_delta'].abs()
#     df['spot_delta_abs'] = df['spot_delta'].abs()
#     end_time = time.time()
#     print(f"计算 Delta 绝对值相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['delta_abs', 'uf_delta_abs', 'spot_delta_abs'])

#     start_time = time.time()
#     df['EMA_20'] = ti.EMA(window=20)
#     df['EMA_50'] = ti.EMA(window=50)
#     df['EMA_80'] = ti.EMA(window=80)
#     df['DEMA_25'] = ti.DEMA(window=25)
#     end_time = time.time()
#     print(f"计算 EMA 和 DEMA 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['EMA_20', 'EMA_50', 'EMA_80', 'DEMA_25'])

#     start_time = time.time()
#     ema_subset = df[['EMA_20', 'EMA_50', 'EMA_80', 'DEMA_25']]
#     ema_subset_ranked = ema_subset.rank(axis=1, method='min', ascending=False)
#     df['ema20_rank'] = ema_subset_ranked['EMA_20']
#     df['ema50_rank'] = ema_subset_ranked['EMA_50']
#     df['ema80_rank'] = ema_subset_ranked['EMA_80']
#     df['dema25_rank'] = ema_subset_ranked['DEMA_25']
#     df['ema_std'] = ema_subset.std(axis=1)
#     df['ema_range'] = ema_subset.max(axis=1) - ema_subset.min(axis=1)
#     df['c_pct_dev_ema20'] = ((df['close'] - df['EMA_20']) / df['EMA_20']) * 100
#     df['c_pct_dev_ema50'] = ((df['close'] - df['EMA_50']) / df['EMA_50']) * 100
#     df['c_pct_dev_ema80'] = ((df['close'] - df['EMA_80']) / df['EMA_80']) * 100
#     df['c_pct_dev_dema25'] = ((df['close'] - df['DEMA_25']) / df['DEMA_25']) * 100
#     end_time = time.time()
#     print(f"计算 EMA 派生指标耗时: {end_time - start_time:.4f} 秒")


#     start_time = time.time()
#     df['is_up'] = df['close'] > df['open']
#     df['cont_count'] = ti.CONT_BOOL_COUNT('is_up')
#     end_time = time.time()
#     print(f"计算 is_up 和 cont_count 耗时: {end_time - start_time:.4f} 秒")


#     start_time = time.time()
#     df['MACD_line'], df['MACD_sig'], df['MACD_hist'] = ti.MACD()
#     hist_abs = abs(df['MACD_hist'])
#     df['macd_hist_q70'] = hist_abs.rolling(window=200).quantile(0.7)
#     df['hist_mean'] = abs(df['MACD_hist']/df['macd_hist_q70'])
#     df['macd_hist_oo'] = ti.COMPARE_BS(df['MACD_hist'],df['macd_hist_q70'])
#     end_time = time.time()
#     print(f"计算 MACD 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['macd_hist_q70'])

#     start_time = time.time()
#     df['涨幅'] = (df['close'] - df['open']) / df['open'] * 100
#     df['振幅'] = (df['high'] - df['low']) / df['open'] * 100
#     end_time = time.time()
#     print(f"计算 涨幅 和 振幅 耗时: {end_time - start_time:.4f} 秒")

#     # TODO wr/rsi等超买超卖
#     # TODO 上涨且超过前一个高点
#     # TODO 斜率?通道
#     # TODO 所有指标必须做归一化处理


#     #rsi 超卖超买 金叉 上下 中轴
#     start_time = time.time()
#     df['RSI_7'] = ti.RSI(window=7)
#     df['RSI_14'] = ti.RSI(window=14)
#     df['RSI7_OO'] = ti.OVER_BS(df['RSI_7'], thresholds=[30, 70])
#     df['RSI14_OO'] = ti.OVER_BS(df['RSI_14'], thresholds=[30, 70])
#     df['RSI7_OVER_MID'] = df['RSI_7']>50
#     df['RSI14_OVER_MID'] = df['RSI_14']>50
#     df['RSI_CROSS'] = ti.CROSSOVER(df['RSI_7'], df['RSI_14'])
#     df['RSI_KD'] = df['RSI_7']>df['RSI_14']
#     df['RSI_K-D'] = df['RSI_7']-df['RSI_14']
#     end_time = time.time()
#     print(f"计算 RSI 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['RSI_7', 'RSI_14'])


#     #stochrsi 超买超卖 金叉 上下 中轴
#     start_time = time.time()
#     df['StochRSI_K'], df['StochRSI_D'] = ti.STOCHRSI(rsi_window=14, stoch_window=14, k_window=3, d_window=3)
#     df['SRSI_OO'] = ti.OVER_BS(df['StochRSI_K'], thresholds=[20, 80])
#     df['SRSI_OVER_MID'] = df['StochRSI_K']>50
#     df['SRSI_CROSS'] = ti.CROSSOVER(df['StochRSI_K'], df['StochRSI_D'])
#     df['SRSI_KD'] = df['StochRSI_K']>df['StochRSI_D']
#     end_time = time.time()
#     print(f"计算 StochRSI 相关指标耗时: {end_time - start_time:.4f} 秒")
#     # hid_cols.extend(['StochRSI_K', 'StochRSI_D'])

#     #WR
#     start_time = time.time()
#     df['WR14'] = ti.WR()
#     df['WR39'] = ti.WR(window = 39)
#     df['WR_OVER_MID_14'] = df['WR14']>-50
#     df['WR_OVER_MID_39'] = df['WR39']>-50
#     df['WR14_OO'] = ti.OVER_BS(df['WR14'], thresholds=[-80, -20])
#     df['WR39_OO'] = ti.OVER_BS(df['WR39'], thresholds=[-80, -20])
#     df['WR_CROSS'] = ti.CROSSOVER(df['WR14'], df['WR39'])
#     df['WR_KD'] = df['WR14']>df['WR39']

#     end_time = time.time()
#     print(f"计算 WR 相关指标耗时: {end_time - start_time:.4f} 秒")

#     # kdj
#     start_time = time.time()
#     df['KDJ_K'], df['KDJ_D'], df['KDJ_J'] = ti.KDJ(n=49)
#     df['KDJ_CROSS'] = ti.CROSSOVER(df['KDJ_J'], df['KDJ_K'])
#     df['KDJ_JK'] = df['KDJ_J']>df['KDJ_K']
#     df['KDJ_J-K'] = df['KDJ_J']-df['KDJ_K']
#     end_time = time.time()
#     print(f"计算 KDJ 相关指标耗时: {end_time - start_time:.4f} 秒")

#     #DMI
#     start_time = time.time()
#     df['+DI'], df['-DI'], _, _ = ti.DMI(n=35)
#     df['DI_COMPARE'] = df['+DI']>df['-DI']
#     end_time = time.time()
#     print(f"计算 DMI 相关指标耗时: {end_time - start_time:.4f} 秒")


#     #通道类型
#     start_time = time.time()
#     df['BB_mid'], df['BB_upper'], df['BB_lower'] = ti.BB()
#     df['BB_width'] = df['BB_upper'] - df['BB_lower']/df['BB_mid']
#     df['BB_width_change'] = df['BB_width'].pct_change()
#     df['SLOPE_10'] = ti.SLOPE(window=10)
#     end_time = time.time()
#     print(f"计算 BB 和 SLOPE 耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['BB_mid', 'BB_upper', 'BB_lower',])

#     start_time = time.time()
#     df['DC_L'], df['DC_M'], df['DC_H'] = ti.DONCHIAN_CHANNEL()
#     df['BREAK_UP'] = df['close'] > df['DC_H']
#     df['BREAK_DOWN'] = df['close'] < df['DC_L']
#     end_time = time.time()
#     print(f"计算 Donchian Channel 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['DC_L', 'DC_M', 'DC_H'])

#     #DELTA
#     start_time = time.time()
#     delta_abs = abs(df['delta'])
#     df['delta_q70'] = delta_abs.rolling(window=200).quantile(0.7)
#     df['delta_oo'] = ti.COMPARE_BS(df['delta'],df['delta_q70'])
#     df['delta_mean'] = delta_abs/df['delta_q70']
#     end_time = time.time()
#     print(f"计算 Delta 相关指标耗时: {end_time - start_time:.4f} 秒")

#     #uf_delta
#     start_time = time.time()
#     uf_delta_abs = abs(df['uf_delta'])
#     df['uf_delta_q70'] = uf_delta_abs.rolling(window=200).quantile(0.7)
#     df['uf_delta_oo'] = ti.COMPARE_BS(df['uf_delta'],df['uf_delta_q70'])
#     df['uf_delta_mean'] = uf_delta_abs/df['uf_delta_q70']
#     end_time = time.time()
#     print(f"计算 UF Delta 相关指标耗时: {end_time - start_time:.4f} 秒")

#     #spot_delta
#     start_time = time.time()
#     spot_delta_abs = abs(df['spot_delta'])
#     df['spot_delta_q70']  =  spot_delta_abs.rolling(window=200).quantile(0.7)
#     df['spot_delta_oo'] = ti.COMPARE_BS(df['spot_delta'],df['spot_delta_q70'])
#     df['spot_delta_mean'] = spot_delta_abs/df['spot_delta_q70']
#     end_time = time.time()
#     print(f"计算 Spot Delta 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['delta_q70', 'uf_delta_q70', 'spot_delta_q70'])

#     #cvd
#     start_time = time.time()
#     df['CVD_EMA5'] = ti.EMA(window=5,column = 'cvd')
#     df['CVD_EMA10'] = ti.EMA(window=10,column = 'cvd')
#     df['CVD_CROSS'] = ti.CROSSOVER(df['CVD_EMA5'], df['CVD_EMA10'])
#     df['CVD_KD'] = df['CVD_EMA5']>df['CVD_EMA10']
#     df['CVD_SLOPE'] = ti.SLOPE(window=10,column = 'cvd')
#     end_time = time.time()
#     print(f"计算 CVD EMA 和 SLOPE 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['CVD_EMA5', 'CVD_EMA10'])


#     #uf_cvd
#     start_time = time.time()
#     df['UF_CVD_EMA5'] = ti.EMA(window=5,column = 'uf_cvd')
#     df['UF_CVD_EMA10'] = ti.EMA(window=10,column = 'uf_cvd')
#     df['UF_CVD_CROSS'] = ti.CROSSOVER(df['UF_CVD_EMA5'], df['UF_CVD_EMA10'])
#     df['UF_CVD_KD'] = df['UF_CVD_EMA5']>df['UF_CVD_EMA10']
#     df['UF_CVD_SLOPE'] = ti.SLOPE(window=10,column = 'uf_cvd')
#     end_time = time.time()
#     print(f"计算 UF CVD EMA 和 SLOPE 相关指标耗时: {end_time - start_time:.4f} 秒")
#     hid_cols.extend(['UF_CVD_EMA5', 'UF_CVD_EMA10'])

#     #spot_cvd
#     start_time = time.time()
#     df['SPOT_CVD_EMA5'] = ti.EMA(window=5,column = 'spot_cvd')
#     df['SPOT_CVD_EMA10'] = ti.EMA(window=10,column = 'spot_cvd')
#     df['SPOT_CVD_CROSS'] = ti.CROSSOVER(df['SPOT_CVD_EMA5'], df['SPOT_CVD_EMA10'])
#     df['SPOT_CVD_KD'] = df['SPOT_CVD_EMA5']>df['SPOT_CVD_EMA10']
#     df['SPOT_CVD_SLOPE'] = ti.SLOPE(window=10,column = 'spot_cvd')
#     end_time = time.time()
#     print(f"计算 Spot CVD EMA 相关{end_time - start_time:.4f} 秒")
#     hid_cols.extend(['SPOT_CVD_EMA5', 'SPOT_CVD_EMA10'])



#     show_cols = [col for col in df.columns if col not in hid_cols]
#     print(show_cols)
#     display(df[show_cols])
#     df.drop(columns=hid_cols,  inplace=True)
#     df.dropna(inplace=True)
#     df['next'] = df['is_up'].shift(-1)
#     df.drop(df.index[-1], inplace=True)
#     df['next'] = df['next'].astype(int)