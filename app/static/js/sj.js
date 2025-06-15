
var data_init = false;
var is_syncing = false;// 用于标记是否正在同步
const commonOptions = {
    timeScale: {
        timeVisible: true,
        secondsVisible: false,
    },
    localization: {
        locale: 'zh-CN',
        dateFormat: 'MM-dd',
      },
    rightPriceScale: {
        minimumWidth: 100
    },

    grid: {
        vertLines: { visible: false },
        horzLines: { visible: false }
    },
    crossHair: { mode: 0 } // 修改为2，显示十字光标
};

const socket = io();


// 存储每个周期对应的图表实例
const charts = {};

// 监听 kline_init 事件
socket.on('kline_init', function (data) {

    const token = $('#token-group .active').data('value');
    const minutes = $('#interval-group .control-btn.active').map(function () {
        return $(this).data('value');
    }).get();

    minutes.forEach(minute => {
        const containerId = `chart-${token}-${minute}`;

        // ---------------- 主图初始化 ----------------
        const mainChart = LightweightCharts.createChart(
            document.getElementById(`${containerId}-main`),
            { width: document.getElementById(containerId).offsetWidth, height: 260 }
        );
        mainChart.applyOptions(commonOptions);
        const mainSeries = mainChart.addCandlestickSeries({ upColor: '#26a69a', downColor: '#ef5350' });
        const ema20Series = mainChart.addLineSeries({ color: '#FF9800' ,lineWidth: 1 });
        const ema50Series = mainChart.addLineSeries({ color: '#2196F3' ,lineWidth: 1  });

        // ---------------- WR副图初始化 ----------------
        const wrChart = LightweightCharts.createChart(
            document.getElementById(`${containerId}-wr`),
            { width: document.getElementById(containerId).offsetWidth, height: 150 }
        );
        wrChart.applyOptions(commonOptions);
        const wrSeries = wrChart.addLineSeries({ color: '#9C27B0' });
        wrSeries.createPriceLine({ price: -50, color: '#3179F5', lineStyle: 2 });

        // ---------------- MACD副图初始化（含3个子系列） ----------------
        const macdChart = LightweightCharts.createChart(
            document.getElementById(`${containerId}-macd`),
            { width: document.getElementById(containerId).offsetWidth, height: 150 }
        );
        macdChart.applyOptions(commonOptions); 
        const macdMainSeries = macdChart.addLineSeries({ color: '#FF5722', lineWidth: 2 });
        const macdSignalSeries = macdChart.addLineSeries({ color: '#2196F3', lineWidth: 1 });
       
        const macdHistSeries = macdChart.addHistogramSeries({ 
            lineWidth: 1, 
            
        });


        // ---------------- KDJ副图初始化（含3个子系列） ----------------
        const kdjChart = LightweightCharts.createChart(
            document.getElementById(`${containerId}-kdj`),
            { width: document.getElementById(containerId).offsetWidth, height: 150}
        );
        kdjChart.applyOptions(commonOptions); 
        const kdjKSeries = kdjChart.addLineSeries({ color: '#FF9800', lineWidth: 2 });
        const kdjDSeries = kdjChart.addLineSeries({ color: '#795548', lineWidth: 1 });
        const kdjJSeries = kdjChart.addLineSeries({ color: '#9C27B0', lineWidth: 1 });

        // ---------------- DMI副图初始化（含3个子系列） ----------------
        const dmiChart = LightweightCharts.createChart(
            document.getElementById(`${containerId}-dmi`),
            { width: document.getElementById(containerId).offsetWidth, height: 150 }
        );
        dmiChart.applyOptions(commonOptions); 
        const plusDISeries = dmiChart.addLineSeries({ color: '#26a69a', lineWidth: 2 });
        const minusDISeries = dmiChart.addLineSeries({ color: 'red', lineWidth: 2 });
        // const adxSeries = dmiChart.addLineSeries({ color: '#607D8B', lineWidth: 1 });
        // mainChart.applyOptions({ timeScale: { visible: false } });
        // wrChart.applyOptions({ timeScale: { visible: false } });
        // macdChart.applyOptions({ timeScale: { visible: false } });
        // kdjChart.applyOptions({ timeScale: { visible: false } });
        // ---------------- 时间轴同步逻辑 ----------------
        const syncTimeScale = (sourceChart, targetCharts) => {
            sourceChart.timeScale().subscribeVisibleLogicalRangeChange(range => {
                if (!is_syncing) {
                    is_syncing = true;
                    targetCharts.forEach(chart => chart.timeScale().setVisibleLogicalRange(range));
                    is_syncing = false;
                }
            });
        };
        syncTimeScale(mainChart, [wrChart, macdChart, kdjChart, dmiChart]);
        [wrChart, macdChart, kdjChart, dmiChart].forEach(chart => {
            syncTimeScale(chart, [mainChart, ...[wrChart, macdChart, kdjChart, dmiChart].filter(c => c !== chart)]);
        });

        // 删除从这里开始的所有十字光标同步代码
        // 添加十字光标同步
        // const syncCrosshair = (sourceChart, targetCharts, sourceContainer) => {
        //     sourceChart.subscribeCrosshairMove(param => {
        //         if (param.time && !is_syncing) {
        //             is_syncing = true;
        //             targetCharts.forEach((chart) => {
        //                 // 只保留十字线位置同步，移除series相关代码
        //                 chart.setCrosshairPosition(param.time, param.price);
        //             });
        //             setTimeout(() => { is_syncing = false; }, 10);
        //         }
        //     });
        // };

        // // 为主图和所有副图设置十字光标同步
        // const allCharts = [mainChart, wrChart, macdChart, kdjChart, dmiChart];
        // const chartContainers = [
        //     `${containerId}-main`,
        //     `${containerId}-wr`,
        //     `${containerId}-macd`,
        //     `${containerId}-kdj`,
        //     `${containerId}-dmi`
        // ];

        // allCharts.forEach((chart, index) => {
        //     syncCrosshair(chart, allCharts.filter(c => c !== chart), chartContainers[index]);
        // });
        // ---------------- 初始化数据绑定 ----------------
        
        kd = JSON.parse(data[token][minute]); 
        
        const kData = kd.map(item => ({ time: item.ts / 1000, open: item.open, close: item.close, high: item.high, low: item.low }));
        mainSeries.setData(kData);
        ema20Series.setData(kd.map(item => ({ time: item.ts / 1000, value: item.EMA20 })));
        ema50Series.setData(kd.map(item => ({ time: item.ts / 1000, value: item.EMA50 })));
        wrSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.WR })));
        macdMainSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.MACD })));
        macdSignalSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.SIGNAL })));
        macdHistSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.HISTOGRAM,color: item.HISTOGRAM >0 ? '#26a69a' : "red"})));
        kdjKSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.KDJ_K })));
        kdjDSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.KDJ_D })));
        kdjJSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.KDJ_J })));
        plusDISeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.PLUS_DI })));
        minusDISeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.MINUS_DI })));
        // adxSeries.setData(data[minute].map(item => ({ time: item.ts / 1000, value: item.ADX })));

        // ---------------- 隐藏指标最后值和价格线 ----------------
        [ema20Series, ema50Series, wrSeries, macdMainSeries, macdSignalSeries, macdHistSeries,
            kdjKSeries, kdjDSeries, kdjJSeries, plusDISeries, minusDISeries].forEach(series => {
                series.applyOptions({ lastValueVisible: false, priceLineVisible: false });
            });

        // ---------------- 存储所有图表实例 ----------------
        charts[`${token}-${minute}`] = {
            mainChart, mainSeries, ema20Series, ema50Series,
            wrChart, wrSeries,
            macdChart, macdMainSeries, macdSignalSeries, macdHistSeries,
            kdjChart, kdjKSeries, kdjDSeries, kdjJSeries,
            dmiChart, plusDISeries, minusDISeries
            // 删除originalHeights属性
        };


    });
    data_init = true;
});

// 监听 kline_refresh 事件
socket.on('kline_refresh', function (refreshData) {
    
    if ( !data_init) return;
    // Object.keys(refreshData).forEach(key => {
    //     if (typeof refreshData[key] === 'string') {
    //         try { refreshData[key] = JSON.parse(refreshData[key]); }
    //         catch (error) { console.error(`解析 refreshData[${key}] 时出错:`, error); }
    //     }
    // });
    
    const token = $('#token-group .active').data('value');
    const minutes = $('#interval-group .control-btn.active').map(function () {
        return $(this).data('value');
    }).get();

    minutes.forEach(minute => {
        const key = `${token}-${minute}`;
        const chartGroup = charts[key];
        if (!chartGroup) return;
       
        const newData = JSON.parse(refreshData[token][minute]); // 取最新一条数据
        // console.log(newData);
        const time = newData.ts / 1000;

        // 更新主图K线及EMA
        chartGroup.mainSeries.update({ time, open: newData.open, close: newData.close, high: newData.high, low: newData.low });
        chartGroup.ema20Series.update({ time, value: newData.EMA20 });
        chartGroup.ema50Series.update({ time, value: newData.EMA50 });

        // 更新WR指标
        chartGroup.wrSeries.update({ time, value: newData.WR });

        // 更新MACD指标
        chartGroup.macdMainSeries.update({ time, value: newData.MACD });
        chartGroup.macdSignalSeries.update({ time, value: newData.SIGNAL });
        chartGroup.macdHistSeries.update({ time, value: newData.HISTOGRAM });

        // 更新KDJ指标
        chartGroup.kdjKSeries.update({ time, value: newData.KDJ_K });
        chartGroup.kdjDSeries.update({ time, value: newData.KDJ_D });
        chartGroup.kdjJSeries.update({ time, value: newData.KDJ_J });

        // 更新DMI指标
        chartGroup.plusDISeries.update({ time, value: newData.PLUS_DI });
        chartGroup.minusDISeries.update({ time, value: newData.MINUS_DI });
        // // chartGroup.adxSeries.update({ time, value: newData.ADX });
    });
});

// 监听 ws_err 事件
socket.on('ws_err', function (errorMessage) {
    console.error('WebSocket 连接错误:', errorMessage);
    alert(`WebSocket 连接错误: ${errorMessage}`);
});

$(document).ready(function () {
    // 币种选择只能选一个
    $('#token-group').on('click', '.control-btn', function () {
        $('#token-group .control-btn').removeClass('active');
        $(this).addClass('active');
    });

    // 周期选择最多选4个
    $('#interval-group').on('click', '.control-btn', function () {
        const activeCount = $('#interval-group .control-btn.active').length;
        if ($(this).hasClass('active')) {
            $(this).removeClass('active');
        } else if (activeCount < 4) {
            $(this).addClass('active');
        }
    });

    $('#confirm-btn').on('click', function () {
        data_init = false;
        const token = $('#token-group .active').data('value');
        const minutes = $('#interval-group .control-btn.active').map(function () {
            return $(this).data('value');
        }).get();

        $('#additional-charts-container').empty().css({ display: 'flex', flexWrap: 'wrap', gap: '20px' });

        minutes.forEach(minute => {
            const containerId = `chart-${token}-${minute}`;
            $('#additional-charts-container').append(`
                <div id="${containerId}" style="flex: 1; min-width: 300px; display: flex; flex-direction: column; margin-bottom: 20px;">
                    
                <div id="${containerId}-main" style="height: 240px;"></div>
                    <div id="${containerId}-wr" style="height: 150px;"></div>
                    <div id="${containerId}-kdj" style="height: 150px;"></div>
                    <div id="${containerId}-macd" style="height: 150px;"></div>
                    <div id="${containerId}-dmi" style="height: 150px;"></div>
                </div>
            `);
            // $('#additional-charts-container').append(`
            //     <div id="${containerId}"></div>
            // `);
        });

        // 统一向socketio发送订阅请求
        socket.emit('subscribe_kline', {
            token: token,
            minutes: minutes,
        });
    });
});