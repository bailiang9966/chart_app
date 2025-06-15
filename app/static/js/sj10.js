var data_init = false;
var is_syncing = false;
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
    crossHair: { 
        mode: 2, // 显示十字光标
        vertLine: {
            visible: true,
            style: 0,
            width: 1,
            color: 'rgba(197, 203, 206, 0.7)',
            labelVisible: true
        },
        horzLine: {
            visible: true,
            style: 0,
            width: 1,
            color: 'rgba(197, 203, 206, 0.7)',
            labelVisible: true
        }
    }
};

const socket = io();
const charts = {};

// 监听 kline_init 事件
socket.on('kline_init', function (data) {
    const token = $('#token-group .active').data('value');
    const minutes = $('#interval-group .control-btn.active').map(function () {
        return $(this).data('value');
    }).get();

    minutes.forEach(minute => {
        const containerId = `chart-${token}-${minute}`;
        
        // 创建单个图表实例
        const mainChart = LightweightCharts.createChart(
            document.getElementById(containerId),
            {
                width: document.getElementById(containerId).offsetWidth,
                height: 750,
                layout: {
                    backgroundColor: '#ffffff',
                    textColor: '#000000',
                },
                crossHair: commonOptions.crossHair,
                timeScale: commonOptions.timeScale,
                localization: commonOptions.localization
            }
        );
        
        // 主图 pane (v5.0.7正确用法)
        const mainPane = mainChart.addPane({
            height: 260,
            title: { text: '价格' }
        });
        const mainSeries = mainPane.addCandlestickSeries({
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
        });
        const ema20Series = mainPane.addLineSeries({ color: '#FF9800', lineWidth: 1 });
        const ema50Series = mainPane.addLineSeries({ color: '#2196F3', lineWidth: 1 });
        
        // WR 副图 pane
        const wrPane = mainChart.addPane({
            height: 120,
            title: { text: 'WR' }
        });
        const wrSeries = wrPane.addLineSeries({ color: '#9C27B0' });
        wrSeries.createPriceLine({ price: -50, color: '#3179F5', lineStyle: 2 });
        
        // MACD 副图 pane
        const macdPane = mainChart.addPane({
            height: 120,
            title: { text: 'MACD' }
        });
        const macdMainSeries = macdPane.addLineSeries({ color: '#FF5722', lineWidth: 2 });
        const macdSignalSeries = macdPane.addLineSeries({ color: '#2196F3', lineWidth: 1 });
        const macdHistSeries = macdPane.addHistogramSeries({ lineWidth: 1 });
        
        // KDJ 副图 pane
        const kdjPane = mainChart.addPane({
            height: 120,
            title: { text: 'KDJ' }
        });
        const kdjKSeries = kdjPane.addLineSeries({ color: '#FF9800', lineWidth: 2 });
        const kdjDSeries = kdjPane.addLineSeries({ color: '#795548', lineWidth: 1 });
        const kdjJSeries = kdjPane.addLineSeries({ color: '#9C27B0', lineWidth: 1 });
        
        // DMI 副图 pane
        const dmiPane = mainChart.addPane({
            height: 120,
            title: { text: 'DMI' }
        });
        const plusDISeries = dmiPane.addLineSeries({ color: '#E91E63', lineWidth: 2 });
        const minusDISeries = dmiPane.addLineSeries({ color: '#009688', lineWidth: 2 });
        
        // 初始化数据绑定
        const kd = JSON.parse(data[token][minute]);
        console.log('Parsed data for', token, minute, ':', kd);
        const kData = kd.map(item => ({ time: item.ts / 1000, open: item.open, close: item.close, high: item.high, low: item.low }));
        
        mainSeries.setData(kData);
        ema20Series.setData(kd.map(item => ({ time: item.ts / 1000, value: item.EMA20 })));
        ema50Series.setData(kd.map(item => ({ time: item.ts / 1000, value: item.EMA50 })));
        wrSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.WR })));
        macdMainSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.MACD })));
        macdSignalSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.SIGNAL })));
        macdHistSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.HISTOGRAM, color: item.HISTOGRAM > 0 ? '#26a69a' : "red" })));
        kdjKSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.KDJ_K })));
        kdjDSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.KDJ_D })));
        kdjJSeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.KDJ_J })));
        plusDISeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.PLUS_DI })));
        minusDISeries.setData(kd.map(item => ({ time: item.ts / 1000, value: item.MINUS_DI })));
        
        // ---------------- 隐藏指标最后值和价格线 ----------------
        [ema20Series, ema50Series, wrSeries, macdMainSeries, macdSignalSeries, macdHistSeries,
            kdjKSeries, kdjDSeries, kdjJSeries, plusDISeries, minusDISeries].forEach(series => {
                series.applyOptions({ lastValueVisible: false, priceLineVisible: false });
            });
        
        // ---------------- 存储所有图表实例 ----------------
        charts[`${token}-${minute}`] = {
            mainChart, mainSeries, ema20Series, ema50Series,
            wrSeries, macdMainSeries, macdSignalSeries, macdHistSeries,
            kdjKSeries, kdjDSeries, kdjJSeries, plusDISeries, minusDISeries
        };

    });
    data_init = true;
});

// 监听 kline_refresh 事件
socket.on('kline_refresh', function (refreshData) {
    
    if (!data_init) return;
    
    const token = $('#token-group .active').data('value');
    const minutes = $('#interval-group .control-btn.active').map(function () {
        return $(this).data('value');
    }).get();

    minutes.forEach(minute => {
        const key = `${token}-${minute}`;
        const chartGroup = charts[key];
        if (!chartGroup) return;
       
        const newData = JSON.parse(refreshData[token][minute]);
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
                <div id="${containerId}" style="flex: 1; min-width: 800px; height: 750px; border: 1px solid #e0e0e0; border-radius: 4px; padding: 10px;"></div>
            `);
        });

        // 统一向socketio发送订阅请求
        socket.emit('subscribe_kline', {
            token: token,
            minutes: minutes,
        });
    });
});