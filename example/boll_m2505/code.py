import pathlib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import talib


def caculate_boll_bands(df: pd.DataFrame, window: int = 20, dev: float = 2.0):
    df["upper"], df["ma"], df["lower"] = talib.BBANDS(
        df["price"].to_numpy(), timeperiod=window, nbdevup=dev, nbdevdn=dev
    )
    return df


def bollinger_strategy(df: pd.DataFrame, window: int, dev: float) -> pd.DataFrame:
    df = df.copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = caculate_boll_bands(df=df, window=window, dev=dev)
    df["signal"] = 0
    df["pos"] = 0
    df["returns"] = 0.0
    df["equity"] = 1.0
    df["signal_type"] = ""  # 新增标记信号类型的列

    for i in range(1, len(df)):
        signal = 0  # 初始化信号
        signal_type = ""  # 初始化信号类型
        prev_pos = df["pos"].iloc[i - 1]

        # 策略逻辑：价格上穿上轨做空，价格下穿下轨做多
        # 价格回落中轨平多仓，价格回升中轨平空仓

        # 1. 开仓逻辑：仅在无持仓时开仓
        if prev_pos == 0:
            # 检查价格是否突破上轨（当前价格高于上轨）
            if df["price"].iloc[i] > df["upper"].iloc[i - 1]:
                signal = -1  # 做空
                signal_type = "开空"
            # 检查价格是否突破下轨（当前价格低于下轨）
            elif df["price"].iloc[i] < df["lower"].iloc[i - 1]:
                signal = 1  # 做多
                signal_type = "开多"

        # 2. 平仓逻辑
        # 多仓平仓：价格回落至中轨（当前价格低于等于中轨）
        elif prev_pos > 0 and df["price"].iloc[i] <= df["ma"].iloc[i - 1]:
            signal = -prev_pos  # 平仓（信号大小等于持仓量的相反数）
            signal_type = "平多"
        # 空仓平仓：价格回升至中轨（当前价格高于等于中轨）
        elif prev_pos < 0 and df["price"].iloc[i] >= df["ma"].iloc[i - 1]:
            signal = -prev_pos  # 平仓（信号大小等于持仓量的相反数）
            signal_type = "平空"

        # 更新信号和信号类型
        df.loc[df.index[i], "signal"] = signal
        df.loc[df.index[i], "signal_type"] = signal_type

        # 更新仓位
        df.loc[df.index[i], "pos"] = prev_pos + signal

        prev_price = float(df.iloc[i - 1]["price"])
        curr_price = float(df.iloc[i]["price"])
        prev_pos_float = float(df.iloc[i - 1]["pos"])

        ret = prev_pos_float * (curr_price / prev_price - 1.0)
        df.loc[df.index[i], "returns"] = ret
        df.loc[df.index[i], "equity"] = float(df.iloc[i - 1]["equity"]) * (1.0 + ret)

    return df


def calculate_returns(df: pd.DataFrame):
    total_return = df["equity"].iloc[-1] - 1
    years = (df.index[-1] - df.index[0]).days / 365.25
    if years <= 0 or total_return <= -1:  # 增加对total_return的检查
        return None, None
    try:
        annual_return = (1 + total_return) ** (1 / years) - 1
    except ValueError:
        return None, None

    daily_volatility = df["returns"].std() * np.sqrt(252)
    # 避免除以零的错误
    if daily_volatility == 0:
        sharpe_ratio = 0
    else:
        sharpe_ratio = annual_return / daily_volatility

    df["cummax"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["cummax"] - 1
    max_drawdown = df["drawdown"].min()

    # 避免除以零的错误
    non_zero_returns = len(df[df["returns"] != 0])
    if non_zero_returns == 0:
        win_rate = 0
    else:
        win_rate = len(df[df["returns"] > 0]) / non_zero_returns

    trades = len(df[df["signal"] != 0])

    metrics = ["年化收益率", "夏普比率", "最大回撤", "胜率", "总收益率", "交易次数"]
    values = [
        f"{annual_return:.2%}",
        f"{sharpe_ratio:.2f}",
        f"{max_drawdown:.2%}",
        f"{win_rate:.2%}",
        f"{total_return:.2%}",
        f"{trades}",
    ]

    return metrics, values


def optimize_parameters(df: pd.DataFrame, window_range: range, dev_range: np.ndarray):
    results = []
    best_params = None
    best_annual_return = -np.inf
    for window in window_range:
        for dev in dev_range:
            df_ret = bollinger_strategy(df, window, dev)
            metrics, values = calculate_returns(df_ret)
            if values is None:
                continue
            annual_return = float(values[0].strip("%")) / 100
            results.append((window, dev, annual_return))
            if annual_return > best_annual_return:
                best_annual_return = annual_return
                best_params = (window, dev)
    return results, best_params


def plot(df, results):
    """可视化策略结果"""
    metrics, values = calculate_returns(df)
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.25, 0.15, 0.2, 0.5],
        specs=[
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "table"}],
            [{"type": "heatmap"}],
            [{"type": "surface"}],
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["price"],
            name="价格",
            line=dict(color="blue", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["ma"], name="中轨", line=dict(color="yellow")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["upper"], name="上轨", line=dict(color="red", dash="dot")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["lower"], name="下轨", line=dict(color="red", dash="dot")
        ),
        row=1,
        col=1,
    )

    # 根据信号类型标记买卖点
    long_entry = df[df["signal_type"] == "开多"]  # 开多仓
    long_exit = df[df["signal_type"] == "平多"]  # 平多仓
    short_entry = df[df["signal_type"] == "开空"]  # 开空仓
    short_exit = df[df["signal_type"] == "平空"]  # 平空仓

    # 开多仓标记（买入）
    fig.add_trace(
        go.Scatter(
            x=long_entry.index,
            y=long_entry["price"],
            mode="markers",
            name="开多",
            marker=dict(color="red", size=10, symbol="triangle-up"),
        ),
        row=1,
        col=1,
    )

    # 平多仓标记
    fig.add_trace(
        go.Scatter(
            x=long_exit.index,
            y=long_exit["price"],
            mode="markers",
            name="平多",
            marker=dict(color="orange", size=10, symbol="triangle-down"),
        ),
        row=1,
        col=1,
    )

    # 开空仓标记（卖出）
    fig.add_trace(
        go.Scatter(
            x=short_entry.index,
            y=short_entry["price"],
            mode="markers",
            name="开空",
            marker=dict(color="green", size=10, symbol="triangle-down"),
        ),
        row=1,
        col=1,
    )

    # 平空仓标记
    fig.add_trace(
        go.Scatter(
            x=short_exit.index,
            y=short_exit["price"],
            mode="markers",
            name="平空",
            marker=dict(color="blue", size=10, symbol="triangle-up"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df["equity"], name="净值", line=dict(color="purple")),
        row=2,
        col=1,
    )

    headers = ["指标", "数值"]
    table_header = dict(
        values=headers,
        align="center",
        font=dict(size=12, color="white"),
        fill_color="royalblue",
    )

    table_cells = dict(values=[metrics, values], align="center")

    fig.add_trace(
        go.Table(header=table_header, cells=table_cells),
        row=3,
        col=1,
    )

    df_results = pd.DataFrame(results, columns=["Window", "Dev", "Annual Return"])
    heatmap = px.density_heatmap(
        df_results,
        x="Window",
        y="Dev",
        z="Annual Return",
        nbinsx=30,  # 增加x轴bin数量
        nbinsy=30,  # 增加y轴bin数量
        # color_continuous_scale="Viridis",
    )
    for trace in heatmap.data:
        fig.add_trace(trace, row=4, col=1)

    # 创建3D曲面图数据
    df_results = pd.DataFrame(results, columns=["Window", "Dev", "Annual Return"])
    pivot_df = df_results.pivot(index="Dev", columns="Window", values="Annual Return")

    fig.add_trace(
        go.Surface(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale="Viridis",
            name="3D视图",
            showlegend=False,
            showscale=False,  # 隐藏颜色条
        ),
        row=5,
        col=1,
    )

    fig.update_layout(
        title="布林带策略回测",
        height=1500,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(r=150),  # 增加右侧边距
        coloraxis_colorbar=dict(
            title="年化收益率",
            x=1.1,  # 将颜色条向右移动
            len=0.8,  # 调整颜色条长度
        ),
    )

    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="净值", row=2, col=1)

    fig.show()


def plot_best_strategy(df, window, dev, results):
    df_ret = bollinger_strategy(df, window, dev)
    plot(df_ret, results)


def main():
    df = pd.read_csv(pathlib.Path("example/布林带-豆粕M2505/M2505.csv"))
    df = df[["date", "close"]].rename(columns={"date": "datetime", "close": "price"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    # # 先使用固定参数测试策略
    # print("使用固定参数测试策略...")
    # plot_best_strategy(df, 20, 1.3, None)

    # 如果需要优化参数，可以取消下面的注释
    results, best_params = optimize_parameters(
        df, window_range=range(5, 35), dev_range=np.arange(0.5, 4, 0.1)
    )
    if best_params:
        print(f"最佳参数: 窗口大小={best_params[0]}, 标准差倍数={best_params[1]}")
        plot_best_strategy(df, best_params[0], best_params[1], results)


if __name__ == "__main__":
    main()
