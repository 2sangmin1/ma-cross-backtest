"""
백테스트 엔진.

전략 시그널을 받아 일별 수익률을 계산하고,
누적 수익률·MDD·샤프지수·승률 등 핵심 성과지표를 산출한다.
"""

import numpy as np
import pandas as pd


def run_backtest(df: pd.DataFrame, fee: float = 0.00005) -> pd.DataFrame:
    """
    포지션 시그널을 바탕으로 전략 수익률을 계산한다.

    Parameters
    ----------
    df : DataFrame
        add_signals()를 거친 데이터프레임 (position 컬럼 필요).
    fee : float
        매매 1회당 거래비용 (기본 0.005%).

    Returns
    -------
    DataFrame
        일별/누적 수익률이 추가된 데이터프레임.
    """
    out = df.copy()

    # 시장 수익률 (그냥 보유했을 때)
    out["market_ret"] = out["close"].pct_change().fillna(0)

    # 전략 수익률 = 포지션 * 시장 수익률
    out["strategy_ret"] = out["position"] * out["market_ret"]

    # 포지션이 바뀌는 날 거래비용 차감
    trades = out["position"].diff().abs().fillna(0)
    out["strategy_ret"] -= trades * fee

    # 누적 수익 곡선 (1에서 시작)
    out["market_cum"] = (1 + out["market_ret"]).cumprod()
    out["strategy_cum"] = (1 + out["strategy_ret"]).cumprod()

    return out


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    전략의 핵심 성과지표를 계산한다.

    Returns
    -------
    dict
        total_return, mdd, sharpe, win_rate, num_trades.
    """
    ret = df["strategy_ret"]
    cum = df["strategy_cum"]

    total_return = cum.iloc[-1] - 1

    # 최대 낙폭 (MDD): 고점 대비 최대 하락률
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    mdd = drawdown.min()

    # 샤프지수 (연율화, 무위험수익률 0 가정, 252 영업일)
    if ret.std() != 0:
        sharpe = np.sqrt(252) * ret.mean() / ret.std()
    else:
        sharpe = np.nan

    # 승률: 포지션을 보유한 날 중 수익난 날의 비율
    held = df[df["position"] == 1]
    win_rate = (held["strategy_ret"] > 0).mean() if len(held) else np.nan

    # 매매 횟수
    num_trades = int(df["position"].diff().abs().fillna(0).sum())

    return {
        "total_return": total_return,
        "mdd": mdd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "num_trades": num_trades,
    }
