"""
이동평균 크로스(MA Cross) 전략 로직.

단기 이동평균이 장기 이동평균을 위로 뚫으면(골든크로스) 매수 포지션,
아래로 뚫으면(데드크로스) 매도(청산/숏) 포지션을 잡는다.
"""

import pandas as pd


def add_signals(df: pd.DataFrame, short_window: int = 20, long_window: int = 60) -> pd.DataFrame:
    """
    가격 데이터프레임에 이동평균과 포지션 시그널을 추가한다.

    Parameters
    ----------
    df : DataFrame
        'date', 'close' 컬럼을 가진 가격 데이터.
    short_window : int
        단기 이동평균 기간 (기본 20일).
    long_window : int
        장기 이동평균 기간 (기본 60일).

    Returns
    -------
    DataFrame
        ma_short, ma_long, position 컬럼이 추가된 데이터프레임.
        position: 1 = 매수 보유, 0 = 미보유.
    """
    out = df.copy()
    out["ma_short"] = out["close"].rolling(short_window).mean()
    out["ma_long"] = out["close"].rolling(long_window).mean()

    # 단기선이 장기선 위에 있으면 매수(1), 아니면 미보유(0)
    out["position"] = 0
    valid = out["ma_long"].notna()
    out.loc[valid & (out["ma_short"] > out["ma_long"]), "position"] = 1

    # 오늘의 시그널로 '다음 날'부터 진입 (미래참조 방지)
    out["position"] = out["position"].shift(1).fillna(0).astype(int)

    return out
