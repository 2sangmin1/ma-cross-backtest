"""
실행 진입점.

python main.py 한 줄이면 백테스트가 돌고,
성과지표가 출력되며 results/ 폴더에 그래프가 저장된다.

파라미터 변경:
    python main.py --short 10 --long 40
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from strategy import add_signals
from backtest import run_backtest, compute_metrics


def main(short_window: int, long_window: int, data_path: str):
    # 1) 데이터 로드
    df = pd.read_csv(data_path, parse_dates=["date"])

    # 2) 시그널 생성 + 백테스트
    df = add_signals(df, short_window, long_window)
    df = run_backtest(df)

    # 3) 성과지표 출력
    m = compute_metrics(df)
    print("=" * 40)
    print(f"  이동평균 크로스 전략  ({short_window} / {long_window})")
    print("=" * 40)
    print(f"  누적 수익률 : {m['total_return']:+.2%}")
    print(f"  최대 낙폭(MDD) : {m['mdd']:.2%}")
    print(f"  샤프지수 : {m['sharpe']:.2f}")
    print(f"  승률 : {m['win_rate']:.2%}")
    print(f"  매매 횟수 : {m['num_trades']}")
    print("=" * 40)

    # 4) 그래프 저장
    _plot(df, short_window, long_window)
    print("\n그래프 저장 완료 -> results/")


def _plot(df: pd.DataFrame, short_window: int, long_window: int):
    # (1) 가격 + 이동평균 + 매매시점
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["close"], label="Price", color="#333", linewidth=1)
    ax.plot(df["date"], df["ma_short"], label=f"MA{short_window}", color="#2E86DE", linewidth=1)
    ax.plot(df["date"], df["ma_long"], label=f"MA{long_window}", color="#E67E22", linewidth=1)

    # 진입(0->1)과 청산(1->0) 지점 표시
    entries = df[(df["position"] == 1) & (df["position"].shift(1) == 0)]
    exits = df[(df["position"] == 0) & (df["position"].shift(1) == 1)]
    ax.scatter(entries["date"], entries["close"], marker="^", color="green", s=90, label="Buy", zorder=5)
    ax.scatter(exits["date"], exits["close"], marker="v", color="red", s=90, label="Sell", zorder=5)

    ax.set_title(f"MA Cross Strategy — Price & Signals ({short_window}/{long_window})")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/signals.png", dpi=130)
    plt.close(fig)

    # (2) 전략 vs 단순보유 자산곡선
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["strategy_cum"], label="Strategy", color="#2E86DE", linewidth=1.5)
    ax.plot(df["date"], df["market_cum"], label="Buy & Hold", color="#999", linewidth=1.5, linestyle="--")
    ax.set_title("Equity Curve — Strategy vs Buy & Hold")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/equity_curve.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MA Cross 백테스트")
    parser.add_argument("--short", type=int, default=20, help="단기 이동평균 기간")
    parser.add_argument("--long", type=int, default=60, help="장기 이동평균 기간")
    parser.add_argument("--data", type=str, default="data/sample_prices.csv", help="시세 CSV 경로")
    args = parser.parse_args()

    main(args.short, args.long, args.data)
