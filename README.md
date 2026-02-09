# qrc_classification_demo

Google Colab で実行する量子リザバー計算 (QRC) の時系列2値分類デモです。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dainfinity/qrc_classification_demo/blob/main/notebooks/qrc_timeseries_classification_colab.ipynb)

## ローカル開発環境 (uv)

```bash
uv sync
uv run python scripts/make_binary_timeseries_dataset.py
```

## 主要ファイル

- `notebooks/qrc_timeseries_classification_colab.ipynb`: Colab向けデモNotebook
- `src/spin_viz.py`: 多体系スピンのインタラクティブ可視化ウィジェット
- `data/train_X.csv`, `data/train_y.csv`, `data/test_X.csv`, `data/test_y.csv`: 固定データセット
- `scripts/make_binary_timeseries_dataset.py`: データセット再生成スクリプト

データは「低周波→高周波／高周波→低周波」の時系列で、低周波/高周波の範囲は少し重なるように設定しています。切り替え時刻は中央付近に固定され、ノイズ量を大きめ（`noise_std=0.4`）にしています。
