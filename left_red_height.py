#!/usr/bin/env python3
"""画像左側の赤い縦線の高さをピクセル単位で出力する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

def detect_red_pixels(image: np.ndarray, red_min: int, diff_min: int) -> np.ndarray:
  """画像から赤線に見える画素を検出し、真偽値のマスクとして返す。"""
  blue, green, red = cv2.split(image)
  return (red >= red_min) & ((red - green) >= diff_min) & ((red - blue) >= diff_min)

def find_left_red_column(red_mask: np.ndarray, min_column_pixels: int) -> int:
  """赤画素が十分に多い列のうち、最も左にある列番号を返す。"""
  red_counts = np.count_nonzero(red_mask, axis=0)
  columns = np.where(red_counts >= min_column_pixels)[0]
  if len(columns) == 0:
    raise ValueError("左側の赤い縦線を検出できませんでした。")
  return int(columns[0])

def measure_column_height(red_mask: np.ndarray, column: int) -> int:
  """指定した列にある赤画素の上端から下端までの高さを返す。"""
  rows = np.where(red_mask[:, column])[0]
  if len(rows) == 0:
    raise ValueError("指定した列に赤画素がありません。")
  return int(rows[-1] - rows[0] + 1)

def measure_left_red_height(
  input_path: Path,
  *,
  red_min: int,
  diff_min: int,
  min_column_pixels: int,
) -> int:
  """入力画像を読み込み、左側の赤い縦線の高さをピクセル単位で求める。"""
  image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
  if image is None:
    raise FileNotFoundError(f"画像を読み込めませんでした: {input_path}")

  red_mask = detect_red_pixels(image, red_min, diff_min)
  left_column = find_left_red_column(red_mask, min_column_pixels)
  return measure_column_height(red_mask, left_column)

def parse_args() -> argparse.Namespace:
  """コマンドライン引数を解析し、入力画像や赤線判定条件を返す。"""
  parser = argparse.ArgumentParser(
    description="画像左側の赤い縦線の高さをピクセル単位で出力します。",
    add_help=False,
  )
  parser._positionals.title = "位置引数"
  parser._optionals.title = "オプション"
  parser.add_argument(
    "-h",
    "--help",
    action="help",
    help="このヘルプメッセージを表示して終了します",
  )
  parser.add_argument(
    "input",
    nargs="?",
    default=Path("output.jpg"),
    type=Path,
    help="入力画像のパス（初期値: output.jpg）",
  )
  parser.add_argument(
    "--red-min",
    type=int,
    default=120,
    help="赤として扱うR値の下限。0-255で指定します（初期値: 120）",
  )
  parser.add_argument(
    "--diff-min",
    type=int,
    default=40,
    help="R値がG値・B値よりどれだけ大きい画素を赤とみなすか（初期値: 40）",
  )
  parser.add_argument(
    "--min-column-pixels",
    type=int,
    default=100,
    help="縦線候補に必要な、同じ列内の赤画素数（初期値: 100）",
  )
  return parser.parse_args()

def main() -> None:
  """左側の赤い縦線の高さを計算して標準出力に表示する。"""
  args = parse_args()
  height = measure_left_red_height(
    args.input,
    red_min=args.red_min,
    diff_min=args.diff_min,
    min_column_pixels=args.min_column_pixels,
  )
  print(height)

if __name__ == "__main__":
  main()
