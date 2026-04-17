#!/usr/bin/env python3
"""赤線で囲まれた部分のピクセル数を出力する。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

def detect_red_pixels(image: np.ndarray, red_min: int, diff_min: int) -> np.ndarray:
  """画像から赤線に見える画素を検出し、真偽値のマスクとして返す。"""
  blue, green, red = cv2.split(image)
  return (red >= red_min) & ((red - green) >= diff_min) & ((red - blue) >= diff_min)

def close_red_boundary(red_mask: np.ndarray, thickness: int) -> np.ndarray:
  """赤線マスクを太らせて、JPEGのにじみや細い隙間をふさぐ。"""
  kernel = np.ones((thickness, thickness), dtype=np.uint8)
  mask = red_mask.astype(np.uint8) * 255
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.dilate(mask, kernel)
  return mask > 0

def find_outside_pixels(boundary_mask: np.ndarray) -> np.ndarray:
  """赤線の外側から到達できる画素を flood fill で求める。"""
  height, width = boundary_mask.shape
  fill_image = np.where(boundary_mask, 255, 0).astype(np.uint8)
  flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)

  for x in range(width):
    if fill_image[0, x] == 0:
      cv2.floodFill(fill_image, flood_mask, (x, 0), 128)
    if fill_image[height - 1, x] == 0:
      cv2.floodFill(fill_image, flood_mask, (x, height - 1), 128)

  for y in range(height):
    if fill_image[y, 0] == 0:
      cv2.floodFill(fill_image, flood_mask, (0, y), 128)
    if fill_image[y, width - 1] == 0:
      cv2.floodFill(fill_image, flood_mask, (width - 1, y), 128)

  return fill_image == 128

def count_enclosed_pixels(
  input_path: Path,
  *,
  red_min: int,
  diff_min: int,
  boundary_thickness: int,
  include_line: bool,
) -> int:
  """入力画像を読み込み、赤線で囲まれた内側のピクセル数を数える。"""
  image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
  if image is None:
    raise FileNotFoundError(f"画像を読み込めませんでした: {input_path}")

  red_mask = detect_red_pixels(image, red_min, diff_min)
  boundary_mask = close_red_boundary(red_mask, boundary_thickness)
  outside_mask = find_outside_pixels(boundary_mask)
  enclosed_mask = ~outside_mask

  if not include_line:
    enclosed_mask &= ~boundary_mask

  return int(np.count_nonzero(enclosed_mask))

def parse_args() -> argparse.Namespace:
  """コマンドライン引数を解析し、入力画像や判定条件を返す。"""
  parser = argparse.ArgumentParser(
    description="赤線で囲まれた部分のピクセル数を出力します。",
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
    "--boundary-thickness",
    type=int,
    default=3,
    help="赤線の隙間をふさぐために使う太さ（初期値: 3）",
  )
  parser.add_argument(
    "--include-line",
    action="store_true",
    help="赤線そのものの画素もピクセル数に含めます",
  )
  return parser.parse_args()

def main() -> None:
  """赤線で囲まれた部分のピクセル数を計算して標準出力に表示する。"""
  args = parse_args()
  count = count_enclosed_pixels(
    args.input,
    red_min=args.red_min,
    diff_min=args.diff_min,
    boundary_thickness=args.boundary_thickness,
    include_line=args.include_line,
  )
  print(count)

if __name__ == "__main__":
  main()
