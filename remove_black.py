#!/usr/bin/env python3
"""色付きの線を残しながら、画像から黒またはグレーの注釈を取り除く。"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

def should_remove(r: int, g: int, b: int, value_limit: int, saturation_limit: int) -> bool:
  """RGB値から、削除対象の黒またはグレーの画素かどうかを判定する。"""
  value = max(r, g, b)
  chroma = value - min(r, g, b)
  saturation = 0 if value == 0 else round(chroma * 255 / value)
  return value <= value_limit and saturation <= saturation_limit

def remove_black(
  input_path: Path,
  output_path: Path,
  *,
  value_limit: int,
  saturation_limit: int,
  transparent: bool,
) -> None:
  """画像内の黒またはグレーの画素を白、または透明に置き換えて保存する。"""
  image = Image.open(input_path).convert("RGBA")
  pixels = image.load()

  width, height = image.size
  replacement = (255, 255, 255, 0 if transparent else 255)

  for y in range(height):
    for x in range(width):
      r, g, b, a = pixels[x, y]
      if a and should_remove(r, g, b, value_limit, saturation_limit):
        pixels[x, y] = replacement

  if transparent:
    image.save(output_path)
  else:
    image.convert("RGB").save(output_path, quality=95)

def parse_args() -> argparse.Namespace:
  """コマンドライン引数を解析し、入力画像や出力先などの設定を返す。"""
  parser = argparse.ArgumentParser(
    description=(
      "画像から黒またはグレーの部分を取り除きます。"
      "赤い図形を残しながら寸法文字を消す用途に使えます。"
    ),
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
  parser.add_argument("input", type=Path, help="入力画像のパス")
  parser.add_argument("output", type=Path, help="出力画像のパス")
  parser.add_argument(
    "--value-limit",
    type=int,
    default=220,
    help="削除する明るさの上限。0-255で指定します（初期値: 220）",
  )
  parser.add_argument(
    "--saturation-limit",
    type=int,
    default=70,
    help="削除する彩度の上限。0-255で指定します（初期値: 70）",
  )
  parser.add_argument(
    "--transparent",
    action="store_true",
    help="削除した画素を白ではなく透明にします",
  )
  return parser.parse_args()

def main() -> None:
  """コマンドラインから受け取った設定で画像の黒い部分を取り除く。"""
  args = parse_args()
  remove_black(
    args.input,
    args.output,
    value_limit=args.value_limit,
    saturation_limit=args.saturation_limit,
    transparent=args.transparent,
  )

if __name__ == "__main__":
  main()
