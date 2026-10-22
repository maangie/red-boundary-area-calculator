import pathlib

import cv2
import numpy as np
from PIL import Image

from count_enclosed_pixels import count_enclosed_pixels
from left_red_height import measure_left_red_height
from remove_black import remove_black, should_remove

HERE = pathlib.Path(__file__).parent
RED = (0, 0, 255)  # OpenCV は BGR 順
def write_png(path: pathlib.Path, image: np.ndarray) -> pathlib.Path:
  """BGR 画像を PNG で保存し、そのパスを返す。

  引数:
    path: 保存先のパス
    image: BGR 順の画像配列
  戻り値:
    保存したファイルのパス
  """
  cv2.imwrite(str(path), image)
  return path
def white_image(height: int, width: int) -> np.ndarray:
  """白一色の BGR 画像を作る。

  引数:
    height: 画像の高さ（px）
    width: 画像の幅（px）
  戻り値:
    白一色の BGR 画像配列
  """
  return np.full((height, width, 3), 255, dtype=np.uint8)
def test_should_remove_black():
  """黒い画素は削除対象になる。"""
  assert should_remove(0, 0, 0, value_limit=220, saturation_limit=70)
def test_should_remove_gray():
  """グレーの画素は削除対象になる。"""
  assert should_remove(128, 128, 128, value_limit=220, saturation_limit=70)
def test_should_not_remove_white():
  """白い画素は明るさの上限を超えるので残る。"""
  assert not should_remove(255, 255, 255, value_limit=220, saturation_limit=70)
def test_should_not_remove_red():
  """赤い画素は彩度が高いので残る。"""
  assert not should_remove(200, 0, 0, value_limit=220, saturation_limit=70)
def test_remove_black_replaces_black_with_white(tmp_path):
  """黒い画素は白に置き換わり、赤い画素はそのまま残る。"""
  src = Image.new("RGB", (2, 1))
  src.putpixel((0, 0), (0, 0, 0))
  src.putpixel((1, 0), (200, 0, 0))
  src.save(tmp_path / "in.png")

  remove_black(
    tmp_path / "in.png",
    tmp_path / "out.png",
    value_limit=220,
    saturation_limit=70,
    transparent=False,
  )

  out = Image.open(tmp_path / "out.png").convert("RGB")
  assert out.getpixel((0, 0)) == (255, 255, 255)
  assert out.getpixel((1, 0)) == (200, 0, 0)
def test_remove_black_transparent(tmp_path):
  """transparent を指定すると、黒い画素は透明になる。"""
  Image.new("RGB", (1, 1), (0, 0, 0)).save(tmp_path / "in.png")

  remove_black(
    tmp_path / "in.png",
    tmp_path / "out.png",
    value_limit=220,
    saturation_limit=70,
    transparent=True,
  )

  assert Image.open(tmp_path / "out.png").getpixel((0, 0))[3] == 0
def test_count_enclosed_pixels_square(tmp_path):
  """赤い正方形の枠の内側のピクセル数を数える。

  1px の枠は太さ 3 の処理で内外に 1px ずつ太るので、
  枠（10〜29）の内側で数えられるのは 12〜27 の 16×16 px になる。
  """
  image = white_image(40, 40)
  cv2.rectangle(image, (10, 10), (29, 29), RED, 1)
  path = write_png(tmp_path / "square.png", image)

  count = count_enclosed_pixels(
    path,
    red_min=120,
    diff_min=40,
    boundary_thickness=3,
    include_line=False,
  )
  assert count == 16 * 16
def test_count_enclosed_pixels_include_line(tmp_path):
  """include_line を指定すると、太らせた枠の画素も数える。"""
  image = white_image(40, 40)
  cv2.rectangle(image, (10, 10), (29, 29), RED, 1)
  path = write_png(tmp_path / "square.png", image)

  count = count_enclosed_pixels(
    path,
    red_min=120,
    diff_min=40,
    boundary_thickness=3,
    include_line=True,
  )
  assert count == 22 * 22
def test_measure_left_red_height(tmp_path):
  """左側の赤い縦線の高さを測る。"""
  image = white_image(40, 40)
  cv2.line(image, (3, 5), (3, 34), RED, 1)
  cv2.line(image, (30, 0), (30, 39), RED, 1)
  path = write_png(tmp_path / "line.png", image)

  height = measure_left_red_height(
    path,
    red_min=120,
    diff_min=40,
    min_column_pixels=10,
  )
  assert height == 30
def test_output_jpg_enclosed_pixels():
  """output.jpg の赤線で囲まれた部分は README の値（732,466 px）になる。"""
  count = count_enclosed_pixels(
    HERE / "output.jpg",
    red_min=120,
    diff_min=40,
    boundary_thickness=3,
    include_line=False,
  )
  assert count == 732_466
def test_output_jpg_left_red_height():
  """output.jpg の左側の赤線の高さは README の値（926 px）になる。"""
  height = measure_left_red_height(
    HERE / "output.jpg",
    red_min=120,
    diff_min=40,
    min_column_pixels=100,
  )
  assert height == 926
