"""
Quick visual test for subtitle font-shrinking.
Generates sample subtitle bar images for various text lengths and saves them to test_subtitles/.
Run: python test_subtitle_render.py
Then open test_subtitles/ and inspect the images.
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image, ImageDraw
from run_all import _font_for_lang, _wrap_for_lang, _render_m4_subtitle_bar

OUT_DIR = "test_subtitles"
os.makedirs(OUT_DIR, exist_ok=True)

SUB_W    = 1080
SLOT_H   = 90
BASE_SRC = 52
BASE_DST = 50
MAX_LINES = 2

test_cases = [
    ("Short", "Hello world", "สวัสดีชาวโลก"),
    ("Medium", "This is a medium length sentence for testing.", "นี่คือประโยคทดสอบความยาวปานกลาง"),
    ("Long",
     "This is quite a long English sentence that really pushes the limits of what can fit on two lines at the normal font size.",
     "นี่คือประโยคภาษาไทยที่ค่อนข้างยาวและมีเนื้อหามากเพื่อทดสอบว่าการตัดคำทำงานได้ถูกต้องหรือไม่"),
    ("Very long",
     "An extremely verbose and unnecessarily complicated sentence designed specifically to force multiple lines of text beyond what two lines can accommodate.",
     "ประโยคที่ยาวมากๆเพื่อทดสอบการลดขนาดตัวอักษรโดยอัตโนมัติเมื่อข้อความมีปริมาณมากเกินสองบรรทัด"),
]

_tmp  = Image.new('RGB', (SUB_W, 400))
_draw = ImageDraw.Draw(_tmp)
max_w = SUB_W - 60

for label, src_text, dst_text in test_cases:
    # Run the same font-fitting logic as the pipeline
    chosen_fs_src, chosen_fs_dst = BASE_SRC, BASE_DST
    src_lines, dst_lines = [], []
    for shrink in range(0, BASE_SRC - 18, 2):
        fs_src = BASE_SRC - shrink
        fs_dst = BASE_DST - shrink
        f_src = _font_for_lang('en', fs_src)
        f_dst = _font_for_lang('th', fs_dst)
        src_all = _wrap_for_lang('en', src_text, f_src, _draw, max_w, max_lines=99)
        dst_all = _wrap_for_lang('th', dst_text, f_dst, _draw, max_w, max_lines=99)
        src_lines = src_all[:MAX_LINES]
        dst_lines = dst_all[:MAX_LINES]
        chosen_fs_src, chosen_fs_dst = fs_src, fs_dst
        if len(src_all) <= MAX_LINES and len(dst_all) <= MAX_LINES:
            break

    bar = _render_m4_subtitle_bar(
        src_lines, dst_lines,
        highlight_words=[],
        width=SUB_W, slot_h=SLOT_H,
        src_lang='en', dst_lang='th',
        font_size_src=chosen_fs_src, font_size_dst=chosen_fs_dst,
    )
    out_path = os.path.join(OUT_DIR, f"{label.replace(' ', '_')}_fs{chosen_fs_src}.png")
    bar.save(out_path)
    print(f"[{label}] font={chosen_fs_src}/{chosen_fs_dst} "
          f"src_lines={len(src_lines)} dst_lines={len(dst_lines)} → {out_path}")
    print(f"  src: {src_lines}")
    print(f"  dst: {dst_lines}")

print(f"\nDone. Open {OUT_DIR}/ to inspect the images.")
