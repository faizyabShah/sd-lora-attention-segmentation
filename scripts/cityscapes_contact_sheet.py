"""Build labeled contact sheets for visual review, preserving candidate files."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=25)
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--columns', type=int, default=5)
    ap.add_argument('--cell', type=int, default=256)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    paths = ([a.root / 'candidates' / (name + '.png') for name in a.ids]
             if a.ids else [a.root / 'candidates' / f'cs_{i:04d}.png'
                            for i in range(a.start, a.start + a.count)])
    paths = [p for p in paths if p.exists()]
    assert paths, 'No candidate images ready'
    label = 34
    sheet = Image.new('RGB', (a.columns * a.cell, ((len(paths) + a.columns - 1) // a.columns) * (a.cell + label)), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    for i, p in enumerate(paths):
        x, y = (i % a.columns) * a.cell, (i // a.columns) * (a.cell + label)
        with Image.open(p) as img:
            thumb = ImageOps.contain(img.convert('RGB'), (a.cell, a.cell))
            sheet.paste(thumb, (x + (a.cell - thumb.width) // 2, y + (a.cell - thumb.height) // 2))
        record = json.loads(p.with_suffix('.json').read_text())
        draw.text((x + 4, y + a.cell + 2), f'{p.stem}  {record["family"]}', font=font, fill='black')
        draw.text((x + 4, y + a.cell + 17), f'{record["width"]}x{record["height"]}' + (' FLAGGED' if record.get('nsfw_flagged') else ''), font=font, fill='black')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.output)
    print(a.output, f'({len(paths)} images)')


if __name__ == '__main__':
    main()
