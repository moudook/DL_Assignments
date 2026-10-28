"""
export_figures.py - copy the figures report.tex uses from out/ into
report/figures/, flattened to RGB and at most MAX_W px wide (keeps the Overleaf
compile fast). Run after `python run.py all`.

    python export_figures.py [--outdir out] [--report ../report]
"""

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

import config

MAX_W = 1600


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(config.CODE_DIR / "out"))
    ap.add_argument("--report", default=str(config.CODE_DIR.parent / "report"))
    args = ap.parse_args()
    outdir, report = Path(args.outdir), Path(args.report)

    tex = (report / "report.tex").read_text(encoding="utf-8")
    names = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    missing = [n for n in names if not (outdir / n).exists()]
    if missing:
        print("MISSING in %s:\n  %s" % (outdir, "\n  ".join(missing)))
        return 1
    for n in names:
        im = Image.open(outdir / n)
        if im.mode != "RGB":
            bg = Image.new("RGB", im.size, "white")
            bg.paste(im, mask=im.convert("RGBA").split()[3])
            im = bg
        if im.width > MAX_W:
            im = im.resize((MAX_W, round(im.height * MAX_W / im.width)), Image.LANCZOS)
        dst = report / "figures" / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, optimize=True)
    print("%d figures -> %s" % (len(names), report / "figures"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
