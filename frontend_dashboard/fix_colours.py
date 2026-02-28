# -*- coding: utf-8 -*-
"""
fix_colors.py
Run once from inside frontend_dashboard/:
    python fix_colors.py

Converts all 8-digit hex colors like #e74c3c22 -> rgba(231,76,60,0.13)
Plotly does not support the 8-digit hex (alpha channel) format.
"""

import re
from pathlib import Path

def hex8_to_rgba(match):
    hex6 = match.group(1)
    alpha_hex = match.group(2)
    r = int(hex6[0:2], 16)
    g = int(hex6[2:4], 16)
    b = int(hex6[4:6], 16)
    a = round(int(alpha_hex, 16) / 255, 2)
    return f'"rgba({r},{g},{b},{a})"'

pattern = re.compile(r'["\']#([0-9a-fA-F]{6})([0-9a-fA-F]{2})["\']')

# Fix all .py files in pages/ and data/ and root
targets = (
    list(Path("pages").glob("*.py")) +
    list(Path("data").glob("*.py")) +
    [Path("app.py")]
)

for fpath in targets:
    if not fpath.exists():
        continue
    text = fpath.read_text(encoding="utf-8")
    fixed, count = pattern.subn(hex8_to_rgba, text)
    if count:
        fpath.write_text(fixed, encoding="utf-8")
        print(f"  Fixed {count} color(s) in {fpath}")
    else:
        print(f"  OK    {fpath}")

print("\nDone. Restart streamlit.")