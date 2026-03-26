from __future__ import annotations

TEXT_COLOR = "#111827"
MUTED_TEXT = "#6b7280"
GRID_COLOR = "#e5e7eb"
AXIS_COLOR = "#6b7280"
SOFT_DIVIDER = "#cbd5e1"
SURFACE_COLOR = "#ffffff"
SURFACE_ALT = "#f8f9fa"
BORDER_COLOR = "#e0e0e0"
ACCENT_RED = "#cc0000"
HEADER_NAVY = "#0e1628"
SOFT_NAVY = "#7e97b8"

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_COLOR, family="DM Sans"),
    xaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False, color=AXIS_COLOR),
    yaxis=dict(gridcolor=GRID_COLOR, showline=False, zeroline=False, color=AXIS_COLOR),
    margin=dict(l=0, r=0, t=30, b=0),
)

BASE_LAYOUT = {k: v for k, v in CHART_THEME.items() if k not in ("xaxis", "yaxis")}
AXIS = dict(gridcolor=GRID_COLOR, showline=False, zeroline=False, color=AXIS_COLOR)
