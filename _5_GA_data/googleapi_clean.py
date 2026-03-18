import pandas as pd
BASE_PATH = "../data/marketing/googleAPI/"
agg_new_users = pd.read_parquet(BASE_PATH + "agg_new_users.parquet")
agg_restaurants = pd.read_parquet(BASE_PATH + "agg_restaurants.parquet")
agg_user_activity = pd.read_parquet(BASE_PATH + "agg_user_activity.parquet")
campaign_impact = pd.read_parquet(BASE_PATH + "campaign_impact.parquet")
campaigns_outreach = pd.read_parquet(BASE_PATH + "campaigns_outreach.parquet")
user_demographics = pd.read_parquet(BASE_PATH + "user_demographics.parquet")
session_source = pd.read_parquet(BASE_PATH + "session_source.parquet")


# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = {
    'session_source': {
        'datetime': ['yearMonth'],
        'category': ['sessionSource'],
        'float': [
            'sessions',
            'sessionKeyEventRate:add_to_cart',
            'sessionKeyEventRate:add_to_wishlist',
            'sessionKeyEventRate:begin_checkout',
            'sessionKeyEventRate:booking',
            'sessionKeyEventRate:ecommerce_purchase',
            'sessionKeyEventRate:in_app_purchase',
        ],
    },
    'campaigns_outreach': {
        'datetime': ['yearMonth'],
        'string':   ['campaignId'],
        'category': ['campaignName'],
        'float':    ['sessions'],
    },
    'campaign_impact': {
        'datetime': ['yearMonth'],
        'string':   ['itemId'],
        'category': ['itemName'],
        'float':    ['itemsViewed', 'itemsAddedToCart', 'itemsPurchased', 'itemRevenue'],
    },
    'agg_new_users': {
        'datetime': ['yearMonth'],
        'category': ['firstUserSourcePlatform'],
        'float':    ['newUsers', 'activeUsers'],
    },
    'agg_user_activity': {
        'datetime': ['yearMonth'],
        'float': [
            'activeUsers', 'engagedSessions', 'sessionsPerUser',
            'userEngagementDuration', 'addToCarts', 'checkouts',
            'ecommercePurchases', 'totalPurchasers',
            'purchaseRevenue', 'totalRevenue',
        ],
    },
    'agg_restaurants': {
        'int':      ['year'],
        'string':   ['itemId'],
        'category': ['itemName'],
    },
    'user_demographics': {
        'datetime': ['yearMonth'],
        'category': ['userAgeBracket', 'userGender'],
        'float': [
            'activeUsers', 'engagedSessions',
            'ecommercePurchases', 'engagementRate', 'purchaseRevenue',
        ],
    },
}


# ── Drop row values ────────────────────────────────────────────────────
# for each col to row, drop ay values that match
FILTER_RULES = {
    'session_source': [
        ('sessionSource', ['(data not available)']),
    ],
    'campaigns_outreach': [
        ('campaignId',   ['(not set)']),
        ('campaignName', ['(not set)']),
    ],
    'campaign_impact': [
        ('itemId',   ['(not set)']),
        ('itemName', ['(not set)']),
    ],
    'agg_new_users': [
        ('firstUserSourcePlatform', ['(not set)']),
    ],
    'user_demographics': [
        ('userAgeBracket', ['unknown', 'unspecified', 'none', '', '(other)']),
        ('userGender',     ['unknown', 'unspecified', 'none', '']),
    ],
}


# ── Core helpers ──────────────────────────────────────────────────────────────
def _is_string_like(series: pd.Series) -> bool:
    """True if series holds text data (object, string, or category of strings)."""
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
        return True
    if hasattr(series, 'cat'):
        return pd.api.types.is_string_dtype(series.cat.categories)
    return False
def _apply_filters(df: pd.DataFrame, name: str) -> pd.DataFrame:
    rules = FILTER_RULES.get(name, [])
    for col, bad_values in rules:
        if col not in df.columns:
            continue
        if not _is_string_like(df[col]):
            print(f'  [{name}] skipping filter on {col!r} — not a string column ({df[col].dtype})')
            continue
        bad_lower = [v.lower() for v in bad_values]
        before = len(df)
        # Work on raw string representation so this runs before or after casting
        df = df[~df[col].astype(str).str.strip().str.lower().isin(bad_lower)]
        dropped = before - len(df)
        if dropped:
            print(f'  [{name}] dropped {dropped:,} rows where {col!r} in {bad_values}')

    # Remove stale category levels after row drops
    for col in df.select_dtypes('category').columns:
        df[col] = df[col].cat.remove_unused_categories()

    return df

def _apply_dtypes(df: pd.DataFrame, name: str) -> pd.DataFrame:
    schema    = SCHEMA.get(name, {})
    cast_cols = set()
 
    for col in schema.get('datetime', []):
        if col not in df.columns:
            continue
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            # parquet may store yearMonth as int (e.g. 202301) or string
            df[col] = pd.to_datetime(df[col].astype(str).str.strip(), format='%Y%m', errors='coerce')
        cast_cols.add(col)
 
    for col in schema.get('int', []):
        if col not in df.columns:
            continue
        if not pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int16')
        cast_cols.add(col)
 
    for col in schema.get('float', []):
        if col not in df.columns:
            continue
        if not pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        cast_cols.add(col)
 
    for col in schema.get('string', []):
        if col not in df.columns:
            continue
        if df[col].dtype != 'string':
            df[col] = df[col].astype(str).str.strip().astype('string')
        cast_cols.add(col)
 
    for col in schema.get('category', []):
        if col not in df.columns:
            continue
        if df[col].dtype.name != 'category':
            df[col] = df[col].astype(str).str.strip().astype('category')
        cast_cols.add(col)
 
    leftover = [c for c in df.select_dtypes('object').columns if c not in cast_cols]
    if leftover:
        print(f'  WARNING [{name}] columns still object (not in schema): {leftover}')
 
    return df

def _dtype_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a summary DataFrame of dtypes, null counts, and a sample value."""
    report = pd.DataFrame({
        'dtype':       df.dtypes.astype(str),
        'nulls':       df.isnull().sum(),
        'null_%':      (df.isnull().mean() * 100).round(2),
        'n_unique':    df.nunique(),
        'sample':      [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })
    print(f'\n{"─" * 55}')
    print(f'  {name}  |  {len(df):,} rows × {df.shape[1]} cols')
    print(f'{"─" * 55}')
    print(report.to_string())
    return report


# ── Public API ────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame, name: str, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = _apply_filters(df, name)
    df = _apply_dtypes(df, name)

    if verbose:
        _dtype_report(df, name)

    return df


def clean_all(
    raw_frames: dict,
    verbose: bool = True,
) -> dict:
    cleaned = {}
    for name, df in raw_frames.items():
        cleaned[name] = clean(df, name, verbose=verbose)

    print('\n All datasets cleaned.')
    return cleaned
