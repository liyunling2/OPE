# Restaurant Growth Momentum Analysis - Quick Start Guide (Updated)

## Overview

This analysis identifies restaurants with the highest growth potential and provides data-driven recommendations for optimizing marketing resource allocation.

## What This Analysis Does

Unlike your previous static performance matrix, this analysis:

1. **Tracks Growth Over Time** - Calculates month-over-month and rolling average growth rates
2. **Identifies Momentum** - Finds restaurants that are trending upward, not just those currently performing well
3. **Segments by Strategy** - Classifies restaurants into actionable categories with adjustable thresholds
4. **Prioritizes Investment** - Ranks restaurants by their combined growth potential and current performance
5. **Provides Actionable Recommendations** - Gives specific marketing strategies for each segment
6. **Filters Future Bookings** - Only analyzes completed bookings for accurate growth metrics
7. **Configurable Parameters** - Adjust thresholds and weights to match your market reality

## Key Differences from Your Original Code

### Your Original Approach:
```python
# Shows CURRENT state only
- Total bookings (cumulative)
- Total revenue (cumulative)
- Quadrants based on median splits
- Identifies current "Stars"
```

### New Growth Momentum Approach:
```python
# Shows TRAJECTORY and POTENTIAL
- Month-over-month growth rates
- 3-month rolling averages
- Momentum scoring (growth + volume)
- Identifies future opportunities
```

## The Four Segments

### 1. Rising Stars (HIGHEST PRIORITY)
- **Characteristics**: High volume + High growth
- **Investment Level**: 50% of marketing budget
- **Strategy**: Invest heavily, scale aggressively
- **Expected ROI**: Very High
- **Actions**:
  - Increase ad spend by 30-50%
  - Create loyalty programs
  - Premium ad placements
  - Feature prominently

### 2. Emerging Opportunities (HIGH PRIORITY)
- **Characteristics**: Low volume + High growth
- **Investment Level**: 30% of marketing budget
- **Strategy**: Scale up to convert momentum into volume
- **Expected ROI**: High
- **Actions**:
  - Targeted acquisition campaigns
  - First-time booking discounts
  - Social media boost
  - Influencer partnerships

### 3. Established Players (MEDIUM PRIORITY)
- **Characteristics**: High volume + Low/stable growth
- **Investment Level**: 15% of marketing budget
- **Strategy**: Maintain and optimize
- **Expected ROI**: Medium
- **Actions**:
  - Retention campaigns
  - Conversion rate optimization
  - Menu/experience testing
  - Customer feedback collection

### 4. Needs Attention (LOW PRIORITY)
- **Characteristics**: Low volume + Low/negative growth
- **Investment Level**: 5% of marketing budget
- **Strategy**: Minimize spend, fix fundamentals
- **Expected ROI**: Low
- **Actions**:
  - Operational audit
  - Reduce marketing temporarily
  - Address root causes
  - Pricing/menu adjustments

## How to Use

### Complete Workflow (Recommended - 3 Steps)

#### Step 1: Preprocess Your Data (NEW - IMPORTANT!)

```python
from preprocessing_helpers import prepare_data_for_analysis

# This filters out future bookings and shows your date range
df_clean, info = prepare_data_for_analysis(df, date_column='booking_date')

# Output shows:
# - How many future bookings were removed
# - Your actual date range
# - Months of data available
# - Recommended lookback period
```

**Why this matters:**
- Future bookings haven't happened yet (no-show risk unknown)
- Including them gives false growth signals
- Recent months look artificially low if you include future bookings

#### Step 2: Configure Analysis Parameters

```python
from restaurant_growth_analysis_adjustable import GrowthAnalysisConfig, run_growth_analysis_adjustable

# Customize to match your market
config = GrowthAnalysisConfig()
config.recent_months = 6              # Look back 6 months (more stable than 3)
config.min_monthly_bookings = 5       # Filter out very low volume
config.high_volume_threshold = 15     # What YOU consider "high volume"
config.high_growth_threshold = 3      # What YOU consider "high growth" (%)
config.weight_volume = 0.6           # 60% based on current volume
config.weight_booking_growth = 0.2   # 20% based on booking growth
config.weight_revenue_growth = 0.2   # 20% based on revenue growth
```

#### Step 3: Run Analysis

```python
# Run with your config
restaurant_summary, monthly_metrics, recommendations = run_growth_analysis_adjustable(
    df_clean,
    output_dir='./outputs',
    config=config
)
```

---

### Quick Start with Presets (Even Faster!)

Don't want to configure? Use a preset:

**Conservative (Recommended for most cases):**
```python
from restaurant_growth_analysis_adjustable import run_growth_analysis_adjustable, get_conservative_config
from preprocessing_helpers import prepare_data_for_analysis

# Preprocess
df_clean, info = prepare_data_for_analysis(df)

# Use conservative config (more forgiving thresholds)
config = get_conservative_config()
# - 6-month lookback
# - High volume = 10 bookings/month
# - High growth = 2%
# - 70% weight on volume

# Run
restaurant_summary, monthly_metrics, recommendations = run_growth_analysis_adjustable(
    df_clean, output_dir='./outputs', config=config
)
```

**Other presets available:**
- `get_balanced_config()` - Middle ground (20 bookings, 5% growth)
- `get_aggressive_config()` - Strict criteria (50 bookings, 15% growth)

---

### Complete Copy-Paste Example

```python
import pandas as pd
from preprocessing_helpers import prepare_data_for_analysis
from restaurant_growth_analysis_adjustable import GrowthAnalysisConfig, run_growth_analysis_adjustable

# Load your data (adjust as needed)
# df = pd.read_csv('your_data.csv')

# Step 1: Preprocess (removes future bookings)
print("Preprocessing data...")
df_clean, info = prepare_data_for_analysis(df, date_column='booking_date')

# Step 2: Configure analysis
print("Configuring analysis...")
config = GrowthAnalysisConfig()
config.recent_months = 6              
config.high_volume_threshold = 15     # Adjust based on your diagnostic output
config.high_growth_threshold = 3      
config.weight_volume = 0.6           

# Step 3: Run analysis
print("Running analysis...")
restaurant_summary, monthly_metrics, recommendations = run_growth_analysis_adjustable(
    df_clean,
    output_dir='./outputs',
    config=config
)

print("\n✓ Analysis complete! Check ./outputs folder for results.")
```

### What Gets Generated

1. **Visualizations** (PNG files):
   - `growth_momentum_matrix.png` - 4-panel analysis dashboard
   - `top_performers_trends.png` - Time series of top 10 restaurants

2. **Data Files** (CSV):
   - `restaurant_summary.csv` - All restaurants with growth metrics
   - `monthly_metrics.csv` - Month-by-month performance data

3. **Reports** (TXT):
   - `growth_analysis_report.txt` - Executive summary with recommendations
   - `analysis_config.txt` - Configuration settings used (NEW!)

All files saved to your specified output directory.

## Understanding the Metrics

### Momentum Score
A composite metric combining:
- 40% Booking growth rate
- 40% Revenue growth rate  
- 20% Current volume (normalized)

**Higher score = Higher priority for marketing investment**

### Growth Rates
- **Month-over-Month**: Change from previous month (can be volatile)
- **3-Month Average**: Smoothed trend (more reliable indicator)

### Volume Metrics
- **Recent Avg Bookings**: Average monthly bookings in last 3 months
- **Total Bookings**: Historical cumulative bookings

## Interpreting the Growth Matrix

The main visualization has 4 quadrants:

```
           High Growth
                |
    SCALE UP    |    INVEST HEAVILY
 (Low Volume)   |    (High Volume)
                |
----------------+----------------
                |
   RECONSIDER   |     OPTIMIZE
 (Low Volume)   |    (High Volume)
                |
           Low Growth
```

**X-axis**: Recent average monthly bookings (volume)
**Y-axis**: Average growth rate (momentum)
**Size**: Recent revenue (business value)
**Color**: Segment classification

---

## Adjusting Thresholds (If Results Look Marginal)

After running the analysis, check your diagnostic output:

```python
print(f"Median bookings/month: {restaurant_summary['recent_avg_bookings'].median():.1f}")
print(f"Median growth rate: {restaurant_summary['avg_growth_rate'].median():.1f}%")
print("\nSegment distribution:")
print(restaurant_summary['segment'].value_counts())
```

### If Your Results Look Too Strict:

**Problem**: Most restaurants in "Needs Attention" segment

**Solution**: Lower your thresholds
```python
config = GrowthAnalysisConfig()
config.high_volume_threshold = 8      # Lower from 15
config.high_growth_threshold = 2      # Lower from 3%
config.weight_volume = 0.7           # Favor volume more (70%)
```

### If Your Results Look Too Loose:

**Problem**: Too many "Rising Stars" (50%+ of restaurants)

**Solution**: Raise your thresholds
```python
config = GrowthAnalysisConfig()
config.high_volume_threshold = 25     # Raise from 15
config.high_growth_threshold = 8      # Raise from 3%
config.weight_booking_growth = 0.35  # Favor growth more
config.weight_revenue_growth = 0.35
```

### If Growth Rates Are Too Volatile:

**Problem**: Growth rates swinging wildly (e.g., +100%, -80%)

**Solution**: Use longer lookback period
```python
config = GrowthAnalysisConfig()
config.recent_months = 6              # Or even 9-12 months
config.min_monthly_bookings = 10      # Filter very low volume
```

For more details, see `METHODOLOGY_EXPLAINED.md`

## Key Business Questions Answered

### 1. "Which restaurants are gaining momentum?"
**Answer**: Look at the `avg_growth_rate` column in `restaurant_summary.csv`
- Positive values = Growing
- Values > 10% = High growth
- Sort by this to see fastest growing

### 2. "Which have highest growth potential?"
**Answer**: Sort by `momentum_score` in `restaurant_summary.csv`
- Top 20 are your best investment opportunities
- These combine growth trajectory with current performance

### 3. "Where should we focus marketing resources?"
**Answer**: Use the `investment_tier` in `marketing_investment_priority.csv`
- Tier 1: Allocate 50% of budget
- Tier 2: Allocate 30% of budget
- Tier 3: Allocate 15% of budget
- Tier 4: Allocate 5% of budget

## Example Analysis Workflow

```python
# Step 1: Load your data
df = pd.read_csv('your_booking_data.csv')

# Step 2: Run analysis
restaurant_summary, monthly_metrics, recommendations = run_growth_analysis(df)

# Step 3: Get your top investment targets
top_20_investments = restaurant_summary.nlargest(20, 'momentum_score')
print(top_20_investments[['name', 'segment', 'momentum_score', 'avg_growth_rate']])

# Step 4: Focus on Rising Stars
rising_stars = restaurant_summary[restaurant_summary['segment'] == 'Rising Stars']
print(f"\nInvest heavily in these {len(rising_stars)} restaurants:")
print(rising_stars['name'].tolist())

# Step 5: Review recommendations
for rec in recommendations:
    if rec['Priority'] == 'HIGHEST':
        print(f"\nTop Priority: {rec['Segment']}")
        print(f"Restaurants: {rec['Restaurants']}")
        print(f"Action: {rec['Recommendation']}")
```

## Common Questions

### Q: Why not just invest in current best performers?
**A**: Current top performers might be plateauing. Growth momentum identifies restaurants with upward trajectories - these give better ROI because you're amplifying existing momentum rather than fighting stagnation.

### Q: What if a restaurant has low volume but high growth?
**A**: This is "Emerging Opportunities" segment - they're your future stars. Invest to help them scale while momentum is strong.

### Q: Should I ignore "Needs Attention" restaurants completely?
**A**: Not completely, but marketing won't fix operational problems. First diagnose why they're struggling (pricing, food quality, location, etc.), then decide if they're salvageable.

### Q: How often should I run this analysis?
**A**: Monthly is ideal. Growth rates and momentum can shift quickly in the restaurant industry.

## Customization Options

You can adjust parameters in the analysis:

```python
# Change the lookback period for "recent" performance
restaurant_summary = create_growth_summary(monthly_metrics, recent_months=6)  # 6 months instead of 3

# Change number of top performers to visualize
fig = plot_trend_analysis(monthly_metrics, restaurant_summary, top_n=15)  # Top 15 instead of 10

# Adjust momentum score weights
# Edit the momentum_score calculation in create_growth_summary() function
```

## Technical Requirements

### Required DataFrame Columns:
- `booking_date` or `date`: Date of booking
- `restaurant_id`: Unique restaurant ID
- `name`: Restaurant name
- `revenue_baht` or similar: Revenue amount
- `total_guests`: Number of guests
- `id`: Booking ID (for counting)

### Optional Columns:
- `no_show`: Filter out no-shows
- `is_outlier`: Filter out outliers
- `adult`, `kids`: Guest breakdown
- `created_at`: Booking creation date

## Troubleshooting

### Issue: "Future bookings detected"
**Solution**: Use the preprocessing helper to filter them out:
```python
from preprocessing_helpers import prepare_data_for_analysis
df_clean, info = prepare_data_for_analysis(df)
```

### Issue: "Not enough data for growth analysis"
**Solution**: 
- Check your date range using `prepare_data_for_analysis()`
- Analysis works best with 6+ months of data
- Adjust `config.recent_months` to match your available data

### Issue: "Growth rates seem extreme" (+100%, -80%, etc.)
**Solution**: 
1. Increase `config.recent_months` to 6 or more
2. Increase `config.min_monthly_bookings` to filter low-volume volatility
3. Check for data quality issues (gaps, errors)

### Issue: "All restaurants classified as one segment"
**Solution**: Switch to absolute thresholds instead of medians:
```python
config.use_absolute_thresholds = True
config.high_volume_threshold = 15  # Set YOUR threshold
config.high_growth_threshold = 3   # Set YOUR threshold
```

### Issue: "Results look marginal" (low momentum scores, mostly "Needs Attention")
**Solution**: Your market may be slower-growing than default thresholds assume:
```python
# Use conservative config
config = get_conservative_config()
# OR manually adjust:
config.high_volume_threshold = 8   # Lower
config.high_growth_threshold = 2   # Lower
config.weight_volume = 0.7         # Favor current volume
```

### Issue: "Error: FileNotFoundError (Windows path issue)"
**Solution**: Specify your own output directory:
```python
import os
output_dir = os.path.expanduser('~/Desktop/restaurant_analysis')
os.makedirs(output_dir, exist_ok=True)
# Then use: run_growth_analysis_adjustable(df, output_dir=output_dir, ...)
```

### Issue: "ValueError: unconverted data remains (date parsing)"
**Solution**: Dates are probably already datetime type or need preprocessing:
```python
# Check if already datetime
print(df['booking_date'].dtype)

# If object type, convert first:
df['booking_date'] = pd.to_datetime(df['booking_date'].astype(str), format='%Y-%m-%d', errors='coerce')
```

## Next Steps

1. **Preprocess your data** - Use `prepare_data_for_analysis()` to filter future bookings and check date range
2. **Check diagnostics** - Look at median bookings and growth rates to understand your market
3. **Choose configuration** - Start with `get_conservative_config()` if unsure
4. **Run the analysis** - Execute the complete workflow
5. **Review results** - Check visualizations and segment distribution
6. **Adjust if needed** - If results look too strict/loose, adjust thresholds and re-run
7. **Implement recommendations** - Focus marketing on Rising Stars and Emerging Opportunities
8. **Track monthly** - Re-run analysis monthly to measure impact and catch momentum shifts

## Quick Reference - Which Script to Use

| Your Situation | Use This |
|----------------|----------|
| First time running | `preprocessing_helpers.py` + `get_conservative_config()` |
| Need flexibility | `restaurant_growth_analysis_adjustable.py` |
| Results too marginal | Adjust thresholds in config |
| Want to understand methodology | Read `METHODOLOGY_EXPLAINED.md` |
| Different column names | Read `COLUMN_MAPPING_GUIDE.md` |
| Just want it to work | Use the "Complete Copy-Paste Example" above |

## Support Files Included

1. `restaurant_growth_analysis.py` - Original analysis module (basic version)
2. `restaurant_growth_analysis_adjustable.py` - Configurable analysis with presets (RECOMMENDED)
3. `preprocessing_helpers.py` - Data preprocessing utilities (filter future bookings, date diagnostics)
4. `growth_analysis_notebook.ipynb` - Interactive Jupyter notebook
5. `QUICK_START_GUIDE.md` - This document
6. `METHODOLOGY_EXPLAINED.md` - Detailed explanation of calculations and how to adjust
7. `COLUMN_MAPPING_GUIDE.md` - Guide for handling different column names

## Contact & Feedback

This analysis framework can be customized for your specific needs. Common customizations include:
- Industry-specific metrics
- Seasonal adjustment
- Multi-channel attribution
- Geographic segmentation
- Price point analysis

---

**Remember**: Marketing amplifies what's already working. Invest where momentum exists, not where you hope to create it from scratch.
