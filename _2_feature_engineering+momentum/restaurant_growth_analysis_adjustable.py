# ADJUSTABLE GROWTH ANALYSIS - For tuning to your specific data

from restaurant_growth_analysis import *

class GrowthAnalysisConfig:
    """
    Configuration class for adjustable growth analysis parameters
    """
    def __init__(self):
        # TIME PERIOD SETTINGS
        self.recent_months = 3  # How many months to consider "recent"
        self.min_months_required = 3  # Minimum months of data required per restaurant
        
        # VOLUME THRESHOLDS
        self.min_monthly_bookings = 5  # Filter out restaurants below this
        self.high_volume_threshold = 20  # What counts as "high volume"
        
        # GROWTH THRESHOLDS
        self.high_growth_threshold = 5  # % growth to be considered "high growth"
        self.declining_threshold = -10  # % below which is "declining"
        
        # MOMENTUM SCORE WEIGHTS
        self.weight_volume = 0.5  # Weight for current volume (0-1)
        self.weight_booking_growth = 0.25  # Weight for booking growth (0-1)
        self.weight_revenue_growth = 0.25  # Weight for revenue growth (0-1)
        
        # SEGMENTATION METHOD
        self.use_absolute_thresholds = True  # True = use thresholds, False = use medians
        
        # OUTLIER HANDLING
        self.filter_outliers = True
        self.filter_no_shows = True
        
    def validate(self):
        """Check that weights sum to 1"""
        total_weight = self.weight_volume + self.weight_booking_growth + self.weight_revenue_growth
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0 (currently {total_weight})")


def create_growth_summary_adjustable(monthly_metrics, config):
    """
    Create growth summary with configurable parameters
    """
    # Get the most recent N months of data
    max_period = monthly_metrics['period'].max()
    recent_data = monthly_metrics[monthly_metrics['period'] >= max_period - config.recent_months]
    
    # Calculate recent performance
    recent_summary = recent_data.groupby(['restaurant_id', 'name']).agg({
        'bookings': 'mean',
        'revenue': 'mean',
        'bookings_3m_avg_growth': 'last',
        'revenue_3m_avg_growth': 'last',
        'avg_party_size': 'mean',
        'revenue_per_booking': 'mean'
    }).reset_index()
    
    recent_summary.columns = ['restaurant_id', 'name', 'recent_avg_bookings', 'recent_avg_revenue',
                              'avg_growth_rate', 'revenue_growth_rate', 'avg_party_size', 'revenue_per_booking']
    
    # Calculate total historical performance
    total_metrics = monthly_metrics.groupby(['restaurant_id', 'name']).agg({
        'bookings': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    total_metrics.columns = ['restaurant_id', 'name', 'total_bookings', 'total_revenue']
    
    # Merge
    restaurant_summary = recent_summary.merge(total_metrics, on=['restaurant_id', 'name'])
    
    # Calculate ADJUSTABLE momentum score
    max_bookings = restaurant_summary['recent_avg_bookings'].max()
    
    # Normalize volume to 0-100
    volume_component = (restaurant_summary['recent_avg_bookings'] / max_bookings) * 100
    
    # Normalize growth to 0-100 (clipping extreme values)
    booking_growth_component = restaurant_summary['avg_growth_rate'].clip(-50, 50) + 50
    revenue_growth_component = restaurant_summary['revenue_growth_rate'].clip(-50, 50) + 50
    
    # Weighted combination based on config
    restaurant_summary['momentum_score'] = (
        (volume_component * config.weight_volume) +
        (booking_growth_component * config.weight_booking_growth) +
        (revenue_growth_component * config.weight_revenue_growth)
    )
    
    return restaurant_summary


def identify_growth_segments_adjustable(restaurant_summary, config):
    """
    Classify restaurants with configurable thresholds
    """
    if config.use_absolute_thresholds:
        # Use user-defined thresholds
        def classify_segment(row):
            high_volume = row['recent_avg_bookings'] >= config.high_volume_threshold
            high_growth = row['avg_growth_rate'] >= config.high_growth_threshold
            
            if high_volume and high_growth:
                return 'Rising Stars'
            elif high_volume and not high_growth:
                return 'Established Players'
            elif not high_volume and high_growth:
                return 'Emerging Opportunities'
            else:
                return 'Needs Attention'
    else:
        # Use medians (original method)
        median_bookings = restaurant_summary['recent_avg_bookings'].median()
        median_growth = restaurant_summary['avg_growth_rate'].median()
        
        def classify_segment(row):
            high_volume = row['recent_avg_bookings'] >= median_bookings
            high_growth = row['avg_growth_rate'] >= median_growth
            
            if high_volume and high_growth:
                return 'Rising Stars'
            elif high_volume and not high_growth:
                return 'Established Players'
            elif not high_volume and high_growth:
                return 'Emerging Opportunities'
            else:
                return 'Needs Attention'
    
    restaurant_summary['segment'] = restaurant_summary.apply(classify_segment, axis=1)
    
    # Additional classification based on growth trend
    def classify_trend(growth_rate):
        if growth_rate > 10:
            return 'High Growth'
        elif growth_rate > 0:
            return 'Moderate Growth'
        elif growth_rate > config.declining_threshold:
            return 'Stable/Slight Decline'
        else:
            return 'Declining'
    
    restaurant_summary['trend'] = restaurant_summary['avg_growth_rate'].apply(classify_trend)
    
    return restaurant_summary


def run_growth_analysis_adjustable(df, output_dir='./outputs', column_mapping=None, config=None):
    """
    Run growth analysis with adjustable parameters
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    output_dir : str
        Output directory
    column_mapping : dict
        Column name mapping
    config : GrowthAnalysisConfig
        Configuration object with custom parameters
    """
    import os
    
    # Use default config if none provided
    if config is None:
        config = GrowthAnalysisConfig()
    
    # Validate config
    config.validate()
    
    print("Starting Adjustable Restaurant Growth Momentum Analysis...")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Recent months: {config.recent_months}")
    print(f"  Min monthly bookings: {config.min_monthly_bookings}")
    print(f"  High volume threshold: {config.high_volume_threshold} bookings/month")
    print(f"  High growth threshold: {config.high_growth_threshold}%")
    print(f"  Momentum weights: Volume={config.weight_volume}, Booking Growth={config.weight_booking_growth}, Revenue Growth={config.weight_revenue_growth}")
    print(f"  Using: {'Absolute thresholds' if config.use_absolute_thresholds else 'Median-based segmentation'}")
    print("=" * 80)
    
    # Step 1: Prepare data
    print("\n1. Preparing data...")
    if column_mapping:
        print(f"   Applying column mapping: {column_mapping}")
    df, df_arrived = prepare_data(df, column_mapping)
    print(f"   Total records: {len(df):,}")
    print(f"   Records after filtering: {len(df_arrived):,}")
    print(f"   Date range: {df['booking_date'].min()} to {df['booking_date'].max()}")
    
    # Step 2: Calculate time series metrics
    print("\n2. Calculating monthly metrics...")
    monthly_metrics = calculate_time_series_metrics(df_arrived)
    
    # Filter by minimum bookings threshold
    if config.min_monthly_bookings > 0:
        before_filter = len(monthly_metrics)
        monthly_metrics = monthly_metrics[monthly_metrics['bookings'] >= config.min_monthly_bookings]
        filtered_out = before_filter - len(monthly_metrics)
        print(f"   Filtered out {filtered_out} restaurant-months with < {config.min_monthly_bookings} bookings")
    
    print(f"   Monthly records: {len(monthly_metrics):,}")
    print(f"   Unique restaurants: {monthly_metrics['restaurant_id'].nunique()}")
    
    # Step 3: Calculate growth rates
    print("\n3. Calculating growth rates...")
    monthly_metrics = calculate_growth_rates(monthly_metrics)
    
    # Step 4: Create restaurant summary with adjustable parameters
    print("\n4. Creating restaurant summary with custom parameters...")
    restaurant_summary = create_growth_summary_adjustable(monthly_metrics, config)
    
    # Step 5: Identify growth segments with adjustable parameters
    print("\n5. Identifying growth segments...")
    restaurant_summary = identify_growth_segments_adjustable(restaurant_summary, config)
    
    # Print diagnostics
    print("\n   DIAGNOSTIC INFORMATION:")
    print(f"   Median recent bookings: {restaurant_summary['recent_avg_bookings'].median():.1f}")
    print(f"   Median growth rate: {restaurant_summary['avg_growth_rate'].median():.1f}%")
    print(f"   Mean momentum score: {restaurant_summary['momentum_score'].mean():.1f}")
    
    print("\n   Segment Distribution:")
    for segment, count in restaurant_summary['segment'].value_counts().items():
        pct = count / len(restaurant_summary) * 100
        print(f"   {segment}: {count} ({pct:.1f}%)")
    
    # Step 6: Generate visualizations
    print("\n6. Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    
    print("   Creating growth matrix...")
    fig1 = plot_growth_matrix(restaurant_summary)
    fig1.savefig(f'{output_dir}/growth_momentum_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("   Creating trend analysis...")
    fig2 = plot_trend_analysis(monthly_metrics, restaurant_summary, top_n=10)
    fig2.savefig(f'{output_dir}/top_performers_trends.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Step 7: Generate recommendations
    print("\n7. Generating marketing recommendations...")
    recommendations = generate_marketing_recommendations(restaurant_summary)
    
    # Step 8: Create summary report
    print("\n8. Creating summary report...")
    report = create_summary_report(restaurant_summary, recommendations)
    
    # Save report
    with open(f'{output_dir}/growth_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save data files
    print("\n9. Saving data files...")
    restaurant_summary.to_csv(f'{output_dir}/restaurant_summary.csv', index=False)
    monthly_metrics.to_csv(f'{output_dir}/monthly_metrics.csv', index=False)
    
    # Save config used
    with open(f'{output_dir}/analysis_config.txt', 'w') as f:
        f.write("ANALYSIS CONFIGURATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Recent months: {config.recent_months}\n")
        f.write(f"Min monthly bookings filter: {config.min_monthly_bookings}\n")
        f.write(f"High volume threshold: {config.high_volume_threshold}\n")
        f.write(f"High growth threshold: {config.high_growth_threshold}%\n")
        f.write(f"Declining threshold: {config.declining_threshold}%\n")
        f.write(f"\nMomentum Score Weights:\n")
        f.write(f"  Volume: {config.weight_volume}\n")
        f.write(f"  Booking Growth: {config.weight_booking_growth}\n")
        f.write(f"  Revenue Growth: {config.weight_revenue_growth}\n")
        f.write(f"\nSegmentation: {'Absolute thresholds' if config.use_absolute_thresholds else 'Median-based'}\n")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"\nOutput files saved to: {output_dir}/")
    print("  - growth_momentum_matrix.png")
    print("  - top_performers_trends.png")
    print("  - growth_analysis_report.txt")
    print("  - restaurant_summary.csv")
    print("  - monthly_metrics.csv")
    print("  - analysis_config.txt")
    print("=" * 80)
    
    # Print executive summary
    print("\n" + report)
    
    return restaurant_summary, monthly_metrics, recommendations


# EXAMPLE USAGE WITH DIFFERENT CONFIGURATIONS

def get_conservative_config():
    """More forgiving thresholds for slower-growing markets"""
    config = GrowthAnalysisConfig()
    config.recent_months = 6  # Longer period
    config.min_monthly_bookings = 3  # Lower filter
    config.high_volume_threshold = 10  # Lower bar
    config.high_growth_threshold = 2  # Just 2% counts as growth
    config.weight_volume = 0.7  # Favor current volume over growth
    config.weight_booking_growth = 0.15
    config.weight_revenue_growth = 0.15
    return config


def get_aggressive_config():
    """Strict thresholds for fast-growing markets"""
    config = GrowthAnalysisConfig()
    config.recent_months = 3
    config.min_monthly_bookings = 20
    config.high_volume_threshold = 50
    config.high_growth_threshold = 15
    config.weight_volume = 0.3  # Favor growth over current size
    config.weight_booking_growth = 0.35
    config.weight_revenue_growth = 0.35
    return config


def get_balanced_config():
    """Balanced approach"""
    config = GrowthAnalysisConfig()
    config.recent_months = 4
    config.min_monthly_bookings = 10
    config.high_volume_threshold = 20
    config.high_growth_threshold = 5
    config.weight_volume = 0.5
    config.weight_booking_growth = 0.25
    config.weight_revenue_growth = 0.25
    return config


if __name__ == "__main__":
    print("\nADJUSTABLE GROWTH ANALYSIS")
    print("=" * 80)
    print("\nThis version allows you to tune the analysis to your specific data.")
    print("\nExample usage:")
    print("\n# Use a pre-configured setting:")
    print("config = get_conservative_config()  # More forgiving")
    print("# OR")
    print("config = get_aggressive_config()    # Strict criteria")
    print("# OR")
    print("config = get_balanced_config()      # Middle ground")
    print("\n# Run analysis with config:")
    print("restaurant_summary, monthly_metrics, recommendations = run_growth_analysis_adjustable(")
    print("    df,")
    print("    output_dir='./outputs',")
    print("    config=config")
    print(")")
    print("\n# OR create your own custom config:")
    print("config = GrowthAnalysisConfig()")
    print("config.high_volume_threshold = 15  # Your custom threshold")
    print("config.high_growth_threshold = 3   # Your custom threshold")
    print("config.weight_volume = 0.6         # Your custom weight")
    print("# ... etc")
