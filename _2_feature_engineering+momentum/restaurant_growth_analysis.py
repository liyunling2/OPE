"""
Restaurant Growth Momentum Analysis
Identifies restaurants with highest growth potential for marketing resource allocation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 10

def prepare_data(df, column_mapping=None):
    """
    Prepare the dataframe with necessary date columns and filters
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe
    column_mapping : dict, optional
        Dictionary mapping standard column names to your actual column names
        Example: {
            'booking_date': 'your_date_column',
            'revenue_baht': 'your_revenue_column',
            'name': 'your_name_column'
        }
    """
    # Apply column mapping if provided
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert date columns - handle YYYY-MM-DD format
    print(f"   Converting booking_date column...")
    print(f"   Sample value: {df['booking_date'].iloc[0]} (type: {type(df['booking_date'].iloc[0])})")
    
    # Check if already datetime
    if pd.api.types.is_datetime64_any_dtype(df['booking_date']):
        print("   ✓ booking_date is already datetime type")
    else:
        # Convert to string first to ensure consistency
        df['booking_date'] = df['booking_date'].astype(str)
        # Then parse - use errors='coerce' to handle any bad values
        df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce', format='%Y-%m-%d')
        print("   ✓ Converted to datetime")
    
    if 'created_at' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = df['created_at'].astype(str)
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    
    # Check for any dates that failed to parse
    if df['booking_date'].isna().any():
        invalid_count = df['booking_date'].isna().sum()
        print(f"   ⚠ Warning: {invalid_count} rows have invalid dates and will be removed")
        print(f"   Sample invalid values: {df[df['booking_date'].isna()]['booking_date'].head()}")
        df = df.dropna(subset=['booking_date'])
    
    print(f"   ✓ Date conversion successful. Date range: {df['booking_date'].min()} to {df['booking_date'].max()}")
    
    # Extract time periods
    df['year_month'] = df['booking_date'].dt.to_period('M')
    df['year_week'] = df['booking_date'].dt.to_period('W')
    df['month'] = df['booking_date'].dt.month
    df['year'] = df['booking_date'].dt.year
    
    # Filter out outliers and invalid data if needed
    if 'is_outlier' in df.columns:
        df = df[df['is_outlier'] == False].copy()
    
    # Filter out no-shows for revenue analysis
    if 'no_show' in df.columns:
        df_arrived = df[df['no_show'] == False].copy()
    else:
        df_arrived = df.copy()
    
    return df, df_arrived


def calculate_time_series_metrics(df_arrived):
    """
    Calculate monthly and weekly metrics per restaurant
    """
    # Monthly metrics
    monthly_metrics = df_arrived.groupby(['restaurant_id', 'name', 'year_month']).agg({
        'id': 'count',  # total bookings
        'revenue_baht': 'sum',  # total revenue
        'total_guests': 'sum',  # total guests
        'adult': 'sum',  # adult guests
        'kids': 'sum'  # child guests
    }).reset_index()
    
    monthly_metrics.columns = ['restaurant_id', 'name', 'period', 'bookings', 
                               'revenue', 'total_guests', 'adults', 'kids']
    
    # Calculate derived metrics
    monthly_metrics['avg_party_size'] = monthly_metrics['total_guests'] / monthly_metrics['bookings']
    monthly_metrics['revenue_per_booking'] = monthly_metrics['revenue'] / monthly_metrics['bookings']
    monthly_metrics['revenue_per_guest'] = monthly_metrics['revenue'] / monthly_metrics['total_guests']
    
    return monthly_metrics


def calculate_growth_rates(monthly_metrics):
    """
    Calculate growth rates and momentum indicators
    """
    # Sort by restaurant and period
    monthly_metrics = monthly_metrics.sort_values(['restaurant_id', 'period'])
    
    # Calculate month-over-month growth rates
    monthly_metrics['bookings_mom_growth'] = monthly_metrics.groupby('restaurant_id')['bookings'].pct_change() * 100
    monthly_metrics['revenue_mom_growth'] = monthly_metrics.groupby('restaurant_id')['revenue'].pct_change() * 100
    
    # Calculate 3-month rolling average growth
    monthly_metrics['bookings_3m_avg_growth'] = monthly_metrics.groupby('restaurant_id')['bookings_mom_growth'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    monthly_metrics['revenue_3m_avg_growth'] = monthly_metrics.groupby('restaurant_id')['revenue_mom_growth'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    return monthly_metrics


def identify_growth_segments(restaurant_summary):
    """
    Classify restaurants into growth segments based on current performance and growth trajectory
    """
    # Get medians for segmentation
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
        elif growth_rate > -10:
            return 'Stable/Slight Decline'
        else:
            return 'Declining'
    
    restaurant_summary['trend'] = restaurant_summary['avg_growth_rate'].apply(classify_trend)
    
    return restaurant_summary


def create_growth_summary(monthly_metrics, recent_months=3):
    """
    Create summary statistics for each restaurant focusing on recent performance
    """
    # Get the most recent N months of data
    max_period = monthly_metrics['period'].max()
    recent_data = monthly_metrics[monthly_metrics['period'] >= max_period - recent_months]
    
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
    
    # Calculate momentum score (weighted combination of growth and volume)
    restaurant_summary['momentum_score'] = (
        (restaurant_summary['avg_growth_rate'] * 0.4) +  # 40% weight on booking growth
        (restaurant_summary['revenue_growth_rate'] * 0.4) +  # 40% weight on revenue growth
        (restaurant_summary['recent_avg_bookings'] / restaurant_summary['recent_avg_bookings'].max() * 20)  # 20% weight on current volume
    )
    
    return restaurant_summary


def plot_growth_matrix(restaurant_summary, figsize=(16, 10)):
    """
    Create the main growth momentum visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Restaurant Growth Momentum Analysis - Marketing Investment Matrix', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color mapping for segments
    segment_colors = {
        'Rising Stars': '#2ecc71',  # Green
        'Emerging Opportunities': '#f39c12',  # Orange
        'Established Players': '#3498db',  # Blue
        'Needs Attention': '#e74c3c'  # Red
    }
    
    # Plot 1: Growth Rate vs Current Volume (Main Investment Matrix)
    ax1 = axes[0, 0]
    
    for segment in restaurant_summary['segment'].unique():
        segment_data = restaurant_summary[restaurant_summary['segment'] == segment]
        scatter = ax1.scatter(
            segment_data['recent_avg_bookings'],
            segment_data['avg_growth_rate'],
            s=segment_data['recent_avg_revenue'] / 100,  # Size by revenue
            c=segment_colors[segment],
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
            label=segment
        )
    
    # Add median lines
    median_bookings = restaurant_summary['recent_avg_bookings'].median()
    median_growth = restaurant_summary['avg_growth_rate'].median()
    ax1.axhline(y=median_growth, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axvline(x=median_bookings, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add quadrant labels
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    
    ax1.text(xlim[1] * 0.75, ylim[1] * 0.85, 'INVEST HEAVILY\n(High Volume + High Growth)',
             ha='center', va='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax1.text(xlim[0] + (xlim[1] - xlim[0]) * 0.25, ylim[1] * 0.85, 'SCALE UP\n(Low Volume + High Growth)',
             ha='center', va='top', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    ax1.text(xlim[1] * 0.75, ylim[0] + (ylim[1] - ylim[0]) * 0.15, 'OPTIMIZE\n(High Volume + Low Growth)',
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax1.text(xlim[0] + (xlim[1] - xlim[0]) * 0.25, ylim[0] + (ylim[1] - ylim[0]) * 0.15, 
             'RECONSIDER\n(Low Volume + Low Growth)',
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax1.set_xlabel('Recent Average Monthly Bookings', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Booking Growth Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Growth Rate vs Current Volume\n(Size = Recent Revenue)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Revenue Growth vs Booking Growth
    ax2 = axes[0, 1]
    
    for segment in restaurant_summary['segment'].unique():
        segment_data = restaurant_summary[restaurant_summary['segment'] == segment]
        ax2.scatter(
            segment_data['avg_growth_rate'],
            segment_data['revenue_growth_rate'],
            s=segment_data['recent_avg_bookings'] * 3,
            c=segment_colors[segment],
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
            label=segment
        )
    
    # Add diagonal line (equal growth)
    lims = [
        np.min([ax2.get_xlim(), ax2.get_ylim()]),
        np.max([ax2.get_xlim(), ax2.get_ylim()]),
    ]
    ax2.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='Equal Growth Line')
    
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Booking Growth Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Revenue Growth Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Revenue Growth vs Booking Growth\n(Size = Current Bookings)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Momentum Score Ranking
    ax3 = axes[1, 0]
    
    top_momentum = restaurant_summary.nlargest(15, 'momentum_score').sort_values('momentum_score')
    
    colors_list = [segment_colors[seg] for seg in top_momentum['segment']]
    
    bars = ax3.barh(range(len(top_momentum)), top_momentum['momentum_score'], color=colors_list, alpha=0.7, edgecolor='black')
    ax3.set_yticks(range(len(top_momentum)))
    ax3.set_yticklabels(top_momentum['name'], fontsize=9)
    ax3.set_xlabel('Momentum Score', fontsize=11, fontweight='bold')
    ax3.set_title('Top 15 Restaurants by Momentum Score\n(Combined Growth + Volume)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_momentum['momentum_score'])):
        ax3.text(value, i, f' {value:.1f}', va='center', fontsize=8, fontweight='bold')
    
    # Plot 4: Segment Distribution
    ax4 = axes[1, 1]
    
    segment_counts = restaurant_summary['segment'].value_counts()
    colors_pie = [segment_colors[seg] for seg in segment_counts.index]
    
    wedges, texts, autotexts = ax4.pie(
        segment_counts.values,
        labels=segment_counts.index,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    ax4.set_title('Restaurant Distribution by Segment', fontsize=12, fontweight='bold')
    
    # Add count labels
    for i, (label, count) in enumerate(zip(segment_counts.index, segment_counts.values)):
        texts[i].set_text(f'{label}\n(n={count})')
    
    plt.tight_layout()
    return fig


def plot_trend_analysis(monthly_metrics, restaurant_summary, top_n=10):
    """
    Create time series plots for top growth restaurants
    """
    # Get top growth restaurants
    top_growth = restaurant_summary.nlargest(top_n, 'momentum_score')
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f'Top {top_n} High-Momentum Restaurants - Trend Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Filter monthly data for top restaurants
    top_restaurants_data = monthly_metrics[monthly_metrics['restaurant_id'].isin(top_growth['restaurant_id'])]
    
    # Plot 1: Booking trends
    ax1 = axes[0]
    for restaurant_id in top_growth['restaurant_id']:
        restaurant_data = top_restaurants_data[top_restaurants_data['restaurant_id'] == restaurant_id]
        restaurant_name = restaurant_data['name'].iloc[0]
        
        # Convert period to timestamp for plotting
        periods = restaurant_data['period'].astype(str)
        
        ax1.plot(range(len(periods)), restaurant_data['bookings'], 
                marker='o', label=restaurant_name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Bookings', fontsize=11, fontweight='bold')
    ax1.set_title('Monthly Booking Trends', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Revenue trends
    ax2 = axes[1]
    for restaurant_id in top_growth['restaurant_id']:
        restaurant_data = top_restaurants_data[top_restaurants_data['restaurant_id'] == restaurant_id]
        restaurant_name = restaurant_data['name'].iloc[0]
        
        periods = restaurant_data['period'].astype(str)
        
        ax2.plot(range(len(periods)), restaurant_data['revenue'] / 1000,  # Convert to thousands
                marker='o', label=restaurant_name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Revenue (Thousands Baht)', fontsize=11, fontweight='bold')
    ax2.set_title('Monthly Revenue Trends', fontsize=12, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_marketing_recommendations(restaurant_summary):
    """
    Generate actionable marketing recommendations based on segments
    """
    recommendations = []
    
    # Rising Stars - highest priority
    rising_stars = restaurant_summary[restaurant_summary['segment'] == 'Rising Stars'].nlargest(10, 'momentum_score')
    if len(rising_stars) > 0:
        recommendations.append({
            'Priority': 'HIGHEST',
            'Segment': 'Rising Stars',
            'Count': len(rising_stars),
            'Restaurants': rising_stars['name'].tolist()[:5],  # Top 5
            'Recommendation': 'Invest heavily in these restaurants. They have high volume AND high growth. Increase ad spend, create exclusive promotions, and feature prominently in marketing materials.',
            'Expected ROI': 'Very High',
            'Suggested Actions': [
                'Increase marketing budget allocation by 30-50%',
                'Create loyalty programs to retain growing customer base',
                'Feature in premium ad placements',
                'Develop case studies for other restaurants'
            ]
        })
    
    # Emerging Opportunities
    emerging = restaurant_summary[restaurant_summary['segment'] == 'Emerging Opportunities'].nlargest(10, 'momentum_score')
    if len(emerging) > 0:
        recommendations.append({
            'Priority': 'HIGH',
            'Segment': 'Emerging Opportunities',
            'Count': len(emerging),
            'Restaurants': emerging['name'].tolist()[:5],
            'Recommendation': 'Scale up marketing for these high-growth restaurants to convert momentum into volume. They show strong growth but need visibility.',
            'Expected ROI': 'High',
            'Suggested Actions': [
                'Run targeted acquisition campaigns',
                'Offer first-time booking discounts',
                'Boost social media presence',
                'Partner with local influencers'
            ]
        })
    
    # Established Players
    established = restaurant_summary[restaurant_summary['segment'] == 'Established Players'].nlargest(5, 'total_bookings')
    if len(established) > 0:
        recommendations.append({
            'Priority': 'MEDIUM',
            'Segment': 'Established Players',
            'Count': len(established),
            'Restaurants': established['name'].tolist()[:5],
            'Recommendation': 'Maintain current marketing levels. Focus on retention and optimization rather than aggressive growth campaigns.',
            'Expected ROI': 'Medium',
            'Suggested Actions': [
                'Implement retention campaigns',
                'Optimize booking conversion rates',
                'Test new menu items or experiences',
                'Gather customer feedback for improvements'
            ]
        })
    
    # Needs Attention
    needs_attention = restaurant_summary[restaurant_summary['segment'] == 'Needs Attention']
    if len(needs_attention) > 0:
        recommendations.append({
            'Priority': 'LOW',
            'Segment': 'Needs Attention',
            'Count': len(needs_attention),
            'Restaurants': needs_attention['name'].tolist()[:5] if len(needs_attention) >= 5 else needs_attention['name'].tolist(),
            'Recommendation': 'Minimize marketing spend until performance improves. Investigate root causes (service quality, pricing, location, etc.).',
            'Expected ROI': 'Low',
            'Suggested Actions': [
                'Conduct operational audit',
                'Reduce marketing budget temporarily',
                'Focus on fixing fundamental issues',
                'Consider menu/pricing adjustments'
            ]
        })
    
    return recommendations


def create_summary_report(restaurant_summary, recommendations):
    """
    Create a text summary report
    """
    report = []
    report.append("=" * 80)
    report.append("RESTAURANT GROWTH MOMENTUM ANALYSIS - EXECUTIVE SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total Restaurants Analyzed: {len(restaurant_summary)}")
    report.append(f"Average Growth Rate: {restaurant_summary['avg_growth_rate'].mean():.2f}%")
    report.append(f"Average Revenue Growth Rate: {restaurant_summary['revenue_growth_rate'].mean():.2f}%")
    report.append("")
    
    # Segment breakdown
    report.append("SEGMENT BREAKDOWN")
    report.append("-" * 80)
    for segment in restaurant_summary['segment'].value_counts().index:
        count = len(restaurant_summary[restaurant_summary['segment'] == segment])
        pct = count / len(restaurant_summary) * 100
        report.append(f"{segment}: {count} restaurants ({pct:.1f}%)")
    report.append("")
    
    # Top performers
    report.append("TOP 10 HIGH-MOMENTUM RESTAURANTS")
    report.append("-" * 80)
    top_10 = restaurant_summary.nlargest(10, 'momentum_score')
    for idx, row in top_10.iterrows():
        report.append(f"{row['name']}")
        report.append(f"  Segment: {row['segment']}")
        report.append(f"  Momentum Score: {row['momentum_score']:.2f}")
        report.append(f"  Booking Growth: {row['avg_growth_rate']:.2f}%")
        report.append(f"  Revenue Growth: {row['revenue_growth_rate']:.2f}%")
        report.append(f"  Recent Avg Bookings/Month: {row['recent_avg_bookings']:.0f}")
        report.append("")
    
    # Marketing recommendations
    report.append("=" * 80)
    report.append("MARKETING RECOMMENDATIONS")
    report.append("=" * 80)
    report.append("")
    
    for rec in recommendations:
        report.append(f"PRIORITY: {rec['Priority']} - {rec['Segment']}")
        report.append("-" * 80)
        report.append(f"Number of Restaurants: {rec['Count']}")
        report.append(f"\nTop Restaurants in Segment:")
        for restaurant in rec['Restaurants']:
            report.append(f"  • {restaurant}")
        report.append(f"\nRecommendation: {rec['Recommendation']}")
        report.append(f"\nExpected ROI: {rec['Expected ROI']}")
        report.append(f"\nSuggested Actions:")
        for action in rec['Suggested Actions']:
            report.append(f"  • {action}")
        report.append("")
        report.append("")
    
    return "\n".join(report)


# Main execution function
def run_growth_analysis(df, output_dir='/mnt/user-data/outputs', column_mapping=None):
    """
    Run the complete growth momentum analysis
    
    Parameters:
    -----------
    df : DataFrame
        The input dataframe with booking data
    output_dir : str
        Directory to save output files
    column_mapping : dict, optional
        Dictionary mapping standard column names to your actual column names
        
        Standard names (keys) and what they represent:
        - 'booking_date': Date of the booking
        - 'restaurant_id': Unique restaurant identifier
        - 'name': Restaurant name
        - 'revenue_baht': Revenue amount
        - 'total_guests': Number of guests
        - 'adult': Number of adult guests
        - 'kids': Number of child guests
        - 'id': Booking ID
        - 'no_show': Whether customer was a no-show
        - 'is_outlier': Outlier flag
        - 'created_at': When booking was created
        
        Example:
        column_mapping = {
            'booking_date': 'date',
            'revenue_baht': 'revenue',
            'name': 'restaurant_name'
        }
    """
    import os
    
    print("Starting Restaurant Growth Momentum Analysis...")
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
    print(f"   Monthly records: {len(monthly_metrics):,}")
    print(f"   Unique restaurants: {monthly_metrics['restaurant_id'].nunique()}")
    
    # Step 3: Calculate growth rates
    print("\n3. Calculating growth rates...")
    monthly_metrics = calculate_growth_rates(monthly_metrics)
    
    # Step 4: Create restaurant summary
    print("\n4. Creating restaurant summary...")
    restaurant_summary = create_growth_summary(monthly_metrics, recent_months=3)
    
    # Step 5: Identify growth segments
    print("\n5. Identifying growth segments...")
    restaurant_summary = identify_growth_segments(restaurant_summary)
    
    # Print segment counts
    print("\n   Segment Distribution:")
    for segment, count in restaurant_summary['segment'].value_counts().items():
        print(f"   {segment}: {count}")
    
    # Step 6: Generate visualizations
    print("\n6. Generating visualizations...")
    
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
    with open(f'{output_dir}/growth_analysis_report.txt', 'w') as f:
        f.write(report)
    
    # Save data files
    print("\n9. Saving data files...")
    restaurant_summary.to_csv(f'{output_dir}/restaurant_summary.csv', index=False)
    monthly_metrics.to_csv(f'{output_dir}/monthly_metrics.csv', index=False)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"\nOutput files saved to: {output_dir}/")
    print("  - growth_momentum_matrix.png")
    print("  - top_performers_trends.png")
    print("  - growth_analysis_report.txt")
    print("  - restaurant_summary.csv")
    print("  - monthly_metrics.csv")
    print("=" * 80)
    
    # Print executive summary
    print("\n" + report)
    
    return restaurant_summary, monthly_metrics, recommendations


# Example usage (uncomment when you have the dataframe loaded):
# Assuming your dataframe is called 'df' and has the columns mentioned
# restaurant_summary, monthly_metrics, recommendations = run_growth_analysis(df)
