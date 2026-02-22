## Clustering info
- Add the following into requirements:
    - scikit-learn
    - umap-learn
    - plotly
- The following files will be created upon running the entire clustering.ipynb notebook:
    - - **`restaurant_clusters_multi_theme.html`** - Interactive visualization showing restaurant segmentation by customer perception themes. Each dot represents a restaurant, sized by number of posts, colored by primary theme. Hover data reveals all assigned themes and percentages.

    - **`restaurants_with_multi_themes.csv`** - Restaurant-level output with primary theme, all assigned themes, number of posts, and primary theme percentage for each restaurant.

    - **`restaurant_theme_details.csv`** - Detailed breakdown showing the percentage distribution of each theme for every restaurant, useful for understanding restaurants with multiple theme associations.

# What is UMAP (Uniform Manifold Axpproximation & Projection) ?
- Reduces high-dimensional data into 2D while keeping similar things close together
- Preserves local structure
- Shows natural grouping

# X-axis is the Semantic (UMAP Component 1)
- How similar restaurant are based on reviews

Left (X: -2 to 4)              →              Right (X: 10 to 14)
━━━━━━━━━━━━━━━━━━━━━━━━━
Food-focused                                  Experience-focused
Casual/everyday                               Special occasion
Menu variety                                  Atmosphere/ambiance
"What you eat"                                "Where/how you dine"

# y-axis is the Theme variation (UMAP Component 2)
- How strongly a restaurant is associated with a single theme vs multiple themes

Top (Y: 4)
   ↑
   │  Mixed/casual (cafes, service-focused)
   │
   │  Mid-range (general restaurants)
   │
   │  Specialized (bars, fine dining)
   ↓
Bottom (Y: -10)

## Overview of the Cluster Characteristics:

# 🟡 Yellow (Various Menus & Quality Food) - DOMINANT
Position: Entire left side
Size: Massive - probably 40-50% of your restaurants
Spread: Very wide (Y: -8 to +4)
Interpretation:
This is your largest segment
Shows high diversity within "food quality" restaurants
Vertical spread suggests sub-themes:
Top yellow (Y: 0 to 4) = Cafe/casual dining
Middle yellow (Y: -4 to 0) = Family restaurants
Bottom yellow (Y: -8 to -4) = Fine dining

# 🟢 Teal (Good Location) - CONCENTRATED
Position: Center-right, middle-bottom
Size: Large but compact
Clustering: Tightest cluster - most homogeneous
Interpretation:
Reviews consistently mention location/convenience
Strong identity as "easy to access" restaurants
Overlaps slightly with:
Yellow (bottom-left) = Food + location
Blue (right) = View + location

# 🔵 Blue (Restaurant with Good View) - RIGHT CLUSTER
Position: Far right side
Size: Large, elongated vertically
Distinction: Most separated from others
Interpretation:
Clear differentiation from other themes
Reviews heavily feature view/scenery/ambiance
Vertical spread suggests different view types:
Top blue (Y: 2 to 4) = Rooftop/skyline views
Middle blue (Y: -2 to 2) = Waterfront/river views
Bottom blue (Y: -6 to -2) = Garden/nature views

# 🔴 Red (Extensive Alcohol Beverage Options) - BOTTOM CLUSTER
Position: Bottom center, isolated
Size: Medium, very distinct
Separation: Clear gap from food-focused clusters
Interpretation:
Strong bar/nightlife identity
Reviews mention cocktails/drinks/bar
Least overlap with other themes (most unique)
Some mixing with:
Teal (Good Location) = Bars in convenient locations
Blue (Good View) = Rooftop bars with views

# 🟣 Purple (Attentive Service but Pricey) - SCATTERED
Position: Top-center, mixed throughout
Size: Smallest distinct region
Pattern: Most scattered - no tight cluster
Interpretation:
"Service/price" is a cross-cutting theme
Not a primary differentiator
Appears alongside other themes:
Purple in yellow area = Good food + pricey
Purple in blue area = Nice view + pricey
Purple in teal area = Convenient + pricey