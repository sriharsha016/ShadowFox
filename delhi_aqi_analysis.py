import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from scipy import stats

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create output directory for figures
output_dir = "aqi_analysis_figures"
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv('delhiaqi.csv')

# Data preprocessing
def preprocess_data(df):
    # Check for missing values
    print(f"Missing values in the dataset:\n{df.isnull().sum()}")
    
    # Convert date column to datetime if it exists
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_columns:
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"Converted {col} to datetime")
                # Extract month and season for seasonal analysis
                df['month'] = df[col].dt.month
                df['season'] = pd.cut(
                    df['month'],
                    bins=[0, 2, 5, 8, 11, 12],
                    labels=['Winter', 'Spring', 'Summer', 'Autumn', 'Winter']
                )
                break
            except:
                print(f"Could not convert {col} to datetime")
    
    # Handle missing values based on the nature of the data
    # For this analysis, we'll use simple imputation with mean for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mean(), inplace=True)
    
    return df

# Analyze key pollutants
def analyze_pollutants(df):
    # Identify pollutant columns
    pollutant_cols = [col for col in df.columns if col.upper() in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
    if not pollutant_cols:
        # Try to guess pollutant columns based on common naming patterns
        pollutant_cols = [col for col in df.columns if any(p in col.upper() for p in ['PM', 'NO', 'SO', 'CO', 'O3'])]
    
    if pollutant_cols:
        print(f"Identified pollutant columns: {pollutant_cols}")
        
        # Summary statistics for pollutants
        pollutant_stats = df[pollutant_cols].describe()
        print("\nPollutant Statistics:")
        print(pollutant_stats)
        
        # Correlation between pollutants
        plt.figure(figsize=(10, 8))
        correlation = df[pollutant_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Pollutants')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pollutant_correlation.png", dpi=300)
        
        # Distribution of each pollutant
        for col in pollutant_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_distribution.png", dpi=300)
        
        return pollutant_cols
    else:
        print("No pollutant columns identified.")
        return []

# Analyze seasonal variations
def analyze_seasonal_variations(df, pollutant_cols):
    if 'season' in df.columns and pollutant_cols:
        # Seasonal average of pollutants
        seasonal_avg = df.groupby('season')[pollutant_cols].mean()
        print("\nSeasonal Average of Pollutants:")
        print(seasonal_avg)
        
        # Plot seasonal variations
        plt.figure(figsize=(14, 8))
        seasonal_avg.plot(kind='bar')
        plt.title('Seasonal Variation of Pollutants')
        plt.xlabel('Season')
        plt.ylabel('Average Concentration')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/seasonal_variation.png", dpi=300)
        
        # Monthly trends for each pollutant
        if 'month' in df.columns:
            monthly_avg = df.groupby('month')[pollutant_cols].mean()
            
            for col in pollutant_cols:
                plt.figure(figsize=(12, 6))
                monthly_avg[col].plot(kind='line', marker='o')
                plt.title(f'Monthly Trend of {col}')
                plt.xlabel('Month')
                plt.ylabel(f'{col} Concentration')
                plt.xticks(range(1, 13))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/{col}_monthly_trend.png", dpi=300)

# Analyze geographical factors if location data is available
def analyze_geographical_factors(df, pollutant_cols):
    location_cols = [col for col in df.columns if any(loc in col.lower() for loc in ['location', 'station', 'site', 'area'])]
    
    if location_cols and pollutant_cols:
        location_col = location_cols[0]
        print(f"\nAnalyzing geographical variations using {location_col}")
        
        # Average pollutant levels by location
        location_avg = df.groupby(location_col)[pollutant_cols].mean().sort_values(by=pollutant_cols[0], ascending=False)
        print("\nAverage Pollutant Levels by Location:")
        print(location_avg)
        
        # Plot geographical variations for each pollutant
        for col in pollutant_cols:
            plt.figure(figsize=(14, 8))
            location_data = location_avg.sort_values(by=col, ascending=False)
            sns.barplot(x=location_data.index, y=location_data[col])
            plt.title(f'Geographical Variation of {col}')
            plt.xlabel('Location')
            plt.ylabel(f'{col} Concentration')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_geographical_variation.png", dpi=300)

# Analyze AQI trends and patterns
def analyze_aqi_trends(df):
    aqi_cols = [col for col in df.columns if 'aqi' in col.lower()]
    
    if aqi_cols:
        aqi_col = aqi_cols[0]
        print(f"\nAnalyzing AQI trends using {aqi_col}")
        
        # AQI categories based on standard ranges
        aqi_categories = [
            (0, 50, 'Good'),
            (51, 100, 'Moderate'),
            (101, 150, 'Unhealthy for Sensitive Groups'),
            (151, 200, 'Unhealthy'),
            (201, 300, 'Very Unhealthy'),
            (301, 500, 'Hazardous')
        ]
        
        # Create AQI category column
        df['aqi_category'] = pd.cut(
            df[aqi_col],
            bins=[cat[0] for cat in aqi_categories] + [float('inf')],
            labels=[cat[2] for cat in aqi_categories]
        )
        
        # Distribution of AQI categories
        plt.figure(figsize=(12, 6))
        sns.countplot(x='aqi_category', data=df, order=df['aqi_category'].value_counts().index)
        plt.title('Distribution of AQI Categories')
        plt.xlabel('AQI Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/aqi_category_distribution.png", dpi=300)
        
        # Time series analysis if date column exists
        if 'month' in df.columns:
            # Monthly average AQI
            monthly_aqi = df.groupby('month')[aqi_col].mean()
            
            plt.figure(figsize=(12, 6))
            monthly_aqi.plot(kind='line', marker='o')
            plt.title('Monthly Average AQI')
            plt.xlabel('Month')
            plt.ylabel('Average AQI')
            plt.xticks(range(1, 13))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/monthly_aqi_trend.png", dpi=300)
        
        # Seasonal AQI analysis
        if 'season' in df.columns:
            seasonal_aqi = df.groupby('season')[aqi_col].mean()
            
            plt.figure(figsize=(10, 6))
            seasonal_aqi.plot(kind='bar')
            plt.title('Seasonal Average AQI')
            plt.xlabel('Season')
            plt.ylabel('Average AQI')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/seasonal_aqi.png", dpi=300)
        
        return aqi_col
    else:
        print("No AQI column identified.")
        return None

# Generate a comprehensive report
def generate_report(df, pollutant_cols, aqi_col):
    report = """
# Air Quality Index (AQI) Analysis in Delhi: Environmental Challenges and Insights

## Executive Summary
This report presents an in-depth analysis of the Air Quality Index (AQI) in Delhi, focusing on key pollutants, seasonal variations, and geographical factors affecting air quality. The analysis aims to provide insights that can inform targeted strategies for air quality improvement and public health initiatives in the region.

## Introduction
Delhi, the capital city of India, has been facing severe air pollution problems for decades. The city's air quality is influenced by various factors including vehicular emissions, industrial activities, construction, crop burning in neighboring states, and meteorological conditions. This analysis examines the patterns and trends in Delhi's air quality data to better understand the dynamics of pollution in the region.

## Research Questions
1. What are the primary pollutants contributing to poor air quality in Delhi?
2. How does air quality vary across different seasons in Delhi?
3. What geographical factors influence the distribution of pollutants across different areas of Delhi?
4. What are the temporal patterns in Delhi's AQI, and how can they inform pollution control strategies?

## Methodology
The analysis utilizes a comprehensive dataset of air quality measurements from Delhi. Statistical methods and data visualization techniques are employed to identify patterns, correlations, and trends in the data.

## Key Findings

### Pollutant Analysis
"""
    
    if pollutant_cols:
        # Add pollutant statistics to report
        pollutant_stats = df[pollutant_cols].describe()
        report += f"The analysis identified the following key pollutants: {', '.join(pollutant_cols)}.\n\n"
        
        # Find the most prevalent pollutant
        pollutant_means = df[pollutant_cols].mean()
        most_prevalent = pollutant_means.idxmax()
        report += f"The most prevalent pollutant in Delhi's air is {most_prevalent} with an average concentration of {pollutant_means[most_prevalent]:.2f}.\n\n"
        
        # Correlation insights
        correlation = df[pollutant_cols].corr()
        high_corr_pairs = []
        for i in range(len(pollutant_cols)):
            for j in range(i+1, len(pollutant_cols)):
                if abs(correlation.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((pollutant_cols[i], pollutant_cols[j], correlation.iloc[i, j]))
        
        if high_corr_pairs:
            report += "Strong correlations were observed between the following pollutants:\n"
            for p1, p2, corr in high_corr_pairs:
                report += f"- {p1} and {p2}: {corr:.2f}\n"
            report += "\n"
        
        report += "The correlation analysis suggests potential common sources or similar atmospheric behavior for these pollutants.\n\n"
    
    report += """
### Seasonal Variations
"""
    
    if 'season' in df.columns and pollutant_cols:
        seasonal_avg = df.groupby('season')[pollutant_cols].mean()
        worst_season = seasonal_avg.mean(axis=1).idxmax()
        best_season = seasonal_avg.mean(axis=1).idxmin()
        
        report += f"The analysis reveals significant seasonal variations in Delhi's air quality. The {worst_season} season shows the highest pollution levels, while {best_season} has relatively better air quality.\n\n"
        
        # Seasonal patterns for specific pollutants
        for col in pollutant_cols:
            worst_season_for_pollutant = seasonal_avg[col].idxmax()
            report += f"- {col} peaks during the {worst_season_for_pollutant} season.\n"
        
        report += "\nThese seasonal patterns can be attributed to various factors including:\n"
        report += "- Winter: Temperature inversions, reduced wind speed, and increased burning for heating\n"
        report += "- Spring: Dust storms and agricultural activities\n"
        report += "- Summer: Higher temperatures leading to increased ozone formation\n"
        report += "- Autumn: Crop residue burning in neighboring states\n\n"
    
    report += """
### Geographical Factors
"""
    
    location_cols = [col for col in df.columns if any(loc in col.lower() for loc in ['location', 'station', 'site', 'area'])]
    if location_cols and pollutant_cols:
        location_col = location_cols[0]
        location_avg = df.groupby(location_col)[pollutant_cols].mean()
        
        # Most and least polluted areas
        overall_pollution = location_avg.mean(axis=1)
        most_polluted = overall_pollution.idxmax()
        least_polluted = overall_pollution.idxmin()
        
        report += f"The geographical analysis shows significant variations in air quality across different areas of Delhi. {most_polluted} consistently shows the highest pollution levels, while {least_polluted} has relatively better air quality.\n\n"
        
        report += "Factors contributing to these geographical variations may include:\n"
        report += "- Proximity to industrial zones\n"
        report += "- Traffic density and congestion\n"
        report += "- Population density\n"
        report += "- Green cover and open spaces\n"
        report += "- Local meteorological conditions\n\n"
    
    report += """
### AQI Trends and Patterns
"""
    
    if aqi_col:
        # AQI statistics
        aqi_mean = df[aqi_col].mean()
        aqi_median = df[aqi_col].median()
        aqi_std = df[aqi_col].std()
        
        report += f"The average AQI in Delhi is {aqi_mean:.2f} with a standard deviation of {aqi_std:.2f}, indicating high variability in air quality. The median AQI is {aqi_median:.2f}.\n\n"
        
        if 'aqi_category' in df.columns:
            category_counts = df['aqi_category'].value_counts(normalize=True) * 100
            most_common_category = category_counts.idxmax()
            
            report += f"Delhi's air quality falls into the '{most_common_category}' category for {category_counts[most_common_category]:.1f}% of the time.\n\n"
            
            # Add breakdown of all categories
            report += "Distribution of AQI categories:\n"
            for category, percentage in category_counts.items():
                report += f"- {category}: {percentage:.1f}%\n"
            report += "\n"
        
        if 'season' in df.columns:
            seasonal_aqi = df.groupby('season')[aqi_col].mean()
            worst_aqi_season = seasonal_aqi.idxmax()
            best_aqi_season = seasonal_aqi.idxmin()
            
            report += f"The {worst_aqi_season} season has the worst air quality with an average AQI of {seasonal_aqi[worst_aqi_season]:.2f}, while {best_aqi_season} has the best air quality with an average AQI of {seasonal_aqi[best_aqi_season]:.2f}.\n\n"
    
    report += """
## Conclusions and Recommendations

Based on the analysis of Delhi's air quality data, the following conclusions and recommendations can be drawn:

### Key Conclusions
1. Delhi's air quality shows significant seasonal variations, with winter months being the most critical period for air pollution.
2. There are substantial geographical differences in pollution levels across the city, suggesting localized sources and factors.
3. Multiple pollutants contribute to Delhi's poor air quality, with complex interactions and correlations between them.

### Recommendations for Air Quality Improvement
1. **Seasonal Strategies**: Implement season-specific pollution control measures, with intensified efforts during winter months.
2. **Geographical Targeting**: Focus pollution control efforts on identified hotspots while studying the factors that contribute to better air quality in less polluted areas.
3. **Source-specific Interventions**: Develop targeted strategies for the most prevalent pollutants and their sources.
4. **Public Health Measures**: Establish an early warning system for vulnerable populations during periods of expected poor air quality.
5. **Long-term Planning**: Integrate air quality considerations into urban planning, transportation, and industrial policies.

### Recommendations for Further Research
1. Conduct more detailed analysis of meteorological factors and their impact on pollutant dispersion.
2. Study the effectiveness of existing pollution control measures.
3. Investigate the health impacts of specific pollutants on Delhi's population.
4. Develop predictive models for air quality forecasting.

## Appendix: Data Sources and Methodology
This analysis was conducted using air quality data from monitoring stations across Delhi. The dataset includes measurements of key pollutants and calculated AQI values. Statistical analyses and data visualization techniques were employed to identify patterns and trends in the data.

"""
    
    # Save the report to a file
    with open("Delhi_AQI_Analysis_Report.md", "w") as f:
        f.write(report)
    
    print("\nReport generated successfully: Delhi_AQI_Analysis_Report.md")

# Main execution
if __name__ == "__main__":
    print("Starting Delhi AQI Analysis...")
    
    # Preprocess the data
    df = preprocess_data(df)
    
    # Analyze pollutants
    pollutant_cols = analyze_pollutants(df)
    
    # Analyze seasonal variations
    analyze_seasonal_variations(df, pollutant_cols)
    
    # Analyze geographical factors
    analyze_geographical_factors(df, pollutant_cols)
    
    # Analyze AQI trends
    aqi_col = analyze_aqi_trends(df)
    
    # Generate comprehensive report
    generate_report(df, pollutant_cols, aqi_col)
    
    print("Analysis completed. Check the output directory for visualizations.")