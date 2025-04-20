"""
# Superstore Data Analysis Notebook

This script performs an interactive analysis of the Superstore dataset,
with clear sections and step-by-step exploration.

"""

#%% [markdown]
"""
# 1. Setup and Data Loading
First, let's import the necessary libraries and load our dataset.

**Note:** This analysis will help us understand:
- Sales trends and patterns
- Customer behaviour and segmentation
- Product category performance
- Regional market analysis
- Key business insights and recommendations
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style for visualisations
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

#%% [markdown]
"""
# 2. Data Loading and Initial Exploration
Let's load the dataset and take a first look at its structure.

**Note:** We'll examine:
- Dataset dimensions
- Column types
- Sample data
- Basic statistics
"""

#%%
def load_data():
    """Load the Superstore dataset from CSV file."""
    try:
        df = pd.read_csv('Superstore.csv', encoding='latin1')
        print("Dataset loaded successfully!")
        print("\nDataset Shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())
        print("\nData Types:")
        print(df.dtypes)
        print("\nBasic Statistics:")
        print(df.describe())
        return df
    except FileNotFoundError:
        print("Error: Superstore.csv not found in the current directory.")
        print("Please ensure the Superstore.csv file is in the same directory as this script.")
        exit(1)

# Load the data
df = load_data()

#%% [markdown]
"""
# 3. Data Cleaning and Preprocessing
Let's clean and prepare our data for analysis.

**Note:** We'll:
- Convert date columns
- Extract time-based features
- Calculate derived metrics
- Create customer segments
"""

#%%
def clean_data(df):
    """Clean and preprocess the dataset."""
    print("\nCleaning and preprocessing data...")
    
    # Convert date columns
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date'])
    
    # Extract time-based features
    df['Order Year'] = df['Order Date'].dt.year
    df['Order Month'] = df['Order Date'].dt.month
    df['Order Day'] = df['Order Date'].dt.day
    df['Order Quarter'] = df['Order Date'].dt.quarter
    df['Order Day of Week'] = df['Order Date'].dt.dayofweek
    
    # Calculate shipping time
    df['Shipping Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Calculate profit margin
    df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100
    
    # Create customer segments
    df['Customer Segment'] = pd.cut(df['Sales'],
                                  bins=[0, 100, 500, 1000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    
    print("Data cleaning completed!")
    print("\nNew Features Created:")
    print("- Order Year, Month, Day, Quarter, Day of Week")
    print("- Shipping Time (days)")
    print("- Profit Margin (%)")
    print("- Customer Segments (Low, Medium, High, Very High)")
    
    return df

# Clean the data
df = clean_data(df)

#%% [markdown]
"""
# 4. Data Validation
Let's check for any data quality issues.

**Note:** We'll examine:
- Missing values
- Duplicate records
- Negative values in numeric columns
- Data consistency
"""

#%%
def validate_data(df):
    """Validate the dataset for data quality issues."""
    print("\nValidating data quality...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print("\nNumber of Duplicates:", duplicates)
    
    # Check for negative values in numeric columns
    negative_values = (df.select_dtypes(include=[np.number]) < 0).sum()
    print("\nNegative Values:")
    print(negative_values[negative_values > 0])
    
    # Check data consistency
    print("\nData Consistency Checks:")
    print(f"Date Range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    print(f"Number of Unique Customers: {df['Customer ID'].nunique()}")
    print(f"Number of Unique Products: {df['Product ID'].nunique()}")
    
    return {
        'missing_values': missing_values,
        'duplicates': duplicates,
        'negative_values': negative_values
    }

# Validate the data
validation_results = validate_data(df)

#%% [markdown]
"""
# 5. Sales Analysis
Let's analyse the sales performance and trends.

**Note:** We'll examine:
- Overall sales and profit metrics
- Monthly sales trends
- Seasonal patterns
- Growth rates
"""

#%%
def analyse_sales(df):
    """Analyse sales performance and trends."""
    print("\nAnalysing sales performance...")
    
    # Calculate overall metrics
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    profit_margin = (total_profit / total_sales) * 100
    
    print(f"\nOverall Sales Performance:")
    print(f"Total Sales: ${total_sales:,.2f}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Overall Profit Margin: {profit_margin:.2f}%")
    
    # Monthly sales trend
    monthly_sales = df.groupby(['Order Year', 'Order Month'])['Sales'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(
        monthly_sales['Order Year'].astype(str) + '-' + 
        monthly_sales['Order Month'].astype(str).str.zfill(2) + '-01'
    )
    
    # Create and show the plot
    plt.figure(figsize=(15, 6))
    sns.lineplot(x='Date', y='Sales', data=monthly_sales)
    plt.title('Monthly Sales Trend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()  # Show the plot
    
    # Save the plot
    plt.savefig('sales_trend.png')
    plt.close()
    
    print("\nSales trend visualisation saved as 'sales_trend.png'")
    
    # Calculate growth rates
    monthly_sales['Growth Rate'] = monthly_sales['Sales'].pct_change() * 100
    print("\nRecent Growth Rates:")
    print(monthly_sales[['Date', 'Sales', 'Growth Rate']].tail(6))
    
    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'profit_margin': profit_margin,
        'monthly_sales': monthly_sales
    }

# Analyse sales
sales_analysis = analyse_sales(df)

#%% [markdown]
"""
# 6. Category Analysis
Let's examine performance by product categories.

**Note:** We'll analyse:
- Sales by category
- Profit margins
- Quantity sold
- Category growth
"""

#%%
def analyse_categories(df):
    """Analyse performance by product categories."""
    print("\nAnalysing category performance...")
    
    # Calculate category statistics
    category_stats = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).round(2)
    
    # Calculate profit margins
    category_stats['Profit Margin'] = (category_stats['Profit'] / category_stats['Sales']) * 100
    
    print("\nCategory-wise Statistics:")
    print(category_stats)
    
    # Create and show the visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Sales by category
    sns.barplot(x=category_stats.index, y='Sales', data=category_stats, ax=axes[0, 0])
    axes[0, 0].set_title('Total Sales by Category')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Profit by category
    sns.barplot(x=category_stats.index, y='Profit', data=category_stats, ax=axes[0, 1])
    axes[0, 1].set_title('Total Profit by Category')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Quantity by category
    sns.barplot(x=category_stats.index, y='Quantity', data=category_stats, ax=axes[1, 0])
    axes[1, 0].set_title('Total Quantity by Category')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Profit margin by category
    sns.barplot(x=category_stats.index, y='Profit Margin', data=category_stats, ax=axes[1, 1])
    axes[1, 1].set_title('Profit Margin by Category')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()  # Show the plot
    
    # Save the plot
    plt.savefig('category_performance.png')
    plt.close()
    
    print("\nCategory performance visualisation saved as 'category_performance.png'")
    
    return category_stats

# Analyse categories
category_analysis = analyse_categories(df)

#%% [markdown]
"""
# 7. Customer Analysis
Let's examine customer behaviour and segments.

**Note:** We'll analyse:
- Customer segments
- Purchase frequency
- Average order value
- Customer lifetime value
"""

#%%
def analyse_customers(df):
    """ Analyse customer behaviour and segments."""
    print("\nAnalysing customer data...")
    
    # Create customer dataset
    customer_data = df.groupby('Customer ID').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count',
        'Discount': 'mean',
        'Shipping Time': 'mean',
        'Order Date': ['min', 'max']
    }).round(2)
    
    customer_data.columns = ['Total Sales', 'Total Profit', 'Number of Orders',
                           'Average Discount', 'Average Shipping Time',
                           'First Purchase', 'Last Purchase']
    
    customer_data['Customer Lifetime'] = (customer_data['Last Purchase'] - 
                                        customer_data['First Purchase']).dt.days
    customer_data['Average Order Value'] = (customer_data['Total Sales'] / 
                                          customer_data['Number of Orders'])
    
    print(f"\nTotal Customers: {len(customer_data)}")
    print("\nTop 5 Customers by Sales:")
    print(customer_data.nlargest(5, 'Total Sales'))
    
    # Create and show the visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Customer segment distribution
    df['Customer Segment'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0, 0])
    axes[0, 0].set_title('Customer Segment Distribution')
    axes[0, 0].set_ylabel('')
    
    # Average order value distribution
    sns.histplot(data=customer_data, x='Average Order Value', bins=50, ax=axes[0, 1])
    axes[0, 1].set_title('Average Order Value Distribution')
    
    # Purchase frequency
    sns.histplot(data=customer_data, x='Number of Orders', bins=50, ax=axes[1, 0])
    axes[1, 0].set_title('Purchase Frequency Distribution')
    
    # Customer lifetime
    sns.histplot(data=customer_data, x='Customer Lifetime', bins=50, ax=axes[1, 1])
    axes[1, 1].set_title('Customer Lifetime Distribution')
    
    plt.tight_layout()
    plt.show()  # Show the plot
    
    # Save the plot
    plt.savefig('customer_analysis.png')
    plt.close()
    
    print("\nCustomer analysis visualisation saved as 'customer_analysis.png'")
    
    return customer_data

# Analyse customers
customer_analysis = analyse_customers(df)

#%% [markdown]
"""
# 8. Regional Analysis
Let's examine performance by region.

**Note:** We'll analyse:
- Market share by region
- Regional profit margins
- Sales distribution
- Regional growth patterns
"""

#%%
def analyse_regions(df):
    """ Analyse performance by region."""
    print("\nAnalysing regional performance...")
    
    # Create regional dataset
    regional_data = df.groupby(['Region', 'State']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': 'count'
    }).round(2)
    
    regional_data['Profit Margin'] = (regional_data['Profit'] / 
                                    regional_data['Sales']) * 100
    regional_data['Average Order Value'] = (regional_data['Sales'] / 
                                          regional_data['Order ID'])
    regional_data['Market Share'] = (regional_data['Sales'] / 
                                   regional_data['Sales'].sum()) * 100
    
    print("\nRegional Market Share:")
    print(regional_data.groupby(level=0)['Market Share'].sum().round(2))
    
    # Create and show the visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Market share by region
    regional_data.groupby(level=0)['Market Share'].sum().plot(kind='pie', 
                                                            autopct='%1.1f%%', 
                                                            ax=axes[0, 0])
    axes[0, 0].set_title('Market Share by Region')
    axes[0, 0].set_ylabel('')
    
    # Profit by region
    sns.barplot(x=regional_data.index.get_level_values(0), 
                y='Profit', 
                data=regional_data, 
                ax=axes[0, 1])
    axes[0, 1].set_title('Total Profit by Region')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Sales by region
    sns.barplot(x=regional_data.index.get_level_values(0), 
                y='Sales', 
                data=regional_data, 
                ax=axes[1, 0])
    axes[1, 0].set_title('Total Sales by Region')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Profit margin by region
    sns.barplot(x=regional_data.index.get_level_values(0), 
                y='Profit Margin', 
                data=regional_data, 
                ax=axes[1, 1])
    axes[1, 1].set_title('Profit Margin by Region')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()  # Show the plot
    
    # Save the plot
    plt.savefig('regional_analysis.png')
    plt.close()
    
    print("\nRegional analysis visualisation saved as 'regional_analysis.png'")
    
    return regional_data

# Analyse regions
regional_analysis = analyse_regions(df)

#%% [markdown]
"""
# 9. Correlation Analysis
Let's examine relationships between numeric variables.

**Note:** We'll analyse:
- Correlations between key metrics
- Strong positive/negative relationships
- Potential causal relationships
"""

#%%
def analyse_correlations(df):
    """ Analyse correlations between numeric variables."""
    print("\nAnalysing correlations...")
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_columns].corr()
    
    # Create and show the visualisation
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()  # Show the plot
    
    # Save the plot
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    print("\nCorrelation matrix visualisation saved as 'correlation_matrix.png'")
    
    # Print strong correlations
    print("\nStrong Correlations (|r| > 0.5):")
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(f"{numeric_columns[i]} - {numeric_columns[j]}: {corr:.2f}")
    
    return correlation_matrix

# Analyse correlations
correlation_analysis = analyse_correlations(df)

#%% [markdown]
"""
# 10. Summary and Insights
Let's go ahead and summarise our key findings and provide actionable recommendations.

**Note:** This section will:
- Highlight key performance metrics
- Identify growth opportunities
- Suggest strategic improvements
- Provide actionable recommendations
"""

#%%
def summarise_insights(df, sales_analysis, category_analysis, customer_analysis, regional_analysis):
    """ Summarise key insights from the analysis."""
    print("\n=== KEY INSIGHTS ===")
    
    # Sales Performance
    print("\n1. Sales Performance:")
    print(f"• Total Sales: ${sales_analysis['total_sales']:,.2f}")
    print(f"• Total Profit: ${sales_analysis['total_profit']:,.2f}")
    print(f"• Overall Profit Margin: {sales_analysis['profit_margin']:.2f}%")
    
    # Category Performance
    print("\n2. Category Performance:")
    best_category = category_analysis['Profit'].idxmax()
    print(f"• Best Performing Category: {best_category}")
    print(f"• Category Sales Distribution:")
    print(category_analysis['Sales'].round(2))
    
    # Customer Insights
    print("\n3. Customer Insights:")
    print(f"• Total Customers: {len(customer_analysis)}")
    print("• Customer Segment Distribution:")
    print(df['Customer Segment'].value_counts(normalize=True).round(2))
    
    # Regional Performance
    print("\n4. Regional Performance:")
    print("• Market Share by Region:")
    print(regional_analysis.groupby(level=0)['Market Share'].sum().round(2))
    
    # Recommendations
    print("\n5. Strategic Recommendations:")
    print("• Focus on high-margin categories and products")
    print("• Develop targeted marketing for high-value customer segments")
    print("• Optimise inventory and pricing strategies by region")
    print("• Implement customer retention programmes")
    print("• Consider regional-specific product offerings")
    print("• Invest in underperforming regions with growth potential")
    print("• Develop cross-selling strategies between high-margin products")
    print("• Implement seasonal inventory planning based on trends")
    print("• Create region-specific marketing campaigns")
    print("• Monitor and adjust discount strategies based on impact analysis")

# Summarise insights
summarise_insights(df, sales_analysis, category_analysis, customer_analysis, regional_analysis)

#%% [markdown]
"""
# Analysis Complete!
All visualisations have been saved to the current directory.

**Visualisations Generated:**
- sales_trend.png: Monthly sales trend analysis
- category_performance.png: Category-wise performance metrics
- customer_analysis.png: Customer behaviour and segmentation
- regional_analysis.png: Regional market performance
- correlation_matrix.png: Relationships between key metrics

**Next Steps:**
1. Review the generated visualisations
2. Implement the recommended strategies
3. Monitor key performance indicators
4. Adjust strategies based on results
"""


