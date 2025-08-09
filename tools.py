import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from crewai.tools import tool
import os
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import tempfile
import json

warnings.filterwarnings('ignore')

# Global variable to store the loaded dataset
current_dataset = None
analysis_results = {}

@tool("load_data_tool")
def load_data_tool(file_path: str) -> str:
    """
    Load a dataset from various file formats (CSV, Excel, JSON).
    
    Args:
        file_path (str): Path to the dataset file
        
    Returns:
        str: Summary of the loaded dataset including shape, columns, and basic info
    """
    global current_dataset, analysis_results
    
    try:
        # Reset analysis results for new dataset
        analysis_results = {}
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            current_dataset = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            current_dataset = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            current_dataset = pd.read_json(file_path)
        else:
            return "Error: Unsupported file format. Please use CSV, Excel, or JSON files."
        
        # Basic dataset information
        shape = current_dataset.shape
        columns = list(current_dataset.columns)
        dtypes = current_dataset.dtypes.to_dict()
        
        # Memory usage
        memory_usage = current_dataset.memory_usage(deep=True).sum() / 1024**2  # MB
        
        # Store basic info
        analysis_results['dataset_info'] = {
            'shape': shape,
            'columns': columns,
            'dtypes': {col: str(dtype) for col, dtype in dtypes.items()},
            'memory_usage_mb': round(memory_usage, 2)
        }
        
        summary = f"""
Dataset Successfully Loaded!

📊 **Dataset Overview:**
- Shape: {shape[0]:,} rows × {shape[1]} columns
- Memory Usage: {memory_usage:.2f} MB
- File: {os.path.basename(file_path)}

📋 **Columns ({len(columns)}):**
{', '.join(columns)}

🔢 **Data Types:**
"""
        for col, dtype in dtypes.items():
            summary += f"\n- {col}: {dtype}"
            
        return summary
        
    except Exception as e:
        return f"Error loading dataset: {str(e)}"

@tool("missing_value_analysis_tool")
def missing_value_analysis_tool() -> str:
    """
    Analyze missing values in the dataset and provide insights.
    
    Returns:
        str: Detailed analysis of missing values with recommendations
    """
    global current_dataset, analysis_results
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_tool first."
    
    try:
        # Calculate missing values
        missing_counts = current_dataset.isnull().sum()
        missing_percentages = (missing_counts / len(current_dataset)) * 100
        
        # Create missing value summary
        missing_summary = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percentage': missing_percentages.values
        })
        missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        # Store results
        analysis_results['missing_values'] = {
            'summary': missing_summary.to_dict('records'),
            'total_missing': int(missing_counts.sum()),
            'columns_with_missing': len(missing_summary)
        }
        
        if missing_summary.empty:
            return "✅ **Excellent Data Quality!** No missing values found in the dataset."
        
        # Generate analysis
        analysis = f"""
🔍 **Missing Value Analysis**

📊 **Summary:**
- Total missing values: {missing_counts.sum():,}
- Columns with missing data: {len(missing_summary)}/{len(current_dataset.columns)}
- Overall completeness: {(1 - missing_counts.sum()/(len(current_dataset) * len(current_dataset.columns))) * 100:.1f}%

📋 **Columns with Missing Data:**
"""
        
        for _, row in missing_summary.iterrows():
            analysis += f"\n• **{row['Column']}**: {row['Missing_Count']:,} missing ({row['Missing_Percentage']:.1f}%)"
        
        # Recommendations
        analysis += "\n\n💡 **Recommendations:**\n"
        for _, row in missing_summary.iterrows():
            if row['Missing_Percentage'] > 50:
                analysis += f"• **{row['Column']}**: Consider dropping (>50% missing)\n"
            elif row['Missing_Percentage'] > 20:
                analysis += f"• **{row['Column']}**: Investigate pattern, consider imputation or feature engineering\n"
            elif row['Missing_Percentage'] > 5:
                analysis += f"• **{row['Column']}**: Apply appropriate imputation strategy\n"
            else:
                analysis += f"• **{row['Column']}**: Simple imputation or dropping rows may suffice\n"
        
        return analysis
        
    except Exception as e:
        return f"Error in missing value analysis: {str(e)}"

@tool("univariate_analysis_tool")
def univariate_analysis_tool() -> str:
    """
    Perform univariate analysis on all columns in the dataset.
    
    Returns:
        str: Comprehensive univariate analysis results
    """
    global current_dataset, analysis_results
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_tool first."
    
    try:
        numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = current_dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        
        analysis = "📊 **Univariate Analysis**\n\n"
        
        # Store analysis results
        analysis_results['univariate'] = {
            'numeric_analysis': {},
            'categorical_analysis': {}
        }
        
        # Numeric variables analysis
        if numeric_cols:
            analysis += f"🔢 **Numeric Variables ({len(numeric_cols)} columns):**\n\n"
            
            for col in numeric_cols:
                series = current_dataset[col].dropna()
                
                if len(series) == 0:
                    continue
                    
                stats = {
                    'count': len(series),
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(),
                    'q25': series.quantile(0.25),
                    'q75': series.quantile(0.75),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis()
                }
                
                analysis_results['univariate']['numeric_analysis'][col] = stats
                
                analysis += f"**{col}:**\n"
                analysis += f"• Count: {stats['count']:,} | Mean: {stats['mean']:.2f} | Median: {stats['median']:.2f}\n"
                analysis += f"• Std: {stats['std']:.2f} | Min: {stats['min']:.2f} | Max: {stats['max']:.2f}\n"
                analysis += f"• Q1: {stats['q25']:.2f} | Q3: {stats['q75']:.2f}\n"
                analysis += f"• Skewness: {stats['skewness']:.2f} | Kurtosis: {stats['kurtosis']:.2f}\n"
                
                # Interpretation
                if abs(stats['skewness']) > 1:
                    skew_desc = "highly skewed"
                elif abs(stats['skewness']) > 0.5:
                    skew_desc = "moderately skewed"
                else:
                    skew_desc = "approximately symmetric"
                
                analysis += f"• Distribution: {skew_desc}\n\n"
        
        # Categorical variables analysis
        if categorical_cols:
            analysis += f"📝 **Categorical Variables ({len(categorical_cols)} columns):**\n\n"
            
            for col in categorical_cols:
                series = current_dataset[col].dropna()
                
                if len(series) == 0:
                    continue
                
                value_counts = series.value_counts()
                unique_count = len(value_counts)
                
                cat_stats = {
                    'unique_count': unique_count,
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0,
                    'value_counts': value_counts.head(10).to_dict()
                }
                
                analysis_results['univariate']['categorical_analysis'][col] = cat_stats
                
                analysis += f"**{col}:**\n"
                analysis += f"• Unique values: {unique_count}\n"
                analysis += f"• Most frequent: '{cat_stats['most_frequent']}' ({cat_stats['most_frequent_count']} times)\n"
                
                if unique_count <= 10:
                    analysis += "• All values:\n"
                    for value, count in value_counts.items():
                        percentage = (count / len(series)) * 100
                        analysis += f"  - {value}: {count} ({percentage:.1f}%)\n"
                else:
                    analysis += f"• Top 5 values:\n"
                    for value, count in value_counts.head(5).items():
                        percentage = (count / len(series)) * 100
                        analysis += f"  - {value}: {count} ({percentage:.1f}%)\n"
                
                analysis += "\n"
        
        return analysis
        
    except Exception as e:
        return f"Error in univariate analysis: {str(e)}"

@tool("correlation_analysis_tool")
def correlation_analysis_tool() -> str:
    """
    Compute correlation matrix for numeric variables and identify key relationships.
    
    Returns:
        str: Correlation analysis results with key findings
    """
    global current_dataset, analysis_results
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_tool first."
    
    try:
        numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return "⚠️ **Insufficient numeric columns** for correlation analysis. Need at least 2 numeric columns."
        
        # Calculate correlation matrix
        corr_matrix = current_dataset[numeric_cols].corr()
        
        # Store correlation matrix
        analysis_results['correlation'] = {
            'matrix': corr_matrix.to_dict(),
            'strong_correlations': [],
            'moderate_correlations': []
        }
        
        analysis = "🔗 **Correlation Analysis**\n\n"
        
        # Find strong correlations (excluding diagonal)
        strong_corr = []
        moderate_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                
                if abs(corr_val) >= 0.7:
                    strong_corr.append((var1, var2, corr_val))
                elif abs(corr_val) >= 0.5:
                    moderate_corr.append((var1, var2, corr_val))
        
        # Store findings
        analysis_results['correlation']['strong_correlations'] = [
            {'var1': x[0], 'var2': x[1], 'correlation': x[2]} for x in strong_corr
        ]
        analysis_results['correlation']['moderate_correlations'] = [
            {'var1': x[0], 'var2': x[1], 'correlation': x[2]} for x in moderate_corr
        ]
        
        analysis += f"📊 **Matrix Size:** {len(numeric_cols)} × {len(numeric_cols)} numeric variables\n\n"
        
        # Strong correlations
        if strong_corr:
            analysis += "🔴 **Strong Correlations (|r| ≥ 0.7):**\n"
            for var1, var2, corr_val in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                direction = "positive" if corr_val > 0 else "negative"
                analysis += f"• **{var1}** ↔ **{var2}**: {corr_val:.3f} ({direction})\n"
            analysis += "\n"
        
        # Moderate correlations
        if moderate_corr:
            analysis += "🟡 **Moderate Correlations (0.5 ≤ |r| < 0.7):**\n"
            for var1, var2, corr_val in sorted(moderate_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:  # Top 10
                direction = "positive" if corr_val > 0 else "negative"
                analysis += f"• **{var1}** ↔ **{var2}**: {corr_val:.3f} ({direction})\n"
            analysis += "\n"
        
        if not strong_corr and not moderate_corr:
            analysis += "✅ **Low Correlations:** No strong correlations detected. Variables are relatively independent.\n\n"
        
        # Recommendations
        analysis += "💡 **Recommendations:**\n"
        if strong_corr:
            analysis += "• Consider multicollinearity issues for modeling\n"
            analysis += "• Evaluate feature selection or dimensionality reduction\n"
        if len(strong_corr) > 5:
            analysis += "• Consider principal component analysis (PCA)\n"
        if not strong_corr and not moderate_corr:
            analysis += "• Good feature independence for modeling\n"
        
        return analysis
        
    except Exception as e:
        return f"Error in correlation analysis: {str(e)}"

@tool("outlier_detection_tool")
def outlier_detection_tool() -> str:
    """
    Detect outliers in numeric variables using IQR method and Z-score.
    
    Returns:
        str: Outlier detection results with recommendations
    """
    global current_dataset, analysis_results
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_tool first."
    
    try:
        numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return "⚠️ **No numeric columns** found for outlier detection."
        
        analysis = "🎯 **Outlier Detection Analysis**\n\n"
        
        outlier_summary = {}
        
        for col in numeric_cols:
            series = current_dataset[col].dropna()
            
            if len(series) == 0:
                continue
            
            # IQR method
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Z-score method (|z| > 3)
            z_scores = np.abs((series - series.mean()) / series.std())
            z_outliers = series[z_scores > 3]
            
            outlier_info = {
                'iqr_outliers_count': len(iqr_outliers),
                'iqr_outliers_percentage': (len(iqr_outliers) / len(series)) * 100,
                'z_outliers_count': len(z_outliers),
                'z_outliers_percentage': (len(z_outliers) / len(series)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': iqr_outliers.tolist()[:10]  # Store first 10 outliers
            }
            
            outlier_summary[col] = outlier_info
            
            if len(iqr_outliers) > 0 or len(z_outliers) > 0:
                analysis += f"**{col}:**\n"
                analysis += f"• IQR Method: {len(iqr_outliers)} outliers ({(len(iqr_outliers)/len(series)*100):.1f}%)\n"
                analysis += f"• Z-Score Method: {len(z_outliers)} outliers ({(len(z_outliers)/len(series)*100):.1f}%)\n"
                analysis += f"• Valid range (IQR): [{lower_bound:.2f}, {upper_bound:.2f}]\n"
                
                if len(iqr_outliers) > 0:
                    analysis += f"• Sample outliers: {iqr_outliers.head(5).tolist()}\n"
                
                # Severity assessment
                outlier_percentage = (len(iqr_outliers) / len(series)) * 100
                if outlier_percentage > 10:
                    severity = "High"
                elif outlier_percentage > 5:
                    severity = "Moderate"
                else:
                    severity = "Low"
                
                analysis += f"• Severity: {severity}\n\n"
        
        # Store results
        analysis_results['outliers'] = outlier_summary
        
        # Overall summary
        total_outlier_cols = sum(1 for col_info in outlier_summary.values() if col_info['iqr_outliers_count'] > 0)
        
        analysis += f"📊 **Summary:**\n"
        analysis += f"• Columns analyzed: {len(numeric_cols)}\n"
        analysis += f"• Columns with outliers: {total_outlier_cols}\n"
        
        if total_outlier_cols > 0:
            analysis += "\n💡 **Recommendations:**\n"
            analysis += "• Investigate outliers to determine if they are:\n"
            analysis += "  - Data entry errors (correct or remove)\n"
            analysis += "  - Valid extreme values (keep or cap)\n"
            analysis += "  - Measurement errors (investigate)\n"
            analysis += "• Consider transformation methods (log, sqrt) for skewed data\n"
            analysis += "• Use robust models if outliers are legitimate\n"
        else:
            analysis += "✅ **Clean data:** No significant outliers detected!\n"
        
        return analysis
        
    except Exception as e:
        return f"Error in outlier detection: {str(e)}"

@tool("target_relationship_tool")
def target_relationship_tool(target_column: str = None) -> str:
    """
    Analyze relationships between features and the target variable.
    
    Args:
        target_column (str): Name of the target variable column
        
    Returns:
        str: Analysis of relationships with target variable
    """
    global current_dataset, analysis_results
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_tool first."
    
    try:
        if target_column is None or target_column not in current_dataset.columns:
            # Try to infer target column
            potential_targets = ['target', 'label', 'class', 'outcome', 'y', 'price', 'sales', 'revenue']
            for col in potential_targets:
                if col in current_dataset.columns:
                    target_column = col
                    break
            
            if target_column is None:
                return "⚠️ **No target column specified or found.** Please specify a target column name."
        
        target_series = current_dataset[target_column]
        analysis = f"🎯 **Target Variable Analysis: {target_column}**\n\n"
        
        # Target variable analysis
        is_numeric_target = pd.api.types.is_numeric_dtype(target_series)
        
        target_info = {
            'column_name': target_column,
            'is_numeric': is_numeric_target,
            'missing_count': target_series.isnull().sum(),
            'unique_count': target_series.nunique()
        }
        
        if is_numeric_target:
            target_info.update({
                'mean': target_series.mean(),
                'median': target_series.median(),
                'std': target_series.std(),
                'min': target_series.min(),
                'max': target_series.max()
            })
            analysis += f"📊 **Target Statistics:**\n"
            analysis += f"• Type: Numeric (Regression problem)\n"
            analysis += f"• Mean: {target_info['mean']:.2f}\n"
            analysis += f"• Median: {target_info['median']:.2f}\n"
            analysis += f"• Range: [{target_info['min']:.2f}, {target_info['max']:.2f}]\n"
            analysis += f"• Standard Deviation: {target_info['std']:.2f}\n"
        else:
            value_counts = target_series.value_counts()
            target_info['value_counts'] = value_counts.to_dict()
            analysis += f"📊 **Target Statistics:**\n"
            analysis += f"• Type: Categorical (Classification problem)\n"
            analysis += f"• Classes: {len(value_counts)}\n"
            
            for class_name, count in value_counts.items():
                percentage = (count / len(target_series)) * 100
                analysis += f"• {class_name}: {count} ({percentage:.1f}%)\n"
        
        analysis += f"• Missing values: {target_info['missing_count']}\n\n"
        
        # Analyze relationships with other variables
        numeric_cols = [col for col in current_dataset.select_dtypes(include=[np.number]).columns if col != target_column]
        categorical_cols = [col for col in current_dataset.select_dtypes(include=['object', 'category']).columns if col != target_column]
        
        relationships = {'numeric': {}, 'categorical': {}}
        
        # Numeric feature relationships
        if numeric_cols and is_numeric_target:
            analysis += "🔢 **Numeric Feature Correlations:**\n"
            
            for col in numeric_cols:
                corr = current_dataset[col].corr(target_series)
                if not pd.isna(corr):
                    relationships['numeric'][col] = corr
                    
                    strength = "Strong" if abs(corr) >= 0.7 else "Moderate" if abs(corr) >= 0.5 else "Weak"
                    direction = "positive" if corr > 0 else "negative"
                    
                    analysis += f"• **{col}**: {corr:.3f} ({strength} {direction})\n"
            
            # Top correlated features
            sorted_corr = sorted(relationships['numeric'].items(), key=lambda x: abs(x[1]), reverse=True)
            if sorted_corr:
                analysis += f"\n🏆 **Top correlated features:**\n"
                for col, corr in sorted_corr[:5]:
                    analysis += f"• {col}: {corr:.3f}\n"
            
            analysis += "\n"
        
        # Categorical feature relationships
        if categorical_cols:
            analysis += "📝 **Categorical Feature Analysis:**\n"
            
            for col in categorical_cols:
                unique_vals = current_dataset[col].nunique()
                
                if unique_vals <= 20:  # Only analyze if reasonable number of categories
                    if is_numeric_target:
                        # Mean target by category
                        group_means = current_dataset.groupby(col)[target_column].mean()
                        relationships['categorical'][col] = group_means.to_dict()
                        
                        analysis += f"• **{col}** (Mean {target_column} by category):\n"
                        for cat, mean_val in group_means.items():
                            analysis += f"  - {cat}: {mean_val:.2f}\n"
                    else:
                        # Cross-tabulation for categorical target
                        crosstab = pd.crosstab(current_dataset[col], target_series, normalize='index') * 100
                        analysis += f"• **{col}** (Distribution by {target_column}):\n"
                        for cat in crosstab.index:
                            analysis += f"  - {cat}: "
                            for target_val in crosstab.columns:
                                analysis += f"{target_val} ({crosstab.loc[cat, target_val]:.1f}%) "
                            analysis += "\n"
                
                analysis += "\n"
        
        # Store results
        analysis_results['target_analysis'] = {
            'target_info': target_info,
            'relationships': relationships
        }
        
        # Recommendations
        analysis += "💡 **Recommendations:**\n"
        
        if is_numeric_target:
            strong_corr = [col for col, corr in relationships['numeric'].items() if abs(corr) >= 0.7]
            if strong_corr:
                analysis += f"• Strong predictors identified: {', '.join(strong_corr)}\n"
            else:
                analysis += "• No strong linear relationships found - consider non-linear models\n"
        
        if categorical_cols:
            analysis += "• Consider encoding categorical variables for modeling\n"
        
        analysis += "• Visualize relationships with scatter plots or box plots\n"
        analysis += "• Consider feature engineering based on identified patterns\n"
        
        return analysis
        
    except Exception as e:
        return f"Error in target relationship analysis: {str(e)}"

@tool("generate_visualizations_tool")
def generate_visualizations_tool() -> str:
    """
    Generate comprehensive visualizations for the EDA analysis including:
    - Histograms for numeric distributions
    - Box plots for outlier detection
    - Density plots (KDE) for smooth distributions
    - Bar charts for categorical data
    - Pie charts for categorical percentages
    - Correlation heatmaps
    - Outlier detection plots
    
    Returns:
        str: Summary of generated visualizations with insights
    """
    global current_dataset, analysis_results
    
    if current_dataset is None:
        return "Error: No dataset loaded. Please use load_data_tool first."
    
    try:
        numeric_cols = current_dataset.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = current_dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store visualization info
        visualizations = []
        
        # Create plots directory if it doesn't exist
        plots_dir = "temp_plots"
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        analysis = "📊 **Comprehensive Visualization Analysis**\n\n"
        
        # 1. HISTOGRAMS - Distribution patterns for numeric variables
        if numeric_cols:
            analysis += "� **Histograms - Value Distribution:**\n"
            
            n_numeric = len(numeric_cols)
            if n_numeric <= 4:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                axes = axes.ravel()
            else:
                rows = (n_numeric + 3) // 4
                fig, axes = plt.subplots(rows, 4, figsize=(20, 5*rows))
                axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:min(len(numeric_cols), 16)]):
                if i < len(axes):
                    # Enhanced histogram with better styling
                    current_dataset[col].dropna().hist(bins=30, ax=axes[i], alpha=0.7, 
                                                      color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Frequency', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add distribution statistics
                    mean_val = current_dataset[col].mean()
                    axes[i].axvline(mean_val, color='red', linestyle='--', 
                                   label=f'Mean: {mean_val:.2f}')
                    axes[i].legend()
            
            # Hide unused subplots
            for i in range(len(numeric_cols), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            hist_path = os.path.join(plots_dir, 'histograms.png')
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                'type': 'histogram',
                'title': 'Distribution Histograms',
                'path': hist_path,
                'description': 'Shows value distribution patterns with mean indicators'
            })
            
            analysis += f"• Generated histograms for {len(numeric_cols)} numeric variables\n"
            analysis += "• Red dashed lines show mean values\n"
            analysis += "• Look for normal, skewed, or multimodal distributions\n\n"
        
        
        # 2. BOX PLOTS - Outlier detection and quartile analysis
        if numeric_cols:
            analysis += "📦 **Box Plots - Outlier Detection:**\n"
            
            n_plots = min(len(numeric_cols), 8)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.ravel()
            
            outlier_counts = {}
            for i, col in enumerate(numeric_cols[:n_plots]):
                # Calculate outliers using IQR method
                Q1 = current_dataset[col].quantile(0.25)
                Q3 = current_dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = current_dataset[col][(current_dataset[col] < lower_bound) | 
                                               (current_dataset[col] > upper_bound)]
                outlier_counts[col] = len(outliers)
                
                # Create box plot
                current_dataset.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'Box Plot: {col}\n({len(outliers)} outliers)', 
                                 fontsize=11, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            box_path = os.path.join(plots_dir, 'box_plots.png')
            plt.savefig(box_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                'type': 'boxplot',
                'title': 'Box Plots for Outlier Detection',
                'path': box_path,
                'description': 'Shows quartiles and identifies outliers using IQR method'
            })
            
            analysis += f"• Generated box plots for {n_plots} variables\n"
            analysis += "• Outlier counts shown in titles\n"
            analysis += f"• Total outliers detected: {sum(outlier_counts.values())}\n\n"
        
        # 3. DENSITY PLOTS (KDE) - Smooth distribution curves
        if numeric_cols:
            analysis += "🌊 **Density Plots (KDE) - Smooth Distributions:**\n"
            
            n_kde = min(len(numeric_cols), 8)
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(numeric_cols[:n_kde]):
                # Create KDE plot
                data_clean = current_dataset[col].dropna()
                if len(data_clean) > 1:
                    data_clean.plot.kde(ax=axes[i], color='purple', linewidth=2)
                    axes[i].fill_between(axes[i].get_xlim(), 0, 
                                        axes[i].get_ylim()[1], alpha=0.3, color='purple')
                    axes[i].set_title(f'Density Plot: {col}', fontsize=11, fontweight='bold')
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Density', fontsize=10)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add mean and median lines
                    mean_val = data_clean.mean()
                    median_val = data_clean.median()
                    axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                    axes[i].axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.2f}')
                    axes[i].legend()
            
            # Hide unused subplots
            for i in range(n_kde, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            kde_path = os.path.join(plots_dir, 'density_plots.png')
            plt.savefig(kde_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                'type': 'kde',
                'title': 'Density Plots (KDE)',
                'path': kde_path,
                'description': 'Smooth curves showing distribution patterns with mean/median indicators'
            })
            
            analysis += f"• Generated KDE plots for {n_kde} variables\n"
            analysis += "• Red dashed lines show means, green dotted lines show medians\n"
            analysis += "• Smooth curves reveal distribution shapes and skewness\n\n"
        
        # 4. BAR CHARTS - Categorical frequency distributions
        if categorical_cols:
            analysis += "📊 **Bar Charts - Category Counts:**\n"
            
            # Filter categorical columns with reasonable number of categories
            suitable_cats = [col for col in categorical_cols if current_dataset[col].nunique() <= 15]
            
            if suitable_cats:
                n_cat = min(len(suitable_cats), 8)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(suitable_cats[:n_cat]):
                    value_counts = current_dataset[col].value_counts().head(10)
                    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
                    
                    bars = value_counts.plot(kind='bar', ax=axes[i], color=colors)
                    axes[i].set_title(f'Frequency: {col}\n({current_dataset[col].nunique()} categories)', 
                                     fontsize=11, fontweight='bold')
                    axes[i].set_xlabel(col, fontsize=10)
                    axes[i].set_ylabel('Count', fontsize=10)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for bar in bars.patches:
                        height = bar.get_height()
                        axes[i].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height)}', ha='center', va='bottom', fontsize=8)
                
                # Hide unused subplots
                for i in range(n_cat, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                bar_path = os.path.join(plots_dir, 'bar_charts.png')
                plt.savefig(bar_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                visualizations.append({
                    'type': 'bar_chart',
                    'title': 'Categorical Bar Charts',
                    'path': bar_path,
                    'description': 'Shows frequency distributions with value labels'
                })
                
                analysis += f"• Generated bar charts for {n_cat} categorical variables\n"
                analysis += "• Value counts displayed on top of bars\n"
                analysis += "• Only showing top 10 categories per variable\n\n"
        
        # 5. PIE CHARTS - Categorical percentage breakdowns
        if categorical_cols:
            analysis += "🥧 **Pie Charts - Percentage Breakdown:**\n"
            
            # Select categorical variables suitable for pie charts (5-10 categories)
            pie_suitable = [col for col in categorical_cols 
                           if 3 <= current_dataset[col].nunique() <= 8]
            
            if pie_suitable:
                n_pie = min(len(pie_suitable), 8)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                axes = axes.ravel()
                
                for i, col in enumerate(pie_suitable[:n_pie]):
                    value_counts = current_dataset[col].value_counts()
                    colors = plt.cm.Pastel1(np.linspace(0, 1, len(value_counts)))
                    
                    wedges, texts, autotexts = axes[i].pie(value_counts.values, 
                                                          labels=value_counts.index,
                                                          autopct='%1.1f%%',
                                                          colors=colors,
                                                          startangle=90)
                    axes[i].set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
                    
                    # Enhance text readability
                    for autotext in autotexts:
                        autotext.set_color('black')
                        autotext.set_fontsize(9)
                        autotext.set_fontweight('bold')
                
                # Hide unused subplots
                for i in range(n_pie, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                pie_path = os.path.join(plots_dir, 'pie_charts.png')
                plt.savefig(pie_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                visualizations.append({
                    'type': 'pie_chart',
                    'title': 'Categorical Pie Charts',
                    'path': pie_path,
                    'description': 'Shows percentage breakdown of categories'
                })
                
                analysis += f"• Generated pie charts for {n_pie} categorical variables\n"
                analysis += "• Percentages shown for each category\n"
                analysis += "• Only variables with 3-8 categories included\n\n"
        
        # 6. CORRELATION HEATMAP - Relationship detection
        if len(numeric_cols) >= 2:
            analysis += "🔗 **Correlation Heatmap - Relationship Detection:**\n"
            
            plt.figure(figsize=(12, 10))
            corr_matrix = current_dataset[numeric_cols].corr()
            
            # Generate full heatmap without mask (like the first image)
            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                       fmt='.2f', annot_kws={'size': 10})
            plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            corr_path = os.path.join(plots_dir, 'correlation_heatmap.png')
            plt.savefig(corr_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # Strong correlation threshold
                        strong_corr.append((corr_matrix.columns[i], 
                                          corr_matrix.columns[j], corr_val))
            
            visualizations.append({
                'type': 'correlation_heatmap',
                'title': 'Correlation Matrix',
                'path': corr_path,
                'description': 'Shows relationships between numeric variables'
            })
            
            analysis += "• Generated correlation matrix visualization\n"
            analysis += "• Red indicates positive correlation, blue indicates negative\n"
            analysis += f"• Strong correlations (|r| > 0.7): {len(strong_corr)} pairs\n"
            if strong_corr:
                analysis += "• Strong correlation pairs:\n"
                for var1, var2, corr_val in strong_corr[:5]:  # Show top 5
                    analysis += f"  - {var1} ↔ {var2}: {corr_val:.3f}\n"
            analysis += "\n"
        
        
        # 7. OUTLIER DETECTION SUMMARY - Advanced outlier analysis
        if numeric_cols:
            analysis += "🎯 **Advanced Outlier Detection Summary:**\n"
            
            plt.figure(figsize=(15, 8))
            
            # Create subplots for outlier summary
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # Left plot: Outlier counts per variable
            outlier_summary = {}
            for col in numeric_cols:
                Q1 = current_dataset[col].quantile(0.25)
                Q3 = current_dataset[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = current_dataset[col][(current_dataset[col] < lower_bound) | 
                                               (current_dataset[col] > upper_bound)]
                outlier_summary[col] = len(outliers)
            
            # Bar chart of outlier counts
            vars_with_outliers = {k: v for k, v in outlier_summary.items() if v > 0}
            if vars_with_outliers:
                ax1.bar(vars_with_outliers.keys(), vars_with_outliers.values(), 
                       color='orange', alpha=0.7)
                ax1.set_title('Outlier Count by Variable', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Number of Outliers')
                ax1.tick_params(axis='x', rotation=45)
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for i, (var, count) in enumerate(vars_with_outliers.items()):
                    ax1.text(i, count + 0.1, str(count), ha='center', va='bottom')
            
            # Right plot: Outlier percentage
            total_rows = len(current_dataset)
            outlier_pct = {k: (v/total_rows)*100 for k, v in outlier_summary.items() if v > 0}
            if outlier_pct:
                ax2.bar(outlier_pct.keys(), outlier_pct.values(), 
                       color='red', alpha=0.6)
                ax2.set_title('Outlier Percentage by Variable', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Percentage of Total Data (%)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.grid(True, alpha=0.3)
                
                # Add percentage labels
                for i, (var, pct) in enumerate(outlier_pct.items()):
                    ax2.text(i, pct + 0.1, f'{pct:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            outlier_path = os.path.join(plots_dir, 'outlier_summary.png')
            plt.savefig(outlier_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualizations.append({
                'type': 'outlier_summary',
                'title': 'Outlier Detection Summary',
                'path': outlier_path,
                'description': 'Comprehensive outlier analysis across all numeric variables'
            })
            
            total_outliers = sum(outlier_summary.values())
            analysis += f"• Total outliers detected: {total_outliers}\n"
            analysis += f"• Variables with outliers: {len(vars_with_outliers)}\n"
            if vars_with_outliers:
                max_outlier_var = max(vars_with_outliers, key=vars_with_outliers.get)
                analysis += f"• Variable with most outliers: {max_outlier_var} ({vars_with_outliers[max_outlier_var]})\n"
            analysis += "\n"
        
        # 8. MISSING VALUE HEATMAP - Data quality visualization
        if current_dataset.isnull().sum().sum() > 0:
            analysis += "🔍 **Missing Data Pattern Analysis:**\n"
            
            plt.figure(figsize=(15, 8))
            missing_data = current_dataset.isnull()
            
            if missing_data.sum().sum() > 0:
                # Create missing data heatmap
                sns.heatmap(missing_data, yticklabels=False, cbar=True, 
                           cmap='viridis_r', cbar_kws={'label': 'Missing Data'})
                plt.title('Missing Data Pattern Heatmap\n(Yellow = Missing, Purple = Present)', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('Variables')
                plt.tight_layout()
                
                missing_path = os.path.join(plots_dir, 'missing_data_heatmap.png')
                plt.savefig(missing_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                visualizations.append({
                    'type': 'missing_heatmap',
                    'title': 'Missing Data Pattern',
                    'path': missing_path,
                    'description': 'Shows patterns and distribution of missing values'
                })
                
                # Calculate missing data statistics
                missing_stats = current_dataset.isnull().sum()
                missing_pct = (missing_stats / len(current_dataset)) * 100
                vars_with_missing = missing_stats[missing_stats > 0]
                
                analysis += "• Visualizes missing data patterns across dataset\n"
                analysis += f"• Variables with missing data: {len(vars_with_missing)}\n"
                analysis += f"• Total missing values: {missing_stats.sum()}\n"
                if len(vars_with_missing) > 0:
                    worst_var = missing_pct.idxmax()
                    analysis += f"• Variable with most missing data: {worst_var} ({missing_pct[worst_var]:.1f}%)\n"
                analysis += "\n"
        
        # Store visualization info
        analysis_results['visualizations'] = visualizations
        
        # Final summary
        analysis += f"✅ **Comprehensive Visualization Summary:**\n"
        analysis += f"• Generated {len(visualizations)} visualization type(s)\n"
        analysis += f"• Numeric variables analyzed: {len(numeric_cols)}\n"
        analysis += f"• Categorical variables analyzed: {len(categorical_cols)}\n"
        analysis += f"• All plots saved in '{plots_dir}' directory\n\n"
        
        analysis += "📋 **Generated Visualizations:**\n"
        for viz in visualizations:
            analysis += f"• {viz['title']}: {viz['description']}\n"
        
        analysis += "\n🎨 **Visualization Insights:**\n"
        analysis += "• Histograms reveal distribution shapes and potential data issues\n"
        analysis += "• Box plots identify outliers and show data spread\n"
        analysis += "• KDE plots provide smooth distribution curves\n"
        analysis += "• Bar/Pie charts show categorical data patterns\n"
        analysis += "• Correlation heatmaps reveal variable relationships\n"
        analysis += "• Outlier analysis helps identify data quality issues\n"
        
        return analysis
        
    except Exception as e:
        return f"Error generating comprehensive visualizations: {str(e)}"

@tool("export_report_tool")
def export_report_tool() -> str:
    """
    Compile all EDA findings into a structured report format.
    
    Returns:
        str: Summary of the complete EDA report
    """
    global analysis_results
    
    try:
        if not analysis_results:
            return "Error: No analysis results available. Please run the EDA tools first."
        
        # Compile comprehensive report
        report = {
            'executive_summary': {},
            'dataset_overview': analysis_results.get('dataset_info', {}),
            'data_quality': {
                'missing_values': analysis_results.get('missing_values', {}),
                'outliers': analysis_results.get('outliers', {})
            },
            'statistical_analysis': {
                'univariate': analysis_results.get('univariate', {}),
                'correlation': analysis_results.get('correlation', {}),
                'target_analysis': analysis_results.get('target_analysis', {})
            },
            'visualizations': analysis_results.get('visualizations', []),
            'recommendations': []
        }
        
        # Generate executive summary
        dataset_info = analysis_results.get('dataset_info', {})
        missing_info = analysis_results.get('missing_values', {})
        
        exec_summary = []
        
        if dataset_info:
            shape = dataset_info.get('shape', [0, 0])
            exec_summary.append(f"Dataset contains {shape[0]:,} rows and {shape[1]} columns")
        
        if missing_info:
            total_missing = missing_info.get('total_missing', 0)
            if total_missing > 0:
                exec_summary.append(f"Data quality issue: {total_missing:,} missing values detected")
            else:
                exec_summary.append("Excellent data quality: No missing values")
        
        # Correlation insights
        corr_info = analysis_results.get('correlation', {})
        strong_corr = corr_info.get('strong_correlations', [])
        if strong_corr:
            exec_summary.append(f"Found {len(strong_corr)} strong correlation(s) - consider multicollinearity")
        
        # Outlier insights
        outlier_info = analysis_results.get('outliers', {})
        outlier_cols = sum(1 for col_info in outlier_info.values() if col_info.get('iqr_outliers_count', 0) > 0)
        if outlier_cols > 0:
            exec_summary.append(f"Outliers detected in {outlier_cols} variable(s)")
        
        report['executive_summary'] = exec_summary
        
        # Generate recommendations
        recommendations = []
        
        # Data quality recommendations
        if missing_info and missing_info.get('total_missing', 0) > 0:
            recommendations.append("Address missing values using appropriate imputation strategies")
        
        if outlier_cols > 0:
            recommendations.append("Investigate and handle outliers before modeling")
        
        # Modeling recommendations
        if strong_corr:
            recommendations.append("Consider feature selection due to multicollinearity")
        
        # General recommendations
        recommendations.extend([
            "Perform feature engineering based on identified patterns",
            "Consider data transformations for skewed variables",
            "Validate findings with domain expertise"
        ])
        
        report['recommendations'] = recommendations
        
        # Store final report
        analysis_results['final_report'] = report
        
        # Generate summary text
        summary = "📋 **EDA Report Generated Successfully!**\n\n"
        
        summary += "🎯 **Executive Summary:**\n"
        for item in exec_summary:
            summary += f"• {item}\n"
        summary += "\n"
        
        summary += "📊 **Report Sections:**\n"
        summary += "• Dataset Overview and Basic Statistics\n"
        summary += "• Data Quality Assessment (Missing Values & Outliers)\n"
        summary += "• Univariate Analysis for All Variables\n"
        summary += "• Correlation Analysis and Relationships\n"
        if analysis_results.get('target_analysis'):
            summary += "• Target Variable Analysis\n"
        summary += f"• Visual Analysis ({len(report['visualizations'])} charts)\n"
        summary += "• Actionable Recommendations\n\n"
        
        summary += "💡 **Key Recommendations:**\n"
        for rec in recommendations[:5]:  # Top 5 recommendations
            summary += f"• {rec}\n"
        
        summary += f"\n✅ **Report Status:** Complete and ready for export\n"
        summary += f"📁 **Analysis Results:** Stored in memory for Streamlit display\n"
        
        return summary
        
    except Exception as e:
        return f"Error generating report: {str(e)}"

# Helper function to get current analysis results
def get_analysis_results():
    """Get the current analysis results for Streamlit display"""
    global analysis_results
    return analysis_results

# Helper function to clear analysis results
def clear_analysis_results():
    """Clear analysis results for new analysis"""
    global analysis_results, current_dataset
    analysis_results = {}
    current_dataset = None
