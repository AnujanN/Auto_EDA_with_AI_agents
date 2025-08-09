import streamlit as st
import pandas as pd
import os
import tempfile
from pathlib import Path
import sys
import base64
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.colors import HexColor

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from main import run_eda_analysis
from tools import get_analysis_results, clear_analysis_results

# Configure Streamlit page
st.set_page_config(
    page_title="EDA with CrewAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #000000;
    }
    
    .metric-container h4 {
        color: #000000 !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-container p {
        color: #000000 !important;
        margin: 0;
    }
    
    .insight-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        color: #000000 !important;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #000000 !important;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

def create_pdf_report(analysis_results):
    """Create a PDF report from analysis results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=HexColor('#2E86AB'),
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=HexColor('#A23B72'),
        leftIndent=0
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        textColor=HexColor('#333333'),
        leftIndent=0
    )
    
    # Title
    story.append(Paragraph("Exploratory Data Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    if 'final_report' in analysis_results:
        report = analysis_results['final_report']
        
        story.append(Paragraph("Executive Summary", heading_style))
        exec_summary = report.get('executive_summary', [])
        for item in exec_summary:
            story.append(Paragraph(f"• {item}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Dataset Overview
        if 'dataset_overview' in report:
            story.append(Paragraph("Dataset Overview", heading_style))
            dataset_info = report['dataset_overview']
            if 'shape' in dataset_info:
                shape = dataset_info['shape']
                story.append(Paragraph(f"Dataset Shape: {shape[0]:,} rows × {shape[1]} columns", styles['Normal']))
            if 'memory_usage_mb' in dataset_info:
                story.append(Paragraph(f"Memory Usage: {dataset_info['memory_usage_mb']:.2f} MB", styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Data Quality
        if 'data_quality' in report:
            story.append(Paragraph("Data Quality Assessment", heading_style))
            
            # Missing values
            missing_info = report['data_quality'].get('missing_values', {})
            if missing_info:
                total_missing = missing_info.get('total_missing', 0)
                if total_missing > 0:
                    story.append(Paragraph(f"Missing Values: {total_missing:,} total missing values found", styles['Normal']))
                    missing_summary = missing_info.get('summary', [])
                    for item in missing_summary[:5]:  # Top 5
                        story.append(Paragraph(f"• {item['Column']}: {item['Missing_Count']} missing ({item['Missing_Percentage']:.1f}%)", styles['Normal']))
                else:
                    story.append(Paragraph("Missing Values: No missing values detected", styles['Normal']))
            
            story.append(Spacer(1, 20))
    
    # Add Statistical Summary Tables
    if 'univariate' in analysis_results:
        story.append(Paragraph("Statistical Summary", heading_style))
        univariate = analysis_results['univariate']
        
        # Numeric variables summary
        if 'numeric_analysis' in univariate and univariate['numeric_analysis']:
            story.append(Paragraph("Numeric Variables Statistics", subheading_style))
            
            # Create table data
            table_data = [['Variable', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness']]
            for col, stats in list(univariate['numeric_analysis'].items())[:10]:  # Limit to 10 variables
                table_data.append([
                    col,
                    f"{stats['mean']:.2f}",
                    f"{stats['median']:.2f}",
                    f"{stats['std']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['skewness']:.2f}"
                ])
            
            # Create table
            table = Table(table_data, colWidths=[80, 60, 60, 60, 50, 50, 60])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
    
    # Add Correlation Analysis
    if 'correlation' in analysis_results:
        story.append(Paragraph("Correlation Analysis", heading_style))
        correlation = analysis_results['correlation']
        
        # Strong correlations
        strong_corr = correlation.get('strong_correlations', [])
        if strong_corr:
            story.append(Paragraph("Strong Correlations (|r| ≥ 0.7):", subheading_style))
            for corr in strong_corr[:5]:  # Top 5
                direction = "positive" if corr['correlation'] > 0 else "negative"
                story.append(Paragraph(f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({direction})", styles['Normal']))
        
        # Moderate correlations
        moderate_corr = correlation.get('moderate_correlations', [])
        if moderate_corr:
            story.append(Paragraph("Moderate Correlations (0.5 ≤ |r| < 0.7):", subheading_style))
            for corr in moderate_corr[:5]:  # Top 5
                direction = "positive" if corr['correlation'] > 0 else "negative"
                story.append(Paragraph(f"• {corr['var1']} ↔ {corr['var2']}: {corr['correlation']:.3f} ({direction})", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    # Add Outlier Information
    if 'outliers' in analysis_results:
        story.append(Paragraph("Outlier Detection", heading_style))
        outliers = analysis_results['outliers']
        
        outlier_found = False
        for col, stats in outliers.items():
            if stats.get('iqr_outliers_count', 0) > 0:
                outlier_found = True
                story.append(Paragraph(f"• {col}: {stats['iqr_outliers_count']} outliers ({stats['iqr_outliers_percentage']:.1f}%)", styles['Normal']))
        
        if not outlier_found:
            story.append(Paragraph("No significant outliers detected in the dataset.", styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    # Add Visualizations Section
    if 'visualizations' in analysis_results:
        story.append(PageBreak())  # Start visualizations on new page
        story.append(Paragraph("Data Visualizations", heading_style))
        story.append(Spacer(1, 10))
        
        visualizations = analysis_results['visualizations']
        
        for viz in visualizations:
            # Add visualization title and description
            story.append(Paragraph(viz['title'], subheading_style))
            if viz.get('description'):
                story.append(Paragraph(viz['description'], styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Add the image if it exists
            if 'path' in viz and os.path.exists(viz['path']):
                try:
                    # Calculate image size to fit on page
                    img_width = 450  # Points (about 6.25 inches)
                    img_height = 300  # Points (about 4.17 inches)
                    
                    img = ReportLabImage(viz['path'], width=img_width, height=img_height)
                    story.append(img)
                    story.append(Spacer(1, 20))
                    
                except Exception as e:
                    story.append(Paragraph(f"Error loading visualization: {str(e)}", styles['Normal']))
                    story.append(Spacer(1, 10))
            else:
                story.append(Paragraph("Visualization file not found.", styles['Normal']))
                story.append(Spacer(1, 10))
    
    # Add Recommendations
    if 'final_report' in analysis_results and 'recommendations' in analysis_results['final_report']:
        story.append(PageBreak())  # Start recommendations on new page
        story.append(Paragraph("Key Recommendations", heading_style))
        recommendations = analysis_results['final_report']['recommendations']
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            story.append(Spacer(1, 8))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def display_analysis_results(analysis_results):
    """Display analysis results in Streamlit"""
    
    if not analysis_results:
        st.warning("No analysis results available. Please run the analysis first.")
        return
    
    # Display final report if available
    if 'final_report' in analysis_results:
        report = analysis_results['final_report']
        
        # Executive Summary
        st.markdown("## 🎯 Executive Summary")
        exec_summary = report.get('executive_summary', [])
        if exec_summary:
            for item in exec_summary:
                st.markdown(f"<div class='insight-box'>📊 {item}</div>", unsafe_allow_html=True)
        
        # Dataset Overview
        if 'dataset_overview' in report:
            st.markdown("## 📋 Dataset Overview")
            dataset_info = report['dataset_overview']
            
            col1, col2, col3 = st.columns(3)
            
            if 'shape' in dataset_info:
                shape = dataset_info['shape']
                with col1:
                    st.metric("Total Rows", f"{shape[0]:,}")
                with col2:
                    st.metric("Total Columns", f"{shape[1]}")
            
            if 'memory_usage_mb' in dataset_info:
                with col3:
                    st.metric("Memory Usage", f"{dataset_info['memory_usage_mb']:.2f} MB")
            
            # Display column information
            if 'columns' in dataset_info and 'dtypes' in dataset_info:
                st.markdown("### Column Information")
                col_df = pd.DataFrame({
                    'Column': dataset_info['columns'],
                    'Data Type': [dataset_info['dtypes'].get(col, 'Unknown') for col in dataset_info['columns']]
                })
                st.dataframe(col_df, use_container_width=True)
    
    # Missing Values Analysis
    if 'missing_values' in analysis_results:
        st.markdown("## 🔍 Missing Values Analysis")
        missing_info = analysis_results['missing_values']
        
        total_missing = missing_info.get('total_missing', 0)
        cols_with_missing = missing_info.get('columns_with_missing', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Missing Values", f"{total_missing:,}")
        with col2:
            st.metric("Columns with Missing Data", cols_with_missing)
        
        if total_missing > 0:
            missing_summary = missing_info.get('summary', [])
            if missing_summary:
                missing_df = pd.DataFrame(missing_summary)
                st.markdown("### Missing Data by Column")
                st.dataframe(missing_df, use_container_width=True)
        else:
            st.markdown("<div class='success-box'>✅ No missing values detected in the dataset!</div>", unsafe_allow_html=True)
    
    # Univariate Analysis
    if 'univariate' in analysis_results:
        st.markdown("## 📊 Univariate Analysis")
        univariate = analysis_results['univariate']
        
        # Numeric variables
        if 'numeric_analysis' in univariate and univariate['numeric_analysis']:
            st.markdown("### 🔢 Numeric Variables")
            
            numeric_data = []
            for col, stats in univariate['numeric_analysis'].items():
                numeric_data.append({
                    'Variable': col,
                    'Count': f"{stats['count']:,}",
                    'Mean': f"{stats['mean']:.2f}",
                    'Median': f"{stats['median']:.2f}",
                    'Std Dev': f"{stats['std']:.2f}",
                    'Min': f"{stats['min']:.2f}",
                    'Max': f"{stats['max']:.2f}",
                    'Skewness': f"{stats['skewness']:.2f}"
                })
            
            numeric_df = pd.DataFrame(numeric_data)
            st.dataframe(numeric_df, use_container_width=True)
        
        # Categorical variables
        if 'categorical_analysis' in univariate and univariate['categorical_analysis']:
            st.markdown("### 📝 Categorical Variables")
            
            for col, stats in univariate['categorical_analysis'].items():
                with st.expander(f"📊 {col}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Unique Values", stats['unique_count'])
                    with col2:
                        st.metric("Most Frequent", f"{stats['most_frequent']} ({stats['most_frequent_count']})")
                    
                    # Show value counts
                    if 'value_counts' in stats:
                        value_counts_df = pd.DataFrame(list(stats['value_counts'].items()), 
                                                     columns=['Value', 'Count'])
                        st.dataframe(value_counts_df, use_container_width=True)
    
    # Correlation Analysis
    if 'correlation' in analysis_results:
        st.markdown("## 🔗 Correlation Analysis")
        correlation = analysis_results['correlation']
        
        # Strong correlations
        strong_corr = correlation.get('strong_correlations', [])
        if strong_corr:
            st.markdown("### 🔴 Strong Correlations (|r| ≥ 0.7)")
            for corr in strong_corr:
                direction = "positive" if corr['correlation'] > 0 else "negative"
                st.markdown(f"<div class='warning-box'>⚠️ **{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f} ({direction})</div>", unsafe_allow_html=True)
        
        # Moderate correlations
        moderate_corr = correlation.get('moderate_correlations', [])
        if moderate_corr:
            st.markdown("### 🟡 Moderate Correlations (0.5 ≤ |r| < 0.7)")
            for corr in moderate_corr[:5]:  # Show top 5
                direction = "positive" if corr['correlation'] > 0 else "negative"
                st.markdown(f"• **{corr['var1']}** ↔ **{corr['var2']}**: {corr['correlation']:.3f} ({direction})")
        
        if not strong_corr and not moderate_corr:
            st.markdown("<div class='success-box'>✅ No strong correlations detected - good feature independence!</div>", unsafe_allow_html=True)
    
    # Outlier Analysis
    if 'outliers' in analysis_results:
        st.markdown("## 🎯 Outlier Detection")
        outliers = analysis_results['outliers']
        
        outlier_data = []
        for col, stats in outliers.items():
            if stats.get('iqr_outliers_count', 0) > 0:
                outlier_data.append({
                    'Variable': col,
                    'Outliers (IQR)': f"{stats['iqr_outliers_count']} ({stats['iqr_outliers_percentage']:.1f}%)",
                    'Outliers (Z-Score)': f"{stats['z_outliers_count']} ({stats['z_outliers_percentage']:.1f}%)",
                    'Valid Range': f"[{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]"
                })
        
        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data)
            st.dataframe(outlier_df, use_container_width=True)
        else:
            st.markdown("<div class='success-box'>✅ No significant outliers detected!</div>", unsafe_allow_html=True)
    
    # Target Analysis
    if 'target_analysis' in analysis_results:
        st.markdown("## 🎯 Target Variable Analysis")
        target_analysis = analysis_results['target_analysis']
        
        target_info = target_analysis.get('target_info', {})
        if target_info:
            col_name = target_info.get('column_name', 'Unknown')
            is_numeric = target_info.get('is_numeric', False)
            
            st.markdown(f"### Target Variable: **{col_name}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type", "Numeric" if is_numeric else "Categorical")
            with col2:
                st.metric("Missing Values", target_info.get('missing_count', 0))
            with col3:
                st.metric("Unique Values", target_info.get('unique_count', 0))
            
            if is_numeric:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean", f"{target_info.get('mean', 0):.2f}")
                with col2:
                    st.metric("Std Dev", f"{target_info.get('std', 0):.2f}")
    
    # Visualizations
    if 'visualizations' in analysis_results:
        st.markdown("## 📊 Visualizations")
        visualizations = analysis_results['visualizations']
        
        for viz in visualizations:
            with st.expander(f"📈 {viz['title']}"):
                st.markdown(f"**Description:** {viz['description']}")
                
                # Display image if it exists
                if 'path' in viz and os.path.exists(viz['path']):
                    try:
                        st.image(viz['path'], use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")
    
    # Recommendations
    if 'final_report' in analysis_results and 'recommendations' in analysis_results['final_report']:
        st.markdown("## 💡 Recommendations")
        recommendations = analysis_results['final_report']['recommendations']
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"<div class='insight-box'>{i}. {rec}</div>", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class='main-header'>
        <h1>🤖 EDA with CrewAI</h1>
        <p>Automated Exploratory Data Analysis using AI Agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🚀 Get Started")
        st.markdown("Upload your dataset and let AI agents perform comprehensive EDA!")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file for analysis"
        )
        
        # Target column selection (will be populated after file upload)
        target_column = None
        if uploaded_file is not None:
            # Read the file to get column names
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                columns = df_preview.columns.tolist()
                
                target_column = st.selectbox(
                    "Select target column (optional)",
                    options=["None"] + columns,
                    help="Select the target variable for analysis"
                )
                
                if target_column == "None":
                    target_column = None
                
                # Show preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df_preview, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Analysis button
        run_analysis = st.button("🔍 Run EDA Analysis", type="primary", use_container_width=True)
        
        # Clear results button
        if st.button("🗑️ Clear Results", use_container_width=True):
            clear_analysis_results()
            st.success("Results cleared!")
            st.experimental_rerun()
    
    # Main content area
    if uploaded_file is None:
        # Welcome message
        st.markdown("""
        ## Welcome to Automated EDA! 🎉
        
        This application uses **CrewAI** agents to perform comprehensive exploratory data analysis on your datasets.
        
        ### 🔧 What this tool does:
        
        1. **📊 Data Loading & Overview** - Analyzes dataset structure and basic statistics
        2. **🔍 Missing Value Analysis** - Identifies and summarizes missing data patterns
        3. **📈 Univariate Analysis** - Examines distribution of each variable
        4. **🔗 Correlation Analysis** - Finds relationships between variables
        5. **🎯 Outlier Detection** - Identifies anomalous data points
        6. **🎪 Target Relationships** - Analyzes relationships with target variable
        7. **📊 Visualizations** - Generates comprehensive charts and plots
        8. **📋 Report Generation** - Creates detailed PDF reports
        
        ### 🚀 Getting Started:
        
        1. Upload your CSV file using the sidebar
        2. Optionally select a target column
        3. Click "Run EDA Analysis"
        4. View results and download PDF report
        
        **Let AI agents do the heavy lifting of exploratory data analysis!** 🤖
        """)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-container'>
                <h4>🤖 AI-Powered</h4>
                <p>Uses CrewAI agents for intelligent analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-container'>
                <h4>📊 Comprehensive</h4>
                <p>8 specialized tools for complete EDA</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-container'>
                <h4>📋 Report Ready</h4>
                <p>Generate PDF reports instantly</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif uploaded_file is not None and run_analysis:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        try:
            # Run analysis
            with st.spinner("🤖 AI agents are analyzing your data... This may take a few minutes."):
                result = run_eda_analysis(temp_file_path, target_column)
            
            if result['success']:
                st.success("✅ Analysis completed successfully!")
                
                # Get analysis results
                analysis_results = get_analysis_results()
                
                # Display results
                if analysis_results:
                    display_analysis_results(analysis_results)
                    
                    # PDF download button
                    st.markdown("## 📥 Download Report")
                    
                    try:
                        pdf_buffer = create_pdf_report(analysis_results)
                        
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_buffer.getvalue(),
                            file_name="eda_report.pdf",
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"Error creating PDF: {str(e)}")
                
                else:
                    st.error("No analysis results available. Please try running the analysis again.")
            
            else:
                st.error(f"❌ Analysis failed: {result['message']}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    elif uploaded_file is not None:
        # Show existing results if available
        analysis_results = get_analysis_results()
        if analysis_results:
            st.info("📊 Previous analysis results are displayed below. Click 'Run EDA Analysis' to analyze the new file.")
            display_analysis_results(analysis_results)
            
            # PDF download button for existing results
            st.markdown("## 📥 Download Report")
            try:
                pdf_buffer = create_pdf_report(analysis_results)
                
                st.download_button(
                    label="📄 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name="eda_report.pdf",
                    mime="application/pdf",
                    type="primary",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error creating PDF: {str(e)}")
        else:
            st.info("📁 File uploaded! Click 'Run EDA Analysis' in the sidebar to start the analysis.")

if __name__ == "__main__":
    main()
