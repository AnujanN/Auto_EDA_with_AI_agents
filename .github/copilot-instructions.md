<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# EDA with CrewAI - Project Instructions

## Project Overview
This is an automated Exploratory Data Analysis (EDA) system built with CrewAI agents and Streamlit frontend. The system uses AI agents to perform comprehensive data analysis and generate insights automatically.

## Key Components

### 1. **tools.py** - 8 Specialized EDA Tools
- `load_data_tool` - Dataset loading and basic info
- `missing_value_analysis_tool` - Missing data analysis
- `univariate_analysis_tool` - Individual variable analysis
- `correlation_analysis_tool` - Variable relationships
- `outlier_detection_tool` - Anomaly detection
- `target_relationship_tool` - Target variable analysis
- `generate_visualizations_tool` - Chart generation
- `export_report_tool` - Report compilation

### 2. **crew.py** - CrewAI Orchestration
- EDAAgent: Performs analysis using the 8 tools
- StreamlitAgent: Formats output for web display
- Sequential task execution

### 3. **app.py** - Streamlit Frontend
- File upload interface
- Interactive results display
- PDF report generation
- Professional styling

### 4. **config.py** - Configuration Management
- LLM initialization (Google Gemini)
- Path management
- Environment variables

## Architecture

```
User Upload CSV → CrewAI Agents → 8 EDA Tools → Analysis Results → Streamlit Display + PDF Export
```

## Development Guidelines

1. **Tool Development**: Each tool is self-contained and stores results in `analysis_results`
2. **Agent Communication**: Sequential execution with result passing
3. **Error Handling**: Comprehensive try-catch blocks in all tools
4. **Data Storage**: Global variables for analysis state management
5. **Visualization**: Matplotlib/Seaborn for static plots, stored as images

## Key Features

- **AI-Powered**: Uses Google Gemini for intelligent analysis
- **Comprehensive**: 8-step systematic EDA process
- **Interactive**: Real-time Streamlit interface
- **Professional**: PDF report generation
- **Scalable**: Modular tool-based architecture

## Setup Requirements

1. Python 3.8+
2. Google API key for Gemini
3. Required packages in requirements.txt
4. Environment configuration in .env

## Usage Patterns

1. **Web Interface**: Upload CSV → Select target → Run analysis → View results → Download PDF
2. **Command Line**: `python main.py dataset.csv target_column`
3. **Programmatic**: Import and use `run_eda_analysis()` function

## Code Style

- Follow PEP 8 conventions
- Use type hints where appropriate
- Comprehensive docstrings for all functions
- Error messages with context
- Progress indicators for long operations

## Testing Considerations

- Test with various dataset sizes
- Validate missing data handling
- Check visualization generation
- Verify PDF creation
- Test error scenarios

## Performance Notes

- Large datasets may require chunking
- Visualization generation can be memory intensive
- PDF creation has file size limits
- Streamlit state management for session persistence
