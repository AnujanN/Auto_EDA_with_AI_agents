# 🤖 EDA with CrewAI
Automated Exploratory Data Analysis powered by AI agents.

## 🚀 Quick Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/EDA_with_crewai.git
cd EDA_with_crewai

# Create environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure API
echo "GEMINI_API_KEY=your_key_here" > .env

# Run app
streamlit run app.py
```

Open `http://localhost:8501` to start analyzing!

## 🎯 Features
- 🤖 **AI-Powered**: CrewAI agents with Google Gemini
- 📊 **Complete Analysis**: Missing values, correlations, outliers, distributions
- 📈 **Rich Visuals**: Histograms, box plots, heatmaps, density plots
- 📋 **PDF Reports**: Professional reports with charts and insights
- 🎨 **Easy Upload**: Drag-and-drop CSV interface

## 🔧 Requirements
- Python 3.8+
- Google Gemini API key