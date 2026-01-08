import streamlit as st

st.set_page_config(
    page_title="About - ASE Insight",
    page_icon="ℹ️",
    layout="wide"
)

# Hide the default page navigation
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-bottom: 2rem;
    }
    
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">About ASE Insight</h1>', unsafe_allow_html=True)

# ========================================
# PLATFORM OVERVIEW
# ========================================
st.markdown("""
<div class="section-card">
    <div class="section-title">Platform Overview</div>
    <p style='font-size: 1.1rem; line-height: 1.8; color: #ccc;'>
        <b>ASE Insight</b> is a professional trading and analytics platform specifically designed 
        for the Amman Stock Exchange (ASE). Built with cutting-edge technology, the platform 
        combines advanced technical analysis with artificial intelligence to provide traders 
        and investors with powerful decision-making tools.
    </p>
    <br>
    <p style='font-size: 1rem; line-height: 1.8; color: #aaa;'>
        Our mission is to democratize access to professional-grade trading tools and make 
        sophisticated market analysis accessible to all Jordanian investors, from beginners 
        to experienced traders.
    </p>
</div>
""", unsafe_allow_html=True)

# ========================================
# KEY FEATURES
# ========================================
st.markdown("""
<div class="section-card">
    <div class="section-title">Key Features</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Technical Analysis
    - Real-time candlestick charts
    - 15+ technical indicators
    - Moving averages (SMA, EMA)
    - RSI, MACD, Bollinger Bands
    - Volume analysis
    - Support/resistance levels
    - AI-powered trading signals
    - Historical data export
    """)
    
    st.markdown("""
    ### Data & Analytics
    - Historical data from 30 to 730 days
    - Statistical performance metrics
    - Return calculations
    - Volatility analysis
    - Correlation studies
    - Excel/CSV export functionality
    """)

with col2:
    st.markdown("""
    ### AI Price Prediction
    - Deep Learning models (LSTM, GRU, RNN)
    - Classical time series (AR, MA, ARMA, ARIMA)
    - Custom model architecture
    - Forecast up to 30 days ahead
    - Model performance metrics
    - Model comparison tools
    - Save/load trained models
    """)
    
    st.markdown("""
    ### User Experience
    - Clean, intuitive interface
    - Real-time data updates
    - Interactive visualizations
    - Comprehensive documentation
    - Mobile-responsive design
    - Fast performance
    """)

st.markdown("---")

# ========================================
# TECHNICAL SPECIFICATIONS
# ========================================
st.markdown("## Technical Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Data Coverage
    - **Companies:** 194+ ASE listed
    - **Sectors:** 12 major sectors
    - **Historical Data:** Up to 2 years
    - **Update Frequency:** Real-time
    - **Data Source:** ASE Official
    """)

with col2:
    st.markdown("""
    ### Technology Stack
    - **Frontend:** Streamlit
    - **Charts:** Plotly
    - **AI/ML:** PyTorch, TensorFlow
    - **Data:** Pandas, NumPy
    - **APIs:** ASE Real-time API
    """)

with col3:
    st.markdown("""
    ### AI Models
    - **Deep Learning:** LSTM, GRU, RNN
    - **Classical:** AR, MA, ARMA, ARIMA
    - **Optimization:** Adam, SGD
    - **Metrics:** MSE, MAE, RMSE, R²
    - **Training:** Custom parameters
    """)

st.markdown("---")

# ========================================
# VERSION INFORMATION
# ========================================
st.markdown("## Version Information")

version_info = {
    "Version": ["2.0", "1.5", "1.0"],
    "Release Date": ["December 2024", "September 2024", "June 2024"],
    "Major Features": [
        "AI Prediction, Enhanced UI, Model Comparison",
        "Technical Dashboard, 15+ Indicators",
        "Initial Release, Basic Charts"
    ],
    "Status": ["Current", "Deprecated", "Archived"]
}

st.table(version_info)

st.markdown("---")

# ========================================
# RESOURCES & LINKS
# ========================================
st.markdown("## Resources & Links")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Official ASE Resources
    - [ASE Official Website](https://www.ase.com.jo)
    - [Trading Guide](https://www.ase.com.jo/en/page/trading-guide)
    - [Market Data](https://www.ase.com.jo/en/market-data)
    - [Historical Data](https://www.ase.com.jo/en/historical-data)
    - [Company Listings](https://www.ase.com.jo/en/companies)
    """)

with col2:
    st.markdown("""
    ### Learning Resources
    - [Technical Analysis Basics](https://www.investopedia.com/technical-analysis-4689657)
    - [Understanding Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)
    - [AI in Trading](https://www.investopedia.com/artificial-intelligence-ai-in-finance-5088543)
    - [Risk Management](https://www.investopedia.com/articles/trading/09/risk-management.asp)
    """)

st.markdown("---")

# ========================================
# DISCLAIMER
# ========================================
st.markdown("""
<div class="section-card" style="border-left-color: #ff6b6b;">
    <div class="section-title" style="color: #ff6b6b;">Important Disclaimer</div>
    <p style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        <b>ASE Insight</b> is provided for educational and informational purposes only. 
        The platform does not provide financial, investment, or trading advice. 
        All information, predictions, and signals are generated through automated 
        algorithms and should not be considered as recommendations to buy or sell securities.
    </p>
    <br>
    <p style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        <b>Users should:</b>
    </p>
    <ul style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        <li>Conduct their own research and due diligence</li>
        <li>Consult with licensed financial advisors</li>
        <li>Understand that past performance does not guarantee future results</li>
        <li>Be aware that all investments carry risk</li>
        <li>Never invest more than they can afford to lose</li>
    </ul>
    <br>
    <p style='font-size: 1rem; line-height: 1.8; color: #ccc;'>
        By using this platform, you acknowledge that you understand and accept these terms.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# CONTACT & SUPPORT
# ========================================
st.markdown("## Contact & Support")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Platform Support
    For technical issues or questions about using the platform.
    
    **Response Time:** 24-48 hours
    """)

with col2:
    st.markdown("""
    ### Feature Requests
    Have an idea for a new feature? We'd love to hear from you.
    
    **Status:** Open for suggestions
    """)

with col3:
    st.markdown("""
    ### Bug Reports
    Found a bug? Please report it so we can fix it quickly.
    
    **Priority:** High
    """)

st.markdown("---")

# ========================================
# FOOTER
# ========================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1rem;'><b>ASE Insight Platform v2.0</b></p>
    <p style='font-size: 0.9rem;'>Built with for Jordanian Traders & Investors</p>
    <p style='font-size: 0.8rem; color: #666; margin-top: 1rem;'>
        © 2025 ASE Insight. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation
if st.button("← Back to Home", use_container_width=True):
    st.switch_page("ase_insight.py")