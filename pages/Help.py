import streamlit as st

st.set_page_config(
    page_title="Help & Tips - ASE Insight",
    page_icon="💡",
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
    
    .tip-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin-bottom: 1.5rem;
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ff6b6b;
        margin-bottom: 1.5rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(46, 213, 115, 0.1) 0%, rgba(46, 213, 115, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2ed573;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Help & Tips</h1>', unsafe_allow_html=True)

# ========================================
# GETTING STARTED
# ========================================
st.markdown("## Getting Started")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("### Technical Analysis Dashboard")
        st.markdown("**Perfect for:** Day traders, technical analysts, active investors")
        
        st.markdown("**How to use:**")
        st.markdown("""
        1. Select sector → subsector → company
        2. Choose analysis period (30-730 days)
        3. Enable technical indicators you want to see
        4. View interactive charts and trading signals
        5. Use the profit calculator for projections
        6. Download data for offline analysis
        """)
        
        st.markdown("**Best Practices:**")
        st.markdown("""
        - Use at least 180 days for reliable signals
        - Combine RSI + MACD + Moving Averages
        - Watch for strong signals (2/3 or 3/3 agreement)
        - Check multiple timeframes before trading
        - Always verify with volume indicators
        """)

with col2:
    with st.container():
        st.markdown("### AI Price Prediction")
        st.markdown("**Perfect for:** Long-term investors, data scientists, quant analysts")
        
        st.markdown("**How to use:**")
        st.markdown("""
        1. Select company from the dropdown menu
        2. Choose model type (Deep Learning or ARIMA)
        3. Configure model parameters
        4. Train the model on historical data
        5. View predictions and performance metrics
        6. Compare different model results
        """)
        
        st.markdown("**Best Practices:**")
        st.markdown("""
        - Use 200+ days of training data
        - Try multiple model types for comparison
        - LSTM works best for volatile stocks
        - ARIMA for stable, trending stocks
        - Validate predictions with technical analysis
        """)

st.markdown("---")

# ========================================
# TECHNICAL INDICATORS GUIDE
# ========================================
st.markdown("## Technical Indicators Guide")

tab1, tab2, tab3, tab4 = st.tabs(["Moving Averages", "Momentum", "Volatility", "Volume"])

with tab1:
    st.markdown("""
    ### Moving Averages
    
    **Simple Moving Average (SMA)**
    - Calculates average price over specified period
    - Smooths out price fluctuations
    - Good for identifying trends
    - **Tip:** Use 20-day for short-term, 50-day for medium, 200-day for long-term
    
    **Exponential Moving Average (EMA)**
    - Gives more weight to recent prices
    - More responsive to recent changes
    - Better for fast-moving markets
    - **Tip:** 12 and 26-day EMAs are commonly used for MACD
    
    **Trading Signals:**
    - Buy when price crosses above MA
    - Sell when price crosses below MA
    - Golden Cross: 50-day MA crosses above 200-day MA (bullish)
    - Death Cross: 50-day MA crosses below 200-day MA (bearish)
    """)

with tab2:
    st.markdown("""
    ### Momentum Indicators
    
    **Relative Strength Index (RSI)**
    - Measures speed and change of price movements
    - Range: 0 to 100
    - **Overbought:** Above 70 (possible sell signal)
    - **Oversold:** Below 30 (possible buy signal)
    - **Tip:** Look for divergences between RSI and price
    
    **Moving Average Convergence Divergence (MACD)**
    - Shows relationship between two moving averages
    - MACD Line = 12 EMA - 26 EMA
    - Signal Line = 9-day EMA of MACD
    - **Buy Signal:** MACD crosses above signal line
    - **Sell Signal:** MACD crosses below signal line
    - **Tip:** Confirm with histogram (difference between MACD and signal)
    """)

with tab3:
    st.markdown("""
    ### Volatility Indicators
    
    **Bollinger Bands**
    - Consists of: Middle band (20 SMA), Upper band (+2 SD), Lower band (-2 SD)
    - Shows price volatility and potential breakouts
    - **Buy Signal:** Price touches lower band
    - **Sell Signal:** Price touches upper band
    - **Squeeze:** Bands narrow = low volatility, potential breakout coming
    - **Expansion:** Bands widen = high volatility
    
    **Average True Range (ATR)**
    - Measures market volatility
    - Higher ATR = higher volatility
    - Used for setting stop-loss levels
    - **Tip:** Set stop-loss at 2x ATR from entry
    """)

with tab4:
    st.markdown("""
    ### Volume Indicators
    
    **Volume Analysis**
    - Confirms price movements
    - High volume + price increase = strong bullish signal
    - High volume + price decrease = strong bearish signal
    - Low volume = weak signal, potential reversal
    
    **On-Balance Volume (OBV)**
    - Cumulative volume indicator
    - Rising OBV = accumulation (bullish)
    - Falling OBV = distribution (bearish)
    - **Tip:** Look for OBV divergence from price
    
    **Volume Moving Average**
    - Average volume over specified period
    - Compare current volume to average
    - Above average = strong interest
    - Below average = weak interest
    """)

st.markdown("---")

# ========================================
# AI MODEL SELECTION GUIDE
# ========================================
st.markdown("## AI Model Selection Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Deep Learning Models
    
    **LSTM (Long Short-Term Memory)**
    - Best for: Complex patterns, volatile stocks
    - Pros: Handles long-term dependencies, captures trends
    - Cons: Requires more data (200+ days), slower training
    - **When to use:** For stocks with high volatility
    
    **GRU (Gated Recurrent Unit)**
    - Best for: Medium complexity, faster training
    - Pros: Similar to LSTM but faster, less parameters
    - Cons: May miss very long-term patterns
    - **When to use:** When you need quick results
    
    **RNN (Recurrent Neural Network)**
    - Best for: Short-term predictions
    - Pros: Simple, fast training
    - Cons: Struggles with long sequences
    - **When to use:** For short-term trading (1-7 days)
    """)

with col2:
    st.markdown("""
    ### Classical Models
    
    **ARIMA (AutoRegressive Integrated Moving Average)**
    - Best for: Stable stocks with clear trends
    - Pros: Statistically robust, interpretable
    - Cons: Assumes linearity, sensitive to outliers
    - **When to use:** For stable, trending stocks
    
    **AR (AutoRegressive)**
    - Best for: Price depends on past prices
    - Pros: Simple, fast
    - Cons: Limited to linear relationships
    - **When to use:** For basic trend forecasting
    
    **MA (Moving Average)**
    - Best for: Smoothing random fluctuations
    - Pros: Reduces noise
    - Cons: Lags behind actual data
    - **When to use:** For noise reduction
    """)

st.markdown("---")

# ========================================
# COMMON MISTAKES TO AVOID
# ========================================
st.markdown("## Common Mistakes to Avoid")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("### Technical Analysis Mistakes")
        st.markdown("""
        - **Using too few data points:** Always use at least 180 days
        - **Relying on single indicator:** Combine multiple indicators
        - **Ignoring volume:** Volume confirms price movements
        - **Not considering context:** Check market conditions
        - **Overtrading:** Wait for strong signals (2/3 or 3/3)
        - **No stop-loss:** Always set risk management levels
        - **Following signals blindly:** Use your judgment
        """)

with col2:
    with st.container():
        st.markdown("### AI Prediction Mistakes")
        st.markdown("""
        - **Insufficient training data:** Use 200+ days minimum
        - **Not validating models:** Always check metrics (R², MSE)
        - **Overfitting:** Don't train too many epochs
        - **Wrong model selection:** Match model to stock behavior
        - **Ignoring uncertainty:** Predictions are probabilities
        - **Not comparing models:** Test multiple approaches
        - **Trading on predictions alone:** Combine with technical analysis
        """)

st.markdown("---")

# ========================================
# BEST PRACTICES
# ========================================
st.markdown("## Best Practices for Success")

st.markdown("### Risk Management")
st.markdown("""
- Never invest more than you can afford to lose
- Diversify your portfolio across different sectors
- Set stop-loss orders for every trade
- Use position sizing (don't put all money in one stock)
- Keep 20-30% cash reserve for opportunities
- Review and rebalance portfolio regularly
""")

st.markdown("### Research & Analysis")
st.markdown("""
- Combine technical and fundamental analysis
- Read company financial reports
- Follow market news and trends
- Understand the company's business model
- Check sector performance and outlook
- Monitor economic indicators
- Keep a trading journal to learn from experience
""")

st.markdown("### Platform Usage Tips")
st.markdown("""
- Start with Technical Dashboard to understand stock behavior
- Use AI Prediction for longer-term outlook
- Export data for deeper offline analysis
- Compare multiple stocks before deciding
- Use profit calculator to set realistic goals
- Check platform regularly for updated signals
- Experiment with different indicator combinations
""")

st.markdown("---")

# ========================================
# FAQ
# ========================================
st.markdown("## Frequently Asked Questions")

with st.expander("How accurate are the AI predictions?"):
    st.markdown("""
    AI predictions are based on historical patterns and provide probability estimates, not certainties. 
    Accuracy varies by stock and market conditions. Always check the model's R² score and error metrics. 
    Typical accuracy ranges from 70-85% for short-term predictions (1-7 days) and decreases for longer 
    periods. Always combine AI predictions with technical analysis and fundamental research.
    """)

with st.expander("What timeframe should I use for analysis?"):
    st.markdown("""
    - **Day Trading:** 30-90 days with hourly/daily charts
    - **Swing Trading:** 90-180 days with daily charts
    - **Position Trading:** 180-365 days with daily/weekly charts
    - **Long-term Investing:** 365-730 days with weekly/monthly charts
    
    For AI models, use at least 200 days for reliable training.
    """)

with st.expander("Which technical indicators should I use together?"):
    st.markdown("""
    Recommended combinations:
    
    1. **Trend Following:** SMA(20) + SMA(50) + MACD
    2. **Momentum Trading:** RSI + MACD + Volume
    3. **Volatility Breakout:** Bollinger Bands + ATR + Volume
    4. **Conservative:** SMA(50) + SMA(200) + RSI + Volume
    
    Always use at least 3 different indicator types for confirmation.
    """)

with st.expander("How do I interpret trading signals?"):
    st.markdown("""
    The platform provides consensus signals based on multiple indicators:
    
    - **Strong Buy (3/3):** All indicators agree - high confidence
    - **Buy (2/3):** Majority agree - moderate confidence
    - **Hold (Mixed):** Indicators disagree - wait for clearer signal
    - **Sell (2/3):** Majority bearish - consider selling
    - **Strong Sell (3/3):** All bearish - high confidence sell
    
    Never trade on signals alone - always verify with your own analysis.
    """)

with st.expander("What's the best AI model to use?"):
    st.markdown("""
    It depends on the stock characteristics:
    
    - **Volatile stocks (banking, tech):** LSTM or GRU
    - **Stable stocks (utilities, dividends):** ARIMA
    - **Trending stocks:** ARIMA or AR
    - **Quick predictions:** GRU or RNN
    
    Best practice: Try multiple models and compare their R² scores and error metrics.
    """)

with st.expander("Can I download the data for my own analysis?"):
    st.markdown("""
    Yes! The Technical Dashboard allows you to download:
    
    - Full historical price data (Excel/CSV)
    - Technical indicator values
    - Volume data
    - Calculated statistics
    
    Click the "Download Data" button in the dashboard to export.
    """)

st.markdown("---")

# ========================================
# TROUBLESHOOTING
# ========================================
st.markdown("## Troubleshooting")

st.markdown("""
### Common Issues and Solutions

**Issue: Charts not loading**
- Check your internet connection
- Refresh the page
- Clear browser cache
- Try a different browser

**Issue: AI model training fails**
- Ensure you have at least 200 days of data
- Reduce the number of epochs
- Try a simpler model (GRU instead of LSTM)
- Check for missing data in the selected period

**Issue: Indicators showing conflicting signals**
- This is normal - not all indicators agree
- Look for 2/3 or 3/3 consensus
- Check multiple timeframes
- Consider market context

**Issue: Slow performance**
- Close unnecessary browser tabs
- Reduce the data period
- Disable unused indicators
- Clear browser cache

**Issue: Can't find a specific company**
- Check the sector/subsector selection
- Company may be delisted
- Try searching by company code
""")

st.markdown("---")

# ========================================
# FOOTER
# ========================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1rem;'>Need more help? Check our <b>About</b> page for detailed information</p>
    <p style='font-size: 0.9rem;'>or visit the <a href='https://www.ase.com.jo' target='_blank'>ASE Official Website</a></p>
</div>
""", unsafe_allow_html=True)

# Navigation
if st.button("← Back to Home", use_container_width=True):
    st.switch_page("ase_insight.py")