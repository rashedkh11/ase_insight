



import streamlit as st
from google import genai

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Stock Market Assessment Assistant",
    page_icon="📈",
    layout="centered"
)

# --------------------------------------------------
# ⚠️ DIRECT API KEY (NO ENV VARIABLES)
# --------------------------------------------------
GEMINI_API_KEY = "AIzaSyABZpAFuvry5bUkEwzOMuyUoP5rYsdpIvM"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #888;
        margin-bottom: 2rem;
    }
    
    
    
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(135deg, #008000 0%, #00000 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
   
    
</style>
""", unsafe_allow_html=True)# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown('<h1 class="main-title">Smart Stock Market Assistant</h1>', unsafe_allow_html=True)
st.markdown("---")

# --------------------------------------------------
# SYSTEM PROMPT
# --------------------------------------------------
SYSTEM_PROMPT = """
You are a professional financial market analyst.

You specialize in:
- Amman Stock Exchange (ASE)
- Global stock markets
- Technical analysis
- Fundamental analysis
- Macroeconomics
- Risk management
- Forex and cryptocurrencies

Behavior rules:
1. If the user's question is related to the Amman Stock Exchange (ASE),
   answer with detailed ASE-specific analysis.
2. If the question is NOT related to ASE,
   answer it normally using global market knowledge.
3. NEVER refuse a question because it is not about ASE.
4. Use clear, professional, educational language.
5. Do NOT provide guaranteed profits.
6. Always mention risks when discussing investments.
"""

# --------------------------------------------------
# EXTENDED ASE KNOWLEDGE
# --------------------------------------------------
ASE_CONTEXT = """
Amman Stock Exchange (ASE) detailed knowledge:

- Country: Jordan
- Market Type: Emerging Market
- Currency: Jordanian Dinar (JOD)
- Main Index: ASE General Index
- Trading Days: Sunday to Thursday
- Trading Hours: 10:00 AM – 12:00 PM local time

Market Characteristics:
- Low to medium liquidity
- Price movements can be sharp due to low volume
- Strong impact from institutional investors
- Sensitive to:
  - Interest rates
  - Government policies
  - Regional geopolitical events
  - Corporate earnings

Main Sectors:
- Banking (dominant sector)
- Insurance
- Mining & Extraction (phosphate, potash)
- Industrial
- Services

Common Investor Profile:
- Long-term investors
- Dividend-focused strategies
- Conservative risk tolerance

Technical Analysis Notes for ASE:
- Indicators work best on daily/weekly charts
- RSI, MACD, Moving Averages commonly used
- False breakouts are common due to low liquidity
"""

# --------------------------------------------------
# ANSWER STRUCTURE (ONLY WHEN ANALYSIS IS NEEDED)
# --------------------------------------------------
ANALYSIS_STRUCTURE = """
When giving market or investment analysis, use this structure when appropriate:

- Market / Asset Overview:
- Current Trend:
- Key Supporting Factors:
- Risks & Weaknesses:
- Technical Perspective (if applicable):
- Suitable Investment Horizon:
- Conclusion:
"""
# --------------------------------------------------
# LOAD GEMINI CLIENT
# --------------------------------------------------
@st.cache_resource
def load_gemini():
    return genai.Client(api_key=GEMINI_API_KEY)

client = load_gemini()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# DISPLAY CHAT HISTORY
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------
if user_prompt := st.chat_input(
    "Ask about ASE, global stocks, crypto, forex, or trading strategies..."
):

    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):

            final_prompt = f"""
{SYSTEM_PROMPT}

{ASE_CONTEXT}

{ANALYSIS_STRUCTURE}

User Question:
{user_prompt}
"""

            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=final_prompt
                )

                st.markdown(response.text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.text}
                )

            except Exception as e:
                st.error(f"Error: {e}")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Settings")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Quick Questions")

    quick_questions = {
        "ASE Outlook": "Assess the current outlook of the Amman Stock Exchange",
        "Banking Sector": "Analyze the banking sector in ASE",
        "Technical Indicators": "Explain RSI and MACD",
        "Risk Management": "Best risk management practices in trading",
        "Global Markets": "How do interest rates affect stock markets?"
    }

    for label, question in quick_questions.items():
        if st.button(label, use_container_width=True):

            st.session_state.messages.append(
                {"role": "user", "content": question}
            )

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"{SYSTEM_PROMPT}\n{ASE_CONTEXT}\n{ANALYSIS_STRUCTURE}\n{question}"
            )

            st.session_state.messages.append(
                {"role": "assistant", "content": response.text}
            )

            st.rerun()

    st.markdown("---")
    st.info("""
**Model:** gemini-3-flash-preview
**Backend:** Google Gemini API
""")

    st.warning("""
 Disclaimer  
This assistant provides educational information only.  
It is NOT financial advice.
""")
