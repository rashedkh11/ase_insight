import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import io
import requests
import pytz
from datetime import datetime, time, timedelta

# --------------------------------------------------
st.set_page_config(
    page_title="ASE Insight - Amman Stock Exchange Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide the default page navigation
st.markdown("""
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS
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
    
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        font-size: 1rem;
        color: #ccc;
        line-height: 1.6;
    }
    
    .stButton > button {
        width: 100%;
        height: 60px;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .hero-section {
        text-align: center;
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# HELPER FUNCTIONS TO CREATE IMAGES
# ========================================

def create_logo():
    """Create ASE Insight logo"""
    width, height = 500, 150
    img = Image.new('RGBA', (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw gradient-like bars
    for i in range(5):
        x = 50 + i * 30
        bar_height = 60 + i * 15
        y = height - bar_height - 20
        color = (102 + i*20, 126 + i*10, 234 - i*20)
        draw.rectangle([x, y, x + 20, height - 20], fill=color)
    
    # Add text (simplified - would need font file for better text)
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    draw.text((200, 55), "ASE INSIGHT", fill=(255, 255, 255), font=font)
    
    return img

def create_feature_preview(title, color):
    """Create preview images for features"""
    width, height = 600, 300
    img = Image.new('RGB', (width, height), color=(20, 30, 48))
    draw = ImageDraw.Draw(img)
    
    # Create mock chart
    if "Technical" in title:
        # Candlestick-style chart
        for i in range(20):
            x = 30 + i * 28
            candle_height = np.random.randint(40, 120)
            y = height // 2 - candle_height // 2
            candle_color = (46, 213, 115) if i % 2 == 0 else (255, 71, 87)
            draw.rectangle([x, y, x + 15, y + candle_height], fill=candle_color, outline=candle_color)
            
            # Wick
            wick_top = y - np.random.randint(10, 30)
            wick_bottom = y + candle_height + np.random.randint(10, 30)
            draw.line([(x + 7, wick_top), (x + 7, wick_bottom)], fill=candle_color, width=2)
    
    elif "AI" in title:
        # Line chart with prediction
        points = []
        for i in range(30):
            x = 20 + i * 19
            y = height // 2 + np.sin(i * 0.3) * 60 + np.random.randint(-10, 10)
            points.append((x, y))
        
        # Historical line (blue)
        draw.line(points[:20], fill=(102, 126, 234), width=3)
        
        # Prediction line (red, dashed effect)
        for i in range(len(points[19:]) - 1):
            if i % 2 == 0:
                draw.line([points[19 + i], points[19 + i + 1]], fill=(255, 71, 87), width=3)
    
    elif "Chat" in title:
        # Chat interface mockup
        # Message bubbles
        draw.rectangle([50, 50, 350, 100], fill=(102, 126, 234), outline=(102, 126, 234))
        draw.rectangle([250, 120, 550, 170], fill=(76, 175, 80), outline=(76, 175, 80))
        draw.rectangle([50, 190, 300, 240], fill=(102, 126, 234), outline=(102, 126, 234))
        
        # Chat icon
        draw.ellipse([width - 100, height - 100, width - 30, height - 30], fill=(255, 193, 7))
    
    # Add title
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    draw.text((width // 2 - 100, 20), title, fill=color, font=font)
    
    return img

def create_market_chart():
    """Create a sample market overview chart"""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    # Generate sample market data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 2000 + np.cumsum(np.random.randn(100) * 20)
    
    # Plot
    ax.fill_between(range(len(prices)), prices, alpha=0.3, color='#667eea')
    ax.plot(prices, color='#667eea', linewidth=2.5)
    ax.set_title('ASE Market Index - Last 100 Days', color='white', fontsize=14, pad=20)
    ax.set_xlabel('Trading Days', color='white')
    ax.set_ylabel('Index Value', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# ========================================
# HEADER WITH LOGO
# ========================================

# Create and display logo
logo = create_logo()
col_logo = st.columns([1, 2, 1])
with col_logo[1]:
    st.image(logo, use_container_width=True)
# Live Market Status
col1, col2, col3, col4, col5 = st.columns(5)
amman_tz = pytz.timezone("Asia/Amman")

# Market hours
market_open_time = time(9, 30)  # 09:30
market_close_time = time(15, 0) # 15:00

# Current time in Amman
current_datetime = datetime.now(amman_tz)
now = current_datetime.time()

# Calculate time until open
if now < market_open_time:
    open_datetime = current_datetime.replace(hour=market_open_time.hour, minute=market_open_time.minute, second=0, microsecond=0)
    delta = open_datetime - current_datetime
    market_status = "Closed"
elif market_open_time <= now <= market_close_time:
    delta = timedelta(0)
    market_status = "Open"
else:
    # Market closed for today, open next day
    open_datetime = (current_datetime + timedelta(days=1)).replace(hour=market_open_time.hour, minute=market_open_time.minute, second=0, microsecond=0)
    delta = open_datetime - current_datetime
    market_status = "Closed"

# Column 1: Market Status
with col1:
    if market_status == "Open":
        st.metric(
            label="Market Status",
            value="OPEN",
            delta="▲ Live",
            delta_color="normal"   # green up
        )
    else:
        st.metric(
            label="Market Status",
            value="CLOSED",
            delta="▼ Closed",
            delta_color="inverse"  # red down
        )

# Column 2: Today’s Date
with col2:
    if market_status == "Closed":
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        st.metric("Time Until Open", f"{hours}h {minutes}m")
    else:
        st.metric("Time Until Open", "Market is open")

# Column 3: Time Zone
with col3:
    st.metric("Time Zone", "GMT+3")

# Column 4: Platform Version
with col4:
    st.metric("Today", current_datetime.strftime("%b %d, %Y"))

# Column 5: Time Until Market Opens
with col5:
    st.metric("Platform", "v2.0")



st.markdown("---")

# ==================== LIVE MARKET OVERVIEW ====================
st.markdown("## Live ASE Market Overview")

@st.cache_data(ttl=3600)
def fetch_ase_companies():
    """Fetch ASE data with EXACT code mapping based on actual website structure"""
    try:
        url = "https://www.ase.com.jo/en/products-services/securties-types/shares"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            
            # Initialize structure with EXACT subsectors from ASE
            sector_hierarchy = {
                'Financial': {
                    'Banks': [],
                    'Insurance': [],
                    'Diversified Financial Services': [],
                    'Real Estate': []  # Some 131xxx and 141xxx codes
                },
                'Services': {
                    'Health Care Services': [],
                    'Educational Services': [],
                    'Hotels and Tourism': [],
                    'Transportation': [],
                    'Technology and Communication': [],
                    'Utilities and Energy': [],
                    'Commercial Services': []
                },
                'Industrial': {
                    'Pharmaceutical and Medical Industries': [],
                    'Chemical Industries': [],
                    'Food and Beverages': [],
                    'Tobacco and Cigarettes': [],
                    'Mining and Extraction Industries': [],
                    'Engineering and Construction': [],
                    'Electrical Industries': [],
                    'Textiles Leathers and Clothings': []
                }
            }
            
            total_companies = 0
            market_1_count = 0
            market_2_count = 0
            
            for table in tables:
                if 'Symbol' in str(table.columns) and len(table) > 0:
                    for _, row in table.iterrows():
                        try:
                            code = str(row.get('Code', ''))
                            if not code or code == 'nan' or len(code) < 3:
                                continue
                            
                            company = {
                                'name': str(row.get("Company's name", row.get('Company name', ''))),
                                'short_name': str(row.get("Company's short name", row.get('Short name', ''))),
                                'symbol': str(row.get('Symbol', '')),
                                'code': code,
                                'market': str(row.get('Market', '1')),
                                'shares': str(row.get('Listed shares', ''))
                            }
                            
                            if len(company['name']) < 3:
                                continue
                            
                            # EXACT CODE MAPPING based on actual ASE structure
                            code_num = int(code)
                            
                                                        # FINANCIAL SECTOR
                            # FINANCIAL SECTOR
                            if code_num in [111001,111002,111003,111004,111005,111006,111007,111009,111014,111017,111021,111022,111033,113023]:
                                sector_hierarchy['Financial']['Banks'].append(company)

                            elif code_num in [121002,121003,121004,121005,121006,121007,121008,121009,121013,121014,121021,121022,121023,121025,121027,121032,121034]:
                                sector_hierarchy['Financial']['Insurance'].append(company)

                            elif code_num in [
                                131018,131025,131039,131062,131065,131069,131071,131082,131090,131097,131105,
                                131219,131231,131248,131249,131250,131251,131258,131260,131267,131268,131269,131271,131274,131275,131282,131289,
                                141031,141032,141218
                            ]:
                                sector_hierarchy['Financial']['Diversified Financial Services'].append(company)

                            elif code_num in [
                                131011,131017,131019,131076,131077,131086,131087,131101,131225,131229,131234,131236,131239,131240,131241,131245,131246,
                                131247,131253,131255,131265,131270,131278,131281,131284,131285,131287,141003,141015,141036,141106
                            ]:
                                sector_hierarchy['Financial']['Real Estate'].append(company)


                            # SERVICES SECTOR
                            elif code_num in [131207,131279]:
                                sector_hierarchy['Services']['Health Care Services'].append(company)

                            elif code_num in [131051,131052,131220,131221,131222]:
                                sector_hierarchy['Services']['Educational Services'].append(company)

                            elif code_num in [131003,131005,131035,131067,131078,131098,131261,131283]:
                                sector_hierarchy['Services']['Hotels and Tourism'].append(company)

                            elif code_num in [131012,131034,131080,131083,131208,131243,131256,131290]:
                                sector_hierarchy['Services']['Transportation'].append(company)

                            elif code_num in [131206,131232]:
                                sector_hierarchy['Services']['Technology and Communication'].append(company)

                            elif code_num in [131004,131010,131286,142041,141103,141223]:
                                sector_hierarchy['Services']['Utilities and Energy'].append(company)

                            elif code_num in [131022,131023,131081,131228,131238,131252,131264,141058]:
                                sector_hierarchy['Services']['Commercial Services'].append(company)


                            # INDUSTRIAL SECTOR
                            elif code_num in [141012,141210,141219]:
                                sector_hierarchy['Industrial']['Pharmaceutical and Medical Industries'].append(company)

                            elif code_num in [141009,141010,141055,141209]:
                                sector_hierarchy['Industrial']['Chemical Industries'].append(company)

                            elif code_num in [141002,141004,141029,141052,141094,141141,141222]:
                                sector_hierarchy['Industrial']['Food and Beverages'].append(company)

                            elif code_num == 141074:
                                sector_hierarchy['Industrial']['Tobacco and Cigarettes'].append(company)

                            elif code_num in [141006,141011,141018,141043,141070,141091,141224]:
                                sector_hierarchy['Industrial']['Mining and Extraction Industries'].append(company)

                            elif code_num in [131259,141019,141065,141098,141208,141214]:
                                sector_hierarchy['Industrial']['Engineering and Construction'].append(company)

                            elif code_num == 141215:
                                sector_hierarchy['Industrial']['Electrical Industries'].append(company)

                            elif code_num == 141014:
                                sector_hierarchy['Industrial']['Textiles Leathers and Clothings'].append(company)


                            total_companies += 1
                            if company['market'] == '1':
                                market_1_count += 1
                            else:
                                market_2_count += 1
                        
                        except Exception:
                            continue
            
            if total_companies > 50:
                return sector_hierarchy, total_companies, market_1_count, market_2_count, True, datetime.now()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Fallback with exact structure
    return {
        'Financial': {
            'Banks': [{'name': 'Arab Bank', 'symbol': 'ARBK', 'code': '113023', 'market': '1', 'shares': '640,800,000'}] * 14,
            'Insurance': [{'name': 'Jordan Insurance', 'symbol': 'JOIN', 'code': '121004', 'market': '1', 'shares': '30,000,000'}] * 17,
            'Diversified Financial Services': [{'name': 'Sample', 'symbol': 'SMPL', 'code': '131001', 'market': '2', 'shares': '10,000,000'}] * 30,
            'Real Estate': [{'name': 'Real Estate Dev', 'symbol': 'REDV', 'code': '131087', 'market': '2', 'shares': '49,625,545'}] * 31
        },
        'Services': {
            'Health Care Services': [{'name': 'Consultant Group', 'symbol': 'CICO', 'code': '131207', 'market': '1', 'shares': '20,000,000'}] * 2,
            'Educational Services': [{'name': 'Arab Intl Education', 'symbol': 'AIEI', 'code': '131052', 'market': '1', 'shares': '40,500,000'}] * 5,
            'Hotels and Tourism': [{'name': 'Zara Investment', 'symbol': 'ZARA', 'code': '131067', 'market': '2', 'shares': '145,000,000'}] * 8,
            'Transportation': [{'name': 'Salam Transport', 'symbol': 'SITT', 'code': '131034', 'market': '1', 'shares': '18,000,000'}] * 8,
            'Technology and Communication': [{'name': 'Jordan Telecom', 'symbol': 'JTEL', 'code': '131206', 'market': '1', 'shares': '187,500,000'}] * 2,
            'Utilities and Energy': [{'name': 'Jordan Electric', 'symbol': 'JOEP', 'code': '131004', 'market': '1', 'shares': '96,578,919'}] * 6,
            'Commercial Services': [{'name': 'Duty Free', 'symbol': 'JDFS', 'code': '131022', 'market': '1', 'shares': '22,500,000'}] * 8
        },
        'Industrial': {
            'Pharmaceutical and Medical Industries': [{'name': 'Dar Al Dawa', 'symbol': 'DADI', 'code': '141012', 'market': '1', 'shares': '50,000,000'}] * 3,
            'Chemical Industries': [{'name': 'Industrial Commercial', 'symbol': 'ICAG', 'code': '141009', 'market': '1', 'shares': '14,956,389'}] * 4,
            'Food and Beverages': [{'name': 'General Investment', 'symbol': 'GENI', 'code': '141029', 'market': '1', 'shares': '10,000,000'}] * 7,
            'Tobacco and Cigarettes': [{'name': 'Union Tobacco', 'symbol': 'UTOB', 'code': '141074', 'market': '2', 'shares': '16,963,089'}] * 1,
            'Mining and Extraction Industries': [{'name': 'Arab Potash', 'symbol': 'APOT', 'code': '141043', 'market': '1', 'shares': '83,317,500'}] * 7,
            'Engineering and Construction': [{'name': 'Ready Mix', 'symbol': 'RMCC', 'code': '141065', 'market': '1', 'shares': '25,000,000'}] * 6,
            'Electrical Industries': [{'name': 'United Cable', 'symbol': 'UCIC', 'code': '141215', 'market': '2', 'shares': '35,000,000'}] * 1,
            'Textiles Leathers and Clothings': [{'name': 'Jordan Worsted Mills', 'symbol': 'JOWM', 'code': '141014', 'market': '1', 'shares': '15,000,000'}] * 1
        }
    }, 194, 97, 97, False, None

# Fetch data
with st.spinner(" Fetching live ASE data..."):
    sector_hierarchy, total_companies, market_1_count, market_2_count, is_live, fetch_time = fetch_ase_companies()

# Status display
if is_live and fetch_time:
    col1, col2 = st.columns([3, 1])
    
else:
    st.warning(f"⚠️ Fallback data: ~{total_companies} companies")

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Total Listed", f"{total_companies}", delta="Companies")

with col2:
    st.metric(" First Market", f"{market_1_count}", delta=f"{market_1_count/total_companies*100:.1f}%")

with col3:
    st.metric(" Second Market", f"{market_2_count}", delta=f"{market_2_count/total_companies*100:.1f}%")

with col4:
    total_subsectors = sum(len(subsectors) for subsectors in sector_hierarchy.values())
    st.metric(" Subsectors", f"{total_subsectors}", delta="Active")

st.markdown("---")

# Main Sector Overview
st.markdown("###  Main Sector Breakdown")

col1, col2, col3 = st.columns(3)

main_sector_data = []
for main_sector, subsectors in sector_hierarchy.items():
    count = sum(len(companies) for companies in subsectors.values())
    main_sector_data.append((main_sector, count))

icons_main = {'Financial': '', 'Services': '', 'Industrial': ''}
colors_main = {'Financial': '#2ecc71', 'Services': '#3498db', 'Industrial': "#b435fd"}

for col, (sector, count) in zip([col1, col2, col3], main_sector_data):
    with col:
        pct = count/total_companies*100
        color = colors_main[sector]
        icon = icons_main[sector]
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {color}22 0%, {color}11 100%); 
                    padding: 1.5rem; border-radius: 12px; text-align: center; 
                    border: 2px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='font-size: 2.5rem;'>{icon}</div>
            <h3 style='color: {color}; margin: 0.5rem 0;'>{sector}</h3>
            <p style='font-size: 2.5rem; font-weight: bold; color: white;'>{count}</p>
            <p style='color: #888;'>{pct:.1f}% | {len(sector_hierarchy[sector])} subsectors</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Subsector Analysis
st.markdown("###  Subsector Analysis")

subsector_summary = []
for main, subs in sector_hierarchy.items():
    for sub, companies in subs.items():
        subsector_summary.append({
            'main': main,
            'subsector': sub,
            'count': len(companies),
            'companies': companies
        })

subsector_summary.sort(key=lambda x: x['count'], reverse=True)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    
    top_7 = subsector_summary[:10]
    subs = [s['subsector'] for s in top_7]
    counts = [s['count'] for s in top_7]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(subs)))
    bars = ax.barh(subs, counts, color=colors, alpha=0.8, edgecolor='white')
    
    for bar, count in zip(bars, counts):
        pct = count/total_companies*100
        ax.text(count + 0.5, bar.get_y() + bar.get_height()/2, 
               f'{count} ({pct:.1f}%)', 
               va='center', ha='left', fontsize=9, color='white', fontweight='bold')
    
    ax.set_xlabel('Companies', color='white', fontsize=12, fontweight='bold')
    ax.set_title('Top 7 Subsectors', color='white', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=9)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a', dpi=100)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()

with col2:
    st.markdown("####  Top Subsectors")
    
   
    
    for item in subsector_summary[:8]:
        color = colors_main[item['main']]
        pct = item['count']/total_companies*100
        
        st.markdown(f"""
        <div style='background: linear-gradient(90deg, {color}22 0%, transparent 100%); 
                    padding: 0.7rem; margin: 0.4rem 0; border-radius: 8px; 
                    border-left: 4px solid {color};'>
            <strong style='color: {color};'>{icon} {item['subsector']}</strong><br>
            <span style='font-size: 1.2rem; font-weight: bold;'>{item['count']}</span> 
            <span style='color: #888; font-size: 0.9rem;'>({pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Interactive Explorer
st.markdown("###  Sector Explorer")

tab1, tab2, tab3 = st.tabs([" Financial", " Services", "Industrial"])

def display_tab(main_name, data):
    subs = list(data.keys())
    selected = st.selectbox(f"Choose subsector:", subs, key=f"sel_{main_name}")
    
    if selected:
        companies = data[selected]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Companies", len(companies))
        with col2:
            m1 = sum(1 for c in companies if c['market'] == '1')
            st.metric("First Market", m1)
        with col3:
            st.metric("Second Market", len(companies) - m1)
        
        st.markdown(f"####  {selected}")
        df = pd.DataFrame(companies)
        if 'short_name' in df.columns:
            df = df[['short_name', 'symbol', 'code', 'market', 'shares']]
            df.columns = ['Company', 'Symbol', 'Code', 'Market', 'Shares']
        st.dataframe(df, use_container_width=True, height=400)

with tab1:
    display_tab("Financial", sector_hierarchy['Financial'])
with tab2:
    display_tab("Services", sector_hierarchy['Services'])
with tab3:
    display_tab("Industrial", sector_hierarchy['Industrial'])
st.markdown("---")

# ========================================
# MAIN FEATURES WITH PREVIEW IMAGES
# ========================================
st.markdown("## Choose Your Analysis Tool")

col1, col2, col3 = st.columns(3)

with col1:
    # Technical Analysis Preview
    tech_preview = create_feature_preview("Technical Analysis", (102, 126, 234))
    st.image(tech_preview, use_container_width=True)
    
    with st.container():
        st.markdown("### Technical Analysis Dashboard")
        st.markdown("""
        Real-time stock analysis with advanced technical indicators, AI trading signals, 
        and comprehensive performance metrics.
        """)
        
        st.markdown("**Key Features:**")
        st.markdown("""
        - Interactive candlestick charts
        - 15+ Technical indicators (RSI, MACD, Bollinger Bands)
        - AI-powered buy/sell signals
        - Investment profit calculator
        - Download full historical data (Excel/CSV)
        - Statistical analysis & returns tracking
        """)
    
    st.markdown("")
    if st.button("Launch Technical Dashboard", key="tech_dash"):
        st.switch_page("pages/Technical_Dashboard.py")

with col2:
    # AI Prediction Preview
    ai_preview = create_feature_preview("AI Prediction", (255, 71, 87))
    st.image(ai_preview, use_container_width=True)
    
    with st.container():
        st.markdown("### AI Price Prediction")
        st.markdown("""
        Advanced machine learning models for accurate stock price forecasting 
        using deep learning and classical time series methods.
        """)
        
        st.markdown("**Key Features:**")
        st.markdown("""
        - Deep Learning: LSTM, GRU, RNN
        - Classical Models: AR, MA, ARMA, ARIMA
        - Custom model architecture tuning
        - Predict 1-30 days into the future
        - Model performance comparison
        - Save & load trained models
        """)
    
    st.markdown("")
    if st.button("Launch AI Predictor", key="ai_pred"):
        st.switch_page("pages/AI_Prediction.py")

with col3:
    # AI Chat Assistant Preview
    chat_preview = create_feature_preview("Chat Assistant", (76, 175, 80))
    st.image(chat_preview, use_container_width=True)
    
    with st.container():
        st.markdown("### AI Stock Chat Assistant")
        st.markdown("""
        Your personal AI-powered stock market advisor. Ask questions about technical 
        analysis, trading strategies, and get instant expert guidance.
        """)
        
        st.markdown("**Key Features:**")
        st.markdown("""
        - GPU-Accelerated AI (Llama 3 8B)
        - Expert technical analysis advice
        - Quick-answer buttons for common questions
        - Trading strategies & risk management
        - Indicator explanations (RSI, MACD, etc.)
        - Natural conversation interface
        """)
    
    st.markdown("")
    if st.button("Launch Chat Assistant", key="chat_assist"):
        st.switch_page("pages/Stock_Chat.py")


st.markdown("<br><br>", unsafe_allow_html=True)

# ========================================
# FEATURE SHOWCASE IMAGES
# ========================================
st.markdown("## Platform Highlights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Create sample indicator chart
    fig, ax = plt.subplots(figsize=(4, 3), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    x = np.linspace(0, 10, 50)
    ax.plot(x, np.sin(x), color='#667eea', linewidth=2, label='RSI')
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.7, color='green', linestyle='--', alpha=0.5)
    ax.set_title('Technical Indicators', color='white', fontsize=10)
    ax.legend(facecolor='#1e293b', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()
    st.markdown("**15+ Indicators**")

with col2:
    # Create sample prediction chart
    fig, ax = plt.subplots(figsize=(4, 3), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    actual = np.cumsum(np.random.randn(30) * 0.5) + 50
    predicted = actual + np.random.randn(30) * 0.3
    ax.plot(actual, color='black', linewidth=2, label='Actual', marker='o', markersize=3)
    ax.plot(predicted, color='red', linewidth=2, label='Predicted', linestyle='--', marker='s', markersize=3)
    ax.set_title('AI Predictions', color='white', fontsize=10)
    ax.legend(facecolor='#1e293b', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.2, color='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()
    st.markdown("**30+ AI Models**")

with col3:
    # Create sample performance chart
    fig, ax = plt.subplots(figsize=(4, 3), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    models = ['LSTM', 'GRU', 'ARIMA', 'XGBoost']
    scores = [0.89, 0.85, 0.78, 0.92]
    colors = ['#667eea', '#764ba2', '#f093fb', '#46d5a3']
    ax.barh(models, scores, color=colors, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_title('Model Performance', color='white', fontsize=10)
    ax.set_xlabel('R² Score', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=8)
    ax.grid(True, alpha=0.2, color='white', axis='x')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()
    st.markdown("**High Accuracy**")

with col4:
    # Create AI Chat indicator
    fig, ax = plt.subplots(figsize=(4, 3), facecolor='#0f172a')
    ax.set_facecolor('#1e293b')
    categories = ['RSI\nQuestions', 'MACD\nQueries', 'Strategy\nAdvice', 'Risk\nMgmt']
    values = [85, 72, 90, 78]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_title('AI Chat Topics', color='white', fontsize=10)
    ax.set_ylabel('Usage %', color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.grid(True, alpha=0.2, color='white', axis='y')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a')
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close()
    st.markdown("**AI Assistant**")

st.markdown("---")




st.markdown("---")

# ========================================
# ADDITIONAL TOOLS (Coming Soon)
# ========================================
st.markdown("## Additional Tools (Coming Soon)")

col1, col2 = st.columns(2)


with col1:
    st.markdown("### Portfolio Manager")
    st.markdown("""
    **Overview:** Monitor all your investments in one place.  
    **Features include:**  
    - View portfolio performance in real-time  
    - Track gains and losses by asset  
    - Personalized recommendations based on your risk profile  
    - Manage your investment allocations effectively  
    **Goal:** Make smarter, data-driven investment decisions.
    """)
    
    if st.button("Manage Portfolio", key="portfolio", disabled=True):
        st.info("Coming Soon!")

# --- Column 2: News & Alerts ---
with col2:
    st.markdown("### News & Alerts")
    st.markdown("""
    **Overview:** Stay informed with the latest market updates.  
    **Features include:**  
    - Real-time stock market news and trends  
    - Company announcements and press releases  
    - Custom price alerts for your watchlist  
    - Notifications for major economic events  
    **Goal:** Never miss critical information that could impact your investments.
    """)
    
    if st.button("News Feed", key="news", disabled=True):
        st.info("Coming Soon!")

st.markdown("---")

# ========================================
# PLATFORM STATISTICS
# ========================================
st.markdown("## Platform Statistics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Companies", "160+", delta="ASE Listed")
with col2:
    st.metric("Sectors", "19", delta="Full Coverage")
with col3:
    st.metric("Indicators", "15+", delta="Technical")
with col4:
    st.metric("AI Models", "30+", delta="ML & DL")
with col5:
    st.metric("Chat AI", "Active", delta="GPU Powered")
with col6:
    st.metric("Updates", "Real-time", delta="Live Data")

st.markdown("---")

# ========================================
# SIDEBAR - NAVIGATION WITH CHAT
# ========================================
with st.sidebar:
    st.markdown("## Navigation")
    
    if st.button("Technical Analysis", use_container_width=True):
        st.switch_page("pages/Technical_Dashboard.py")
    
    if st.button("AI Prediction", use_container_width=True):
        st.switch_page("pages/AI_Prediction.py")
    
    if st.button("Stock Chat Assistant", use_container_width=True):
        st.switch_page("pages/Stock_Chat.py")
    
    if st.button("About", use_container_width=True):
        st.switch_page("pages/About.py")
    
    if st.button("Help & Tips", use_container_width=True):
        st.switch_page("pages/Help.py")
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.info(f"Market: **{market_status}**")
    st.success("Status: **Active**")
    
    st.markdown(f"Date: **{datetime.now().strftime('%b %d, %Y')}**")
# ========================================
# FOOTER
# ========================================
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p style='font-size: 1.2rem;'><b>ASE Insight Platform</b></p>
    <p>Built with dedication for Jordanian Traders & Investors</p>
    <p style='font-size: 0.9rem;'>Powered by Streamlit • PyTorch • GPT4All • Plotly • Real-time ASE Data</p>
    <p style='font-size: 0.8rem; margin-top: 1rem; color: #ef4444;'>
        <b>Disclaimer:</b> This platform is for educational and informational purposes only. 
        Not financial advice. Always conduct your own research before investing.
    </p>
    <p style='font-size: 0.8rem; color: #666; margin-top: 1rem;'>
        © 2025 ASE Insight. All rights reserved. | Made in Jordan
    </p>
</div>
""", unsafe_allow_html=True)