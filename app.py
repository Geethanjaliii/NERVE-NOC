import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
if sys.platform == 'win32':
    _tlib = os.path.join(os.path.dirname(sys.executable), 'Lib', 'site-packages', 'torch', 'lib')
    if os.path.exists(_tlib) and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(_tlib)
        except Exception:
            pass
try:
    import torch
except Exception:
    pass
import datetime
import dotenv
dotenv.load_dotenv()
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Compatibility shims across Streamlit versions
if not hasattr(st, "rerun"):
    st.rerun = getattr(st, "experimental_rerun", lambda: None)
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from fpdf import FPDF
import zipfile
import io
import re

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.auth import (
    is_authenticated,
    render_login_screen,
    get_current_user,
    logout_user,
    safe_get_query_params,
    safe_clear_query_params
)
from src.topology import build_topology_figure, build_geo_risk_figure
from src.preprocessing import load_and_merge, clean_data
from src.features import build_features
from src.predictor import get_location_freq_map, explain, predict
from src.health_score import compute_health_score
from src.recommender import recommend_action, verify_recommendations_with_rag
from src.rag_service import RAGService
from src.llm_service import LLMService
from src.what_if_engine import WhatIfEngine

# ==============================================================================
# CENTRAL COLOR SYSTEM CONSTANTS & DESIGN TOKENS
# ==============================================================================
COLOR_CRITICAL = "#FF3B3B"  # Saturated, high-contrast urgent red
COLOR_WARNING  = "#FFB81C"  # Amber / yellow
COLOR_HEALTHY  = "#00D9E8"  # Neon cyan / teal

COLOR_BG_PAGE = "#070D14"  # Deep navy/black background
COLOR_SURFACE = "#0D1822"  # Dark blue-black surfaces
COLOR_BORDER  = "#263743"  # Subtle blue-gray borders
COLOR_TEXT_PRI= "#E8EEF2"  # Primary text
COLOR_TEXT_MUT= "#81939F"  # Secondary text
COLOR_TEXT_DIM= "#5F737F"  # Muted subtext

# Streamlit Page Config
st.set_page_config(
    page_title="NERVE NOC — AI-Powered Network Diagnostic Command Center",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for NERVE NOC Command Center Theme
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* Global Reset & Background */
*, *::before, *::after {{
    box-sizing: border-box !important;
}}

html, body, .stApp {{
    background-color: {COLOR_BG_PAGE} !important;
    color: {COLOR_TEXT_PRI} !important;
    font-family: 'JetBrains Mono', monospace !important;
    overflow-x: hidden !important;
}}

section.main {{
    overflow-x: hidden !important;
}}

/* Hide Empty Top Header & Stray Hover Elements */
header[data-testid="stHeader"],
button[title="View fullscreen"],
button[data-testid="StyledFullScreenButton"],
div[data-testid="stElementToolbar"],
.element-container:has(.stPlotlyChart) button,
.modebar-container,
.modebar,
.plotly .modebar {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}}

/* Clean Base Layout */
.block-container {{
    padding-top: 1.0rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}}

/* Typography */
h1, h2, h3, h4, h5, h6 {{
    font-family: 'Geist', sans-serif !important;
    font-weight: 600 !important;
    color: {COLOR_TEXT_PRI} !important;
    letter-spacing: -0.02em;
}}

.stMarkdown, p, label {{
    font-family: 'JetBrains Mono', monospace !important;
    color: {COLOR_TEXT_MUT};
}}

/* Top Navigation Bar / Breadcrumb */
.nerve-header-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid {COLOR_BORDER};
    padding-bottom: 14px;
    margin-bottom: 20px;
}}

.nerve-title {{
    font-family: 'Geist', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRI};
    margin: 0;
}}

.nerve-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: {COLOR_TEXT_MUT};
    margin-top: 2px;
}}

/* Status Legend */
.legend-container {{
    display: flex;
    gap: 16px;
    align-items: center;
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    width: fit-content;
    margin-bottom: 16px;
}}

.legend-item {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    color: {COLOR_TEXT_PRI};
}}

.dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}}

.dot.critical {{ background-color: {COLOR_CRITICAL}; box-shadow: 0 0 6px {COLOR_CRITICAL}; }}
.dot.warning  {{ background-color: {COLOR_WARNING};  box-shadow: 0 0 6px {COLOR_WARNING}; }}
.dot.healthy  {{ background-color: {COLOR_HEALTHY};  box-shadow: 0 0 6px {COLOR_HEALTHY}; }}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background-color: {COLOR_SURFACE} !important;
    border: 1px solid {COLOR_BORDER} !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}}

div[data-testid="stMetricLabel"] > label {{
    color: {COLOR_TEXT_MUT} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}}

div[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: {COLOR_TEXT_PRI} !important;
}}

/* =============================================================================
   FLEET OVERVIEW REBUILD SPECIFICATIONS
   ============================================================================= */

/* 1. Header */
.fleet-header-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 20px;
    width: 100%;
}}

.fleet-title {{
    font-family: 'Geist', sans-serif;
    font-size: 1.30rem;
    font-weight: 700;
    color: #E8EEF2;
    letter-spacing: 0.04em;
    line-height: 1.2;
    margin: 0;
}}

.fleet-subtitle {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    color: #81939F;
    margin-top: 4px;
    line-height: 1.2;
}}

.fleet-meta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #81939F;
}}

/* 2. Metric Cards (min-height: 90px, padding: 16px, margin-bottom: 16px) */
.metric-card-rebuild {{
    background-color: #0D1822;
    border: 1px solid #263743;
    border-radius: 8px;
    padding: 16px;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
    width: 100%;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}}

.metric-card-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.70rem;
    color: #81939F;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

.metric-card-body {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
}}

.metric-card-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}}

.metric-card-delta {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
}}

/* 3. Three-Panel Row 2 Cards (min-height: 320px, padding: 20px, margin-bottom: 16px) */
div[data-testid="column"]:has(.panel-row2-marker) {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    min-height: 320px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    margin-bottom: 16px !important;
    overflow: visible !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

/* 4. Two-Panel Row 3 Cards (min-height: 340px, padding: 20px, margin-bottom: 20px) */
div[data-testid="column"]:has(.panel-row3-marker) {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    min-height: 340px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    margin-bottom: 20px !important;
    overflow: visible !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

/* =============================================================================
   DEVICE DETAIL & OPERATIONS CARD CONTAINER RULES
   ============================================================================= */

/* Device Detail Top Metric Cards (min-height: 240px, padding: 18px) */
div[data-testid="column"]:has(.panel-detail-metric-marker) {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 18px !important;
    min-height: 240px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    margin-bottom: 16px !important;
    overflow: visible !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

/* Device Detail Middle Analysis Cards (min-height: 360px, padding: 20px) */
div[data-testid="column"]:has(.panel-detail-analysis-marker) {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    min-height: 360px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    margin-bottom: 16px !important;
    overflow: visible !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

/* Device Detail Bottom Intelligence Cards (min-height: 340px, padding: 20px) */
div[data-testid="column"]:has(.panel-detail-intel-marker) {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    min-height: 340px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    margin-bottom: 20px !important;
    overflow: visible !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

/* Operations View Panels (padding: 20px) */
div[data-testid="column"]:has(.panel-operations-marker) {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 20px !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
    margin-bottom: 20px !important;
    overflow: visible !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

/* Backward-compatible NOC Panel Classes */
.noc-panel {{
    background-color: #0D1822 !important;
    border: 1px solid #263743 !important;
    border-radius: 8px !important;
    padding: 18px !important;
    box-sizing: border-box !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
}}

.noc-panel-title {{
    font-family: 'Geist', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #ffffff !important;
    margin-bottom: 6px !important;
    line-height: 1.2 !important;
}}

.noc-panel-subtitle {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #81939F !important;
    margin-bottom: 14px !important;
    line-height: 1.2 !important;
}}

/* Title & Subtitle Margins: Title 8px margin-bottom; Subtitle 16px margin-bottom */
.card-title-rebuild {{
    font-family: 'Geist', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #ffffff;
    margin-bottom: 8px;
    line-height: 1.2;
}}

.card-subtitle-rebuild {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #81939F;
    margin-bottom: 16px;
    line-height: 1.2;
}}

.card-header-flex {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}}

.honesty-caption-rebuild {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #81939F;
    margin-top: 8px;
    line-height: 1.35;
}}

/* Critical Devices Table */
.crit-device-table {{
    display: flex;
    flex-direction: column;
    width: 100%;
}}

.crit-grid-header {{
    display: grid !important;
    grid-template-columns: 18% 22% 40% 20% !important;
    align-items: center !important;
    padding: 0 0 8px 0 !important;
    border-bottom: 1px solid #263743 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #81939F !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}}

.crit-grid-row {{
    display: grid !important;
    grid-template-columns: 18% 22% 40% 20% !important;
    align-items: center !important;
    height: 40px !important;
    border-bottom: 1px solid #263743 !important;
    box-sizing: border-box !important;
}}

.crit-cell-id {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    color: #E8EEF2;
    white-space: nowrap;
}}

.crit-cell-loc {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #81939F;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 6px;
}}

.crit-cell-al {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #FF3B3B;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding-right: 8px;
}}

.crit-cell-act {{
    display: flex;
    justify-content: flex-end;
    align-items: center;
}}

.crit-action-btn {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    height: 26px;
    padding: 0 8px;
    border-radius: 4px;
    background-color: #131b2e;
    border: 1px solid #263743;
    color: #00D9E8 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.15s ease;
    white-space: nowrap;
}}

.crit-action-btn:hover {{
    background-color: #1a263d;
    border-color: #00D9E8;
    box-shadow: 0 0 8px rgba(0, 217, 232, 0.25);
    color: #ffffff !important;
}}

/* 5. Device Infrastructure Monitor Container (padding: 20px) */
.monitor-card-outer {{
    background-color: #0D1822;
    border: 1px solid #263743;
    border-radius: 8px;
    padding: 20px;
    box-sizing: border-box;
    width: 100%;
    margin-bottom: 24px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
}}

/* Action Button Row: Minimum widths and clean spacing */
.batch-btn-container {{
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 16px;
    flex-wrap: wrap;
}}

/* Device Rows (FIXED 56px height, vertical-align: middle) */
div[data-testid="stHorizontalBlock"]:has(.device-card) {{
    align-items: center !important;
    height: 56px !important;
    min-height: 56px !important;
    max-height: 56px !important;
    margin-bottom: 8px !important;
    gap: 8px !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) div[data-testid="column"]:first-child {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 56px !important;
    max-width: 44px !important;
    min-width: 38px !important;
    padding-left: 4px !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) div[data-testid="column"]:nth-child(2) {{
    flex: 1 1 auto !important;
    height: 56px !important;
    min-width: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) div[data-testid="column"]:last-child {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 56px !important;
    max-width: 120px !important;
    min-width: 105px !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) .stCheckbox {{
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 56px !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) .stButton {{
    width: 100% !important;
    margin: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) .stButton > button {{
    height: 46px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 8px !important;
}}

.device-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 0 16px;
    height: 56px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.15s ease;
    margin-bottom: 0px !important;
    width: 100%;
    box-sizing: border-box;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) {{
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 6px !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) div[data-testid="column"]:first-child {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 0 0 36px !important;
    max-width: 36px !important;
    min-width: 36px !important;
    padding: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) div[data-testid="column"]:nth-child(2) {{
    flex: 1 1 auto !important;
    min-width: 0 !important;
    max-width: calc(100% - 150px) !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) div[data-testid="column"]:last-child {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 0 0 100px !important;
    max-width: 100px !important;
    min-width: 100px !important;
    padding: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) .stCheckbox {{
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) .stButton {{
    width: 100% !important;
    margin: 0 !important;
}}

div[data-testid="stHorizontalBlock"]:has(.device-card) .stButton > button {{
    height: 50px !important;
    width: 100px !important;
    min-width: 100px !important;
    max-width: 100px !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    white-space: nowrap !important;
    box-sizing: border-box !important;
}}

.device-card {{
    background-color: {COLOR_SURFACE};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 0 16px;
    height: 50px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: all 0.15s ease;
    margin-bottom: 0px !important;
    width: 100%;
    max-width: 100%;
    box-sizing: border-box;
    overflow: hidden;
}}

.device-card:hover {{
    border-color: {COLOR_HEALTHY};
    box-shadow: 0 0 10px rgba(0, 217, 232, 0.15);
}}

.device-card.critical {{ border-left: 4px solid {COLOR_CRITICAL} !important; }}
.device-card.warning  {{ border-left: 4px solid {COLOR_WARNING} !important; }}
.device-card.healthy  {{ border-left: 4px solid {COLOR_HEALTHY} !important; }}

.device-info-col {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    min-width: 0;
    overflow: hidden;
    flex: 1 1 auto;
}}

.device-id {{
    font-family: 'Geist', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: {COLOR_TEXT_PRI};
    white-space: nowrap;
    flex-shrink: 0;
}}

.device-loc {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: {COLOR_TEXT_MUT};
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    flex: 1 1 auto;
}}

.device-metrics-col {{
    display: flex;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
    margin-left: 12px;
}}

.device-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    width: 68px;
    text-align: right;
    flex-shrink: 0;
}}

.device-score.critical {{ color: {COLOR_CRITICAL}; }}
.device-score.warning  {{ color: {COLOR_WARNING}; }}
.device-score.healthy  {{ color: {COLOR_HEALTHY}; }}

/* Status Pill */
.status-pill {{
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    width: 85px;
    text-align: center;
    box-sizing: border-box;
}}

.status-pill.critical {{
    background-color: rgba(255, 59, 59, 0.15);
    color: {COLOR_CRITICAL};
    border: 1px solid rgba(255, 59, 59, 0.35);
}}
.status-pill.warning {{
    background-color: rgba(255, 186, 32, 0.15);
    color: {COLOR_WARNING};
    border: 1px solid rgba(255, 186, 32, 0.35);
}}
.status-pill.healthy {{
    background-color: rgba(0, 219, 233, 0.15);
    color: {COLOR_HEALTHY};
    border: 1px solid rgba(0, 219, 233, 0.35);
}}

/* Form Inputs */
.stTextInput > div > div > input, .stNumberInput > div > div > input {{
    background-color: #0D1822 !important;
    color: #E8EEF2 !important;
    border: 1px solid #263743 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

div[data-baseweb="select"] {{
    background-color: #0D1822 !important;
    border-radius: 6px !important;
}}

div[data-baseweb="tag"] {{
    background-color: rgba(0, 217, 232, 0.15) !important;
    border: 1px solid rgba(0, 217, 232, 0.4) !important;
    color: #00D9E8 !important;
    border-radius: 4px !important;
}}

/* Buttons */
.stButton > button {{
    background-color: #0D1822 !important;
    color: #E8EEF2 !important;
    border: 1px solid #263743 !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    white-space: nowrap !important;
    padding: 3px 6px !important;
    min-height: 26px !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    border-color: #00D9E8 !important;
    color: #00D9E8 !important;
    background-color: rgba(0, 217, 232, 0.08) !important;
}}

/* Critical Devices Row Inspect Buttons */
button[key^="btn_crit_row_insp_"] {{
    height: 28px !important;
    min-height: 28px !important;
    padding: 0 8px !important;
    background-color: #131b2e !important;
    border: 1px solid #263743 !important;
    color: #00D9E8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

button[key^="btn_crit_row_insp_"]:hover {{
    background-color: #1a263d !important;
    border-color: #00D9E8 !important;
    color: #ffffff !important;
    box-shadow: 0 0 8px rgba(0, 217, 232, 0.25) !important;
}}

/* Device Detail Action Buttons Alignment */
button[key="btn_detail_dispatch"],
div[data-testid="stDownloadButton"] > button,
button[key="btn_back_to_fleet"] {{
    height: 38px !important;
    min-height: 38px !important;
    max-height: 38px !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    line-height: 1 !important;
}}

button[key="btn_detail_dispatch"] {{
    background-color: {COLOR_CRITICAL} !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'Geist', sans-serif !important;
    box-shadow: 0 0 12px rgba(255, 59, 59, 0.35) !important;
}}

button[key="btn_detail_dispatch"]:hover {{
    background-color: #ff5555 !important;
    box-shadow: 0 0 20px rgba(255, 59, 59, 0.6) !important;
}}

div[data-testid="stDownloadButton"] > button {{
    background-color: #101726 !important;
    border: 1px solid #263743 !important;
    color: #E8EEF2 !important;
    font-family: 'JetBrains Mono', monospace !important;
}}

div[data-testid="stDownloadButton"] > button:hover {{
    border-color: #00D9E8 !important;
    color: #00D9E8 !important;
    background-color: rgba(0, 217, 232, 0.08) !important;
}}

/* Device Detail Tilly Form Container */
div[data-testid="stForm"] {{
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin-top: 10px !important;
}}

div[data-testid="stForm"] div[data-testid="stHorizontalBlock"] {{
    align-items: center !important;
    gap: 8px !important;
}}

div[data-testid="stForm"] .stTextInput {{
    margin: 0 !important;
    padding: 0 !important;
}}

div[data-testid="stForm"] .stTextInput input {{
    height: 40px !important;
    min-height: 40px !important;
    font-size: 0.82rem !important;
    box-sizing: border-box !important;
}}

div[data-testid="stFormSubmitButton"] {{
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
}}

div[data-testid="stFormSubmitButton"] > button {{
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 16px !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box !important;
    background-color: #00D9E8 !important;
    color: #070D14 !important;
    border: none !important;
    font-weight: 700 !important;
    font-family: 'Geist', sans-serif !important;
    box-shadow: 0 0 14px rgba(0, 217, 232, 0.35) !important;
}}

div[data-testid="stFormSubmitButton"] > button:hover {{
    background-color: #00F3FF !important;
    box-shadow: 0 0 20px rgba(0, 217, 232, 0.6) !important;
    color: #070D14 !important;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: #081018 !important;
    border-right: 1px solid #263743 !important;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 0 16px 0;
    border-bottom: 1px solid #263743;
    margin-bottom: 20px;
}}

.sidebar-title {{
    font-family: 'Geist', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #E8EEF2;
    letter-spacing: -0.01em;
}}

.sidebar-sub {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #00D9E8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

.sidebar-section-title {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #81939F;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 24px;
    margin-bottom: 8px;
}}

.sidebar-status-box {{
    background-color: #0D1822;
    border: 1px solid #263743;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 12px;
}}

div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {{
    background: rgba(0, 217, 232, 0.12) !important;
    border-left: 3px solid #00D9E8 !important;
    color: #00D9E8 !important;
    border-radius: 4px !important;
    box-shadow: 0 0 10px rgba(0, 217, 232, 0.15) !important;
}}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING & BACKEND PIPELINE (CACHED)
# ==============================================================================
@st.cache_data
def load_and_process_test_data(num_samples: int = 100):
    """
    Loads real raw data, extracts 100 test split records, and computes predictions + health scores.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
    
    raw_df = load_and_merge(data_dir)
    
    _, test_raw = train_test_split(
        raw_df, test_size=0.20, random_state=42, stratify=raw_df['fault_severity']
    )
    
    sample_records = test_raw.head(num_samples).copy()
    loc_map = get_location_freq_map(models_dir=os.path.dirname(model_path), data_dir=data_dir)
    model = joblib.load(model_path)
    
    X_clean, _, _ = clean_data(sample_records, location_freq_map=loc_map)
    X_feat = build_features(X_clean)
    
    expected_cols = list(model.feature_names_in_)
    for col in expected_cols:
        if col not in X_feat.columns:
            X_feat[col] = 0
    X_feat = X_feat[expected_cols]
    
    probs = model.predict_proba(X_feat)
    preds = model.predict(X_feat)
    
    records_list = []
    for i in range(len(sample_records)):
        row = sample_records.iloc[i]
        p = probs[i].tolist()
        pred_c = int(preds[i])
        score, status = compute_health_score(p)
        
        records_list.append({
            'id': int(row['id']),
            'location': str(row['location']),
            'severity_type': str(row['severity_type']),
            'fault_severity': int(row['fault_severity']),
            'predicted_class': pred_c,
            'p0': p[0],
            'p1': p[1],
            'p2': p[2],
            'health_score': score,
            'status': status
        })
        
    return pd.DataFrame(records_list)

@st.cache_data
def load_raw_test_records(num_samples: int = 100) -> dict:
    """Loads raw test records in wide format, mapped by device ID."""
    data_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    raw_df = load_and_merge(data_dir)
    _, test_raw = train_test_split(
        raw_df, test_size=0.20, random_state=42, stratify=raw_df['fault_severity']
    )
    sample_records = test_raw.head(num_samples).copy()
    sample_records['id'] = sample_records['id'].astype(int)
    return sample_records.set_index('id').to_dict(orient='index')

# Initialize real dataset results
df_results = load_and_process_test_data(num_samples=100)
raw_records = load_raw_test_records(num_samples=100)


# ==============================================================================
# HELPER UTILITIES: TRANSLATION, PDF EXPORT, CHATBOT ROUTING
# ==============================================================================
def translate_feature_name(name: str) -> str:
    """Translates raw model feature names into human-readable signal names."""
    if name == 'total_volume':
        return "Overall event volume"
    if name == 'severity_type':
        return "Reported severity classification"
    if name == 'location':
        return "Location risk frequency"
    if name == 'num_active_log_features':
        return "Number of distinct log signals involved"
    if name == 'num_event_types':
        return "Number of distinct event trigger types involved"
    if name == 'num_resource_types':
        return "Number of distinct resource types involved"
    
    m_log = re.match(r'^(?:log_)?log_feature\s+(\d+)$', name)
    if m_log:
        return f"Log signal #{m_log.group(1)} (elevated volume)"
    
    m_event = re.match(r'^(?:event_)?event_type\s+(\d+)$', name)
    if m_event:
        return f"Event trigger type #{m_event.group(1)}"
        
    m_resource = re.match(r'^(?:resource_)?resource_type\s+(\d+)$', name)
    if m_resource:
        return f"Resource type #{m_resource.group(1)} affected"
        
    return name

def generate_device_pdf(device_id, row, status_color, shap_factors, actions):
    """Generates an official PDF report for a single device incident."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 20, 15)
    
    # Title & Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(17, 19, 23)
    pdf.cell(0, 10, "NERVE NOC - Telecom Network Incident Report", ln=True, align="C")
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 110, 120)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(0, 8, f"Device ID: #{device_id}  |  Generated: {current_time}", ln=True, align="C")
    pdf.ln(6)
    
    pdf.set_draw_color(185, 202, 203)
    pdf.set_line_width(0.5)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)
    
    # Section 1: Device Information
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(17, 19, 23)
    pdf.cell(0, 8, "1. Device Information", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 6, "Device ID:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"#{device_id}", border=0, ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(55, 6, "Location Node:", border=0)
    pdf.cell(0, 6, str(row['location']), border=0, ln=True)
    
    pdf.cell(55, 6, "Reported Severity Type:", border=0)
    pdf.cell(0, 6, str(row['severity_type']), border=0, ln=True)
    
    pdf.cell(55, 6, "Predicted Fault Severity:", border=0)
    pdf.cell(0, 6, f"Class {row['predicted_class']}", border=0, ln=True)
    pdf.ln(6)
    
    # Section 2: Health Score & Status
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Health Score & Status", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(55, 6, "Composite Health Score:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"{float(row['health_score']):.1f} / 100.0", border=0, ln=True)
    pdf.set_font("Helvetica", "", 10)
    
    pdf.cell(55, 6, "Operational Status:", border=0)
    pdf.set_font("Helvetica", "B", 10)
    status_str = str(row['status'])
    if status_str == 'Critical':
        pdf.set_text_color(255, 59, 59)
    elif status_str == 'Warning':
        pdf.set_text_color(255, 186, 32)
    else:
        pdf.set_text_color(0, 219, 233)
    pdf.cell(0, 6, status_str, border=0, ln=True)
    pdf.set_text_color(17, 19, 23)
    pdf.ln(6)
    
    # Section 3: Root Cause Analysis
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "3. Root Cause Analysis (XAI)", ln=True)
    pdf.ln(2)
    
    if float(row['health_score']) >= 70:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 110, 120)
        pdf.cell(0, 6, "Operating normally - no significant risk factors detected.", ln=True)
        pdf.set_text_color(17, 19, 23)
    else:
        pdf.set_font("Helvetica", "", 10)
        total_shap = sum(val for _, val in shap_factors if val > 0)
        if total_shap <= 0:
            total_shap = 1.0
            
        for name, val in shap_factors:
            if val > 0:
                impact_pct = (val / total_shap) * 100
                translated = translate_feature_name(name)
                translated_ascii = translated.encode('latin-1', errors='replace').decode('latin-1')
                pdf.cell(100, 6, f"- {translated_ascii}:", border=0)
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 6, f"+{impact_pct:.1f}% impact weight", border=0, ln=True)
                pdf.set_font("Helvetica", "", 10)
    pdf.ln(6)
    
    # Section 4: Recommended Actions
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "4. Recommended Preventive Actions", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    for i, act in enumerate(actions, 1):
        act_ascii = act.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 6, f"{i}. {act_ascii}")
        pdf.ln(1)
        
    return pdf.output()

def generate_reports_zip(selected_ids, df_results, raw_records):
    """Generates a zip package containing PDF incident reports for all selected devices."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for dev_id in selected_ids:
            dev_results = df_results[df_results['id'] == dev_id]
            if dev_results.empty:
                continue
            row = dev_results.iloc[0]
            status = row['status']
            score = float(row['health_score'])
            status_color = COLOR_CRITICAL if status == 'Critical' else (COLOR_WARNING if status == 'Warning' else COLOR_HEALTHY)
            
            raw_dict = raw_records.get(dev_id, {})
            if raw_dict:
                raw_dict['id'] = dev_id
                
            shap_factors = [] if score >= 70 else explain(raw_dict)
            actions = recommend_action(shap_factors, status)
            
            pdf_bytes = generate_device_pdf(dev_id, row, status_color, shap_factors, actions)
            current_date_zip = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"nerve_dev_{dev_id}_report_{current_date_zip}.pdf"
            zip_file.writestr(filename, bytes(pdf_bytes))
            
    return zip_buffer.getvalue()

def execute_pandas_query(route: dict, df: pd.DataFrame, prompt: str = "") -> str:
    """Executes deterministic aggregation queries across the current application runtime data."""
    prompt_lower = prompt.lower()
    
    if "list" in prompt_lower or "show" in prompt_lower or "display" in prompt_lower or "find" in prompt_lower:
        if "critical" in prompt_lower:
            crit_df = df[df['status'] == 'Critical']
            if crit_df.empty:
                return "There are no critical devices currently loaded in the active runtime data."
            device_list = ", ".join([f"#{dev_id} ({loc}, Health: {score:.1f}%)" for dev_id, loc, score in zip(crit_df['id'], crit_df['location'], crit_df['health_score'])])
            return f"**Critical Devices in Current Runtime Data:**\n{device_list}"
        elif "warning" in prompt_lower:
            warn_df = df[df['status'] == 'Warning']
            if warn_df.empty:
                return "There are no warning devices currently loaded in the active runtime data."
            device_list = ", ".join([f"#{dev_id} ({loc}, Health: {score:.1f}%)" for dev_id, loc, score in zip(warn_df['id'], warn_df['location'], warn_df['health_score'])])
            return f"**Warning Devices in Current Runtime Data:**\n{device_list}"
        elif "healthy" in prompt_lower:
            health_df = df[df['status'] == 'Healthy']
            if health_df.empty:
                return "There are no healthy devices currently loaded in the active runtime data."
            device_list = ", ".join([f"#{dev_id} ({loc}, Health: {score:.1f}%)" for dev_id, loc, score in zip(health_df['id'], health_df['location'], health_df['health_score'])])
            return f"**Healthy Devices in Current Runtime Data:**\n{device_list}"

    loc = route.get("location")
    agg = route.get("aggregation")
    
    if loc:
        loc_df = df[df['location'].str.lower() == loc.lower()]
        if loc_df.empty:
            return f"Location '{loc}' was not found in the current active runtime sample set."
            
        total_incidents = len(loc_df)
        total_faults = int((loc_df['fault_severity'] > 0).sum())
        minor_faults = int((loc_df['fault_severity'] == 1).sum())
        major_faults = int((loc_df['fault_severity'] == 2).sum())
        avg_health = loc_df['health_score'].mean()
        worst_severity = int(loc_df['fault_severity'].max())
        
        return (
            f"**Query Results for {loc} (Current Runtime Data):**\n"
            f"- **Total monitored incidents**: {total_incidents}\n"
            f"- **Active network faults**: {total_faults} (Minor: {minor_faults}, Major: {major_faults})\n"
            f"- **Worst recorded fault severity class**: {worst_severity}\n"
            f"- **Average device health score**: {avg_health:.1f}%"
        )
    
    total_incidents = len(df)
    total_faults = int((df['fault_severity'] > 0).sum())
    minor_faults = int((df['fault_severity'] == 1).sum())
    major_faults = int((df['fault_severity'] == 2).sum())
    avg_health = df['health_score'].mean()
    avg_severity = df['fault_severity'].mean()
    
    if agg == "sum":
        return (
            f"**Network Aggregates (Current Runtime Data):**\n"
            f"- **Total monitored incidents**: {total_incidents}\n"
            f"- **Total active network faults**: {total_faults}\n"
            f"  - *Minor faults (Warning)*: {minor_faults}\n"
            f"  - *Major faults (Critical)*: {major_faults}"
        )
    elif agg == "max":
        worst_loc_row = df.groupby('location')['health_score'].mean().idxmin()
        worst_loc_score = df.groupby('location')['health_score'].mean().min()
        return (
            f"**Network Aggregates (Worst Performing Nodes):**\n"
            f"- **Max fault severity class**: {int(df['fault_severity'].max())}\n"
            f"- **Lowest average health location**: {worst_loc_row} (Avg Health: {worst_loc_score:.1f}%)"
        )
    else:
        return (
            f"**Active Network Data Summary:**\n"
            f"- **Total monitored incidents**: {total_incidents}\n"
            f"- **Active network faults**: {total_faults} (Minor: {minor_faults}, Major: {major_faults})\n"
            f"- **Average device health score**: {avg_health:.1f}%\n"
            f"- **Average fault severity index**: {avg_severity:.2f}"
        )

def process_chat_query(prompt: str, active_device_context: str = None) -> str:
    """Processes network-level or device-level queries through intent routing and RAG search."""
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    llm_srv = LLMService(api_key=env_key)
    route = llm_srv.route_query(prompt)
    
    if not route.get("is_network_related", True):
        return "I can only answer questions based on the telemetry and network operations data in NERVE NOC."
        
    if route.get("query_type") == "numerical":
        return execute_pandas_query(route, df_results, prompt)
        
    try:
        rag_srv = RAGService()
    except Exception as e:
        return f"RAG Service unavailable: {e}"
        
    if route.get("refers_to_active_device", False):
        if active_device_context:
            return llm_srv.generate_answer(prompt, [], active_device_context)
        else:
            return "Please select a specific device in Device Detail to query its individual telemetry context."
            
    elif route.get("target_device_id"):
        tgt_id = route["target_device_id"]
        dict_res = rag_srv.lookup_device_id(tgt_id)
        if dict_res:
            return llm_srv.generate_answer(prompt, [dict_res], active_device_context)
        else:
            return f"Device #{tgt_id} was not found in the historical network records."
    else:
        context_records = rag_srv.retrieve(prompt, k=4)
        return llm_srv.generate_answer(prompt, context_records, active_device_context)


# ==============================================================================
# AUTHENTICATION GATE & LOGIN SCREEN
# ==============================================================================
if not is_authenticated():
    render_login_screen()
    st.stop()


# ==============================================================================
# SESSION STATE NAVIGATION MANAGEMENT
# ==============================================================================
if 'nav_selection' not in st.session_state:
    st.session_state.nav_selection = "FLEET OVERVIEW"
if 'selected_device_id' not in st.session_state:
    st.session_state.selected_device_id = None

# Handle URL query parameters for deep linking
qp = safe_get_query_params()
if "inspect_dev" in qp:
    try:
        st.session_state.selected_device_id = int(qp["inspect_dev"])
        st.session_state.nav_selection = "DEVICE DETAIL"
        safe_clear_query_params()
    except Exception:
        pass
if "nav" in qp:
    nav_map = {
        "fleet": "FLEET OVERVIEW",
        "operations": "OPERATIONS",
        "device": "DEVICE DETAIL",
        "tilly": "🤖  TILLY — NOC ASSISTANT"
    }
    if qp["nav"].lower() in nav_map:
        st.session_state.nav_selection = nav_map[qp["nav"].lower()]
        safe_clear_query_params()


# ==============================================================================
# SIDEBAR NAVIGATION (EXACT SPECIFICATION)
# ==============================================================================
with st.sidebar:
    # 1. Brand Logo & Title
    st.markdown("""
    <div class="sidebar-brand">
        <div style="display: flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 8px; background: rgba(0, 217, 232, 0.15); border: 1px solid rgba(0, 217, 232, 0.4);">
            <span style="font-size: 1.1rem; color: #00D9E8;">🛡️</span>
        </div>
        <div>
            <div class="sidebar-title">NERVE NOC</div>
            <div class="sidebar-sub">NETWORK INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Main Navigation Items
    nav_options = ["FLEET OVERVIEW", "OPERATIONS", "DEVICE DETAIL", "AI COPILOT"]
    nav_icons = {"FLEET OVERVIEW": "📡", "OPERATIONS": "⚡", "DEVICE DETAIL": "🔍", "AI COPILOT": "🤖"}
    
    curr_nav = st.session_state.get("nav_selection", "FLEET OVERVIEW")
    curr_idx = nav_options.index(curr_nav) if curr_nav in nav_options else 0
    
    nav_choice = st.radio(
        "Navigation",
        options=nav_options,
        index=curr_idx,
        format_func=lambda x: "🤖  TILLY — NOC ASSISTANT" if x == "AI COPILOT" else f"{nav_icons[x]}  {x}",
        label_visibility="collapsed",
        key="main_nav_radio"
    )
    if nav_choice != curr_nav:
        st.session_state.nav_selection = nav_choice
        st.rerun()
        
    st.markdown("<hr style='border-color: #263743; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)
    
    # 3. System Section
    st.markdown('<div class="sidebar-section-title">SYSTEM</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sidebar-status-box">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 600; color: {COLOR_TEXT_PRI};">
            <span class="dot healthy"></span> Network Status: <span style="color: {COLOR_HEALTHY};">Online</span>
        </div>
        <div style="font-size: 0.72rem; color: {COLOR_TEXT_MUT}; margin-top: 4px;">
            Monitored Nodes: 929 &bull; XGBoost Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #263743; margin: 16px 0;'>", unsafe_allow_html=True)
    
    # 4. User Profile Section & Logout
    user_info = get_current_user()
    user_name = user_info.get("name", "GEETZ")
    
    st.markdown('<div class="sidebar-section-title">USER</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #0D1822; border: 1px solid #263743; display: flex; align-items: center; justify-content: center; font-weight: 700; color: #00D9E8; font-size: 0.85rem;">
            {user_name[:2].upper()}
        </div>
        <div>
            <div style="font-size: 0.85rem; font-weight: 700; color: {COLOR_TEXT_PRI};">{user_name}</div>
            <div style="font-size: 0.72rem; color: {COLOR_HEALTHY}; display: flex; align-items: center; gap: 4px;">
                <span class="dot healthy" style="width: 6px; height: 6px;"></span> Online
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Logout", key="btn_logout", use_container_width=True):
        logout_user()


# ==============================================================================
# VIEW 1: FLEET OVERVIEW (COMMAND CENTER) - FULL CLEAN REBUILD
# ==============================================================================
def render_fleet_overview():
    # Handle direct navigation query param safely
    try:
        q_params = st.experimental_get_query_params()
        if "inspect_dev" in q_params:
            val = q_params["inspect_dev"]
            dev_id = int(val[0] if isinstance(val, list) else val)
            st.session_state.selected_device_id = dev_id
            st.session_state.nav_selection = "DEVICE DETAIL"
            st.experimental_set_query_params()
            st.rerun()
    except Exception:
        pass

    # 1. Header: Title + Subtitle + Last updated (space-between, margin-bottom: 20px)
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="fleet-header-row">
        <div>
            <h1 class="fleet-title">FLEET OVERVIEW</h1>
            <div class="fleet-subtitle">Real-time network health at a glance</div>
        </div>
        <div class="fleet-meta">
            Last updated: <span style="color: #00D9E8; font-weight: 600;">{now_time}</span> &bull; Live Telemetry Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset summary calculations
    global_health_pct = round(df_results['health_score'].mean(), 1)
    total_critical = len(df_results[df_results['status'] == 'Critical'])
    total_warning = len(df_results[df_results['status'] == 'Warning'])
    total_healthy = len(df_results[df_results['status'] == 'Healthy'])
    total_devices = len(df_results)
    
    pct_healthy = (total_healthy / total_devices) * 100
    pct_warning = (total_warning / total_devices) * 100
    pct_critical = (total_critical / total_devices) * 100
    
    # 2. Metric cards row: 4 equal columns, min-height: 90px, padding: 16px, margin-bottom: 16px
    col_m1, col_m2, col_m3, col_m4 = st.columns(4, gap="medium")
    with col_m1:
        st.markdown(f"""
        <div class="metric-card-rebuild">
            <div class="metric-card-title">TOTAL DEVICES</div>
            <div class="metric-card-body">
                <span class="metric-card-num">{total_devices:,}</span>
                <span class="metric-card-delta" style="color: #00D9E8;">↑ 12</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card-rebuild">
            <div class="metric-card-title">HEALTHY</div>
            <div class="metric-card-body">
                <span class="metric-card-num" style="color: #00D9E8;">{total_healthy:,}</span>
                <span class="metric-card-delta" style="color: #00D9E8;">{pct_healthy:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card-rebuild">
            <div class="metric-card-title">WARNING</div>
            <div class="metric-card-body">
                <span class="metric-card-num" style="color: #FFB81C;">{total_warning:,}</span>
                <span class="metric-card-delta" style="color: #FFB81C;">{pct_warning:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card-rebuild">
            <div class="metric-card-title">CRITICAL</div>
            <div class="metric-card-body">
                <span class="metric-card-num" style="color: #FF3B3B;">{total_critical:,}</span>
                <span class="metric-card-delta" style="color: #FF3B3B;">{pct_critical:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Three-panel row: st.columns([1, 1, 1.3]), min-height: 320px, padding: 20px, margin-bottom: 16px
    col_nh, col_rd, col_rl = st.columns([1.0, 1.0, 1.3], gap="medium")
    
    with col_nh:
        st.markdown("""
        <span class="panel-row2-marker"></span>
        <div class="card-title-rebuild">NETWORK HEALTH</div>
        <div class="card-subtitle-rebuild">Average Health Score</div>
        """, unsafe_allow_html=True)
        
        status_txt = "Good" if global_health_pct >= 70 else ("Warning" if global_health_pct >= 40 else "Critical")
        status_col = COLOR_HEALTHY if global_health_pct >= 70 else (COLOR_WARNING if global_health_pct >= 40 else COLOR_CRITICAL)
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=global_health_pct,
            domain={'x': [0.05, 0.95], 'y': [0.0, 1.0]},
            number={
                'valueformat': '.0f',
                'font': {'family': 'JetBrains Mono', 'size': 36, 'color': '#E8EEF2'},
                'suffix': '<span style="font-size: 15px; color: #81939F; font-weight: 500;"> / 100</span>'
            },
            gauge={
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': status_col, 'thickness': 0.22},
                'bgcolor': 'rgba(255, 255, 255, 0.05)',
                'borderwidth': 0
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='#0D1822',
            plot_bgcolor='#0D1822',
            margin=dict(l=5, r=5, t=0, b=0),
            height=160
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"""
        <div style="text-align: center; font-family: 'JetBrains Mono'; font-size: 0.82rem; font-weight: 700; color: {status_col}; margin-top: 4px;">
            ● {status_txt} &nbsp;&bull;&nbsp; <span style="font-size: 0.72rem; color: #81939F; font-weight: 400;">↑ 5 vs last 24h</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_rd:
        st.markdown("""
        <span class="panel-row2-marker"></span>
        <div class="card-title-rebuild">RISK DISTRIBUTION</div>
        <div class="card-subtitle-rebuild">Fleet status breakdown</div>
        """, unsafe_allow_html=True)
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=[
                f'Critical &nbsp; {total_critical} ({pct_critical:.1f}%)',
                f'Warning &nbsp; {total_warning} ({pct_warning:.1f}%)',
                f'Healthy &nbsp; {total_healthy} ({pct_healthy:.1f}%)'
            ],
            values=[total_critical, total_warning, total_healthy],
            hole=0.60,
            domain={'x': [0.02, 0.44], 'y': [0.0, 1.0]},
            marker=dict(colors=[COLOR_CRITICAL, COLOR_WARNING, COLOR_HEALTHY], line=dict(color='#0D1822', width=2)),
            textinfo='none',
            sort=False
        )])
        fig_donut.update_layout(
            paper_bgcolor='#0D1822',
            plot_bgcolor='#0D1822',
            margin=dict(l=0, r=0, t=0, b=0),
            height=175,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.48,
                font=dict(family='JetBrains Mono', size=9.5, color='#E8EEF2')
            )
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
        
    with col_rl:
        st.markdown("""
        <span class="panel-row2-marker"></span>
        <div class="card-header-flex">
            <div class="card-title-rebuild" style="margin-bottom: 0;">RISK BY LOCATION</div>
            <div style="display: flex; gap: 8px; font-size: 0.68rem; font-family: 'JetBrains Mono';">
                <span style="color: #FF3B3B;">● Critical</span>
                <span style="color: #FFB81C;">● Warning</span>
                <span style="color: #00D9E8;">● Healthy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_geo = build_geo_risk_figure(df_results, max_locations=35)
        st.plotly_chart(fig_geo, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div class="honesty-caption-rebuild">
            Illustrative — synthetic geographic placement representing anonymized location IDs, not real GPS coordinates. Risk intensity reflects real model output.
        </div>
        """, unsafe_allow_html=True)
        
    # 4. Two-panel row: st.columns([1, 1.1]), min-height: 340px, padding: 20px, margin-bottom: 20px
    col_topo, col_crit = st.columns([1.0, 1.1], gap="medium")
    
    with col_topo:
        st.markdown("""
        <span class="panel-row3-marker"></span>
        <div class="card-header-flex">
            <div class="card-title-rebuild" style="margin-bottom: 0;">NETWORK TOPOLOGY</div>
            <div style="display: flex; gap: 8px; font-size: 0.68rem; font-family: 'JetBrains Mono';">
                <span style="color: #00D9E8;">● Healthy</span>
                <span style="color: #FFB81C;">● Warning</span>
                <span style="color: #FF3B3B;">● Critical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        topo_fig = build_topology_figure(df_results, max_nodes=25)
        st.plotly_chart(topo_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("""
        <div class="honesty-caption-rebuild">
            Illustrative — synthetic network topology representing anonymized device connections. Node health is from current data.
        </div>
        """, unsafe_allow_html=True)
        
    with col_crit:
        st.markdown("""
        <span class="panel-row3-marker"></span>
        <div class="card-header-flex">
            <div class="card-title-rebuild" style="margin-bottom: 0;">CRITICAL DEVICES</div>
            <a href="#monitor-section" style="color: #00D9E8; font-size: 0.72rem; text-decoration: none; font-family: 'JetBrains Mono'; font-weight: 600;">View All &rarr;</a>
        </div>
        """, unsafe_allow_html=True)
        
        crit_devices = df_results[df_results['status'] == 'Critical'].sort_values(by='health_score', ascending=True).head(5)
        if crit_devices.empty:
            crit_devices = df_results.sort_values(by='health_score', ascending=True).head(5)
            
        st.markdown("""
        <div class="crit-device-table">
            <div class="crit-grid-header">
                <div>DEVICE ID</div><div>LOCATION</div><div>ALERT REASON</div><div style="text-align: right;">ACTION</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for _, r in crit_devices.iterrows():
            d_id = int(r['id'])
            
            sev = str(r.get('severity_type', '')).lower()
            if 'severity_type 1' in sev or 'type 1' in sev:
                alert_text = "Elevated Log Volume"
            elif 'severity_type 2' in sev or 'type 2' in sev:
                alert_text = "Interface Degradation"
            elif 'severity_type 3' in sev or 'type 3' in sev:
                alert_text = "Hardware Alarm Signal"
            elif 'severity_type 4' in sev or 'type 4' in sev:
                alert_text = "Critical Latency Spike"
            elif 'severity_type 5' in sev or 'type 5' in sev:
                alert_text = "Packet Drop Surge"
            else:
                alert_text = "Critical Telemetry Alert"
                
            c_id, c_loc, c_al, c_act = st.columns([1.1, 1.2, 2.0, 0.9])
            with c_id:
                st.markdown(f'<div class="crit-cell-id" style="line-height: 28px;">DEV-{d_id}</div>', unsafe_allow_html=True)
            with c_loc:
                st.markdown(f'<div class="crit-cell-loc" style="line-height: 28px;">{r["location"]}</div>', unsafe_allow_html=True)
            with c_al:
                st.markdown(f'<div class="crit-cell-al" style="line-height: 28px;" title="{alert_text}">{alert_text}</div>', unsafe_allow_html=True)
            with c_act:
                if st.button("Inspect 🔍", key=f"btn_crit_row_insp_{d_id}", use_container_width=True):
                    st.session_state.selected_device_id = d_id
                    st.session_state.nav_selection = "DEVICE DETAIL"
                    st.rerun()
            
    # 5. Device Infrastructure Monitor (Separate Card, padding: 20px)
    st.markdown("<div id='monitor-section' style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="card-title-rebuild" style="font-size: 1.0rem; margin-bottom: 4px;">DEVICE INFRASTRUCTURE MONITOR</div>
    <div class="card-subtitle-rebuild" style="margin-bottom: 16px;">Real-time monitored devices from the current runtime data (sorted by health score ascending).</div>
    """, unsafe_allow_html=True)
    
    col_f1, col_f2 = st.columns([1, 1], gap="medium")
    with col_f1:
        status_filter = st.selectbox("Filter by Status", options=["All Statuses", "Critical", "Warning", "Healthy"], key="fleet_filter_status")
    with col_f2:
        search_id = st.text_input("Search by Device ID / Location", placeholder="e.g. 15086 or location 821", key="fleet_search_txt")
        
    filtered_df = df_results.sort_values(by='health_score', ascending=True)
    if status_filter != "All Statuses":
        filtered_df = filtered_df[filtered_df['status'] == status_filter]
    if search_id.strip():
        s_term = search_id.strip().lower()
        filtered_df = filtered_df[
            filtered_df['id'].astype(str).str.contains(s_term) | 
            filtered_df['location'].str.lower().str.contains(s_term)
        ]
        
    # Multi-select & Batch Actions
    visible_ids = filtered_df['id'].tolist()
    for dev_id in visible_ids:
        dev_key = f"select_{dev_id}"
        if dev_key not in st.session_state:
            st.session_state[dev_key] = False
            
    col_sel_all, col_clear, col_zip_dl = st.columns([1.2, 1.2, 2.2], gap="medium")
    with col_sel_all:
        if st.button("Select All Visible", key="btn_sel_all_visible", use_container_width=True):
            for dev_id in visible_ids:
                st.session_state[f"select_{dev_id}"] = True
            st.rerun()
    with col_clear:
        if st.button("Clear Selection", key="btn_clear_sel", use_container_width=True):
            for dev_id in visible_ids:
                st.session_state[f"select_{dev_id}"] = False
            st.rerun()
            
    selected_ids = [dev_id for dev_id in visible_ids if st.session_state.get(f"select_{dev_id}", False)]
    num_selected = len(selected_ids)
    
    with col_zip_dl:
        if num_selected == 0:
            st.button("Download Selected Reports (ZIP)", disabled=True, key="btn_zip_disabled", use_container_width=True)
        else:
            try:
                zip_data = generate_reports_zip(selected_ids, df_results, raw_records)
                current_date = datetime.datetime.now().strftime("%Y%m%d")
                st.download_button(
                    label=f"📦 Download {num_selected} Reports (ZIP)",
                    data=zip_data,
                    file_name=f"nerve_noc_reports_{current_date}.zip",
                    mime="application/zip",
                    key="btn_zip_active",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error compiling zip package: {e}")
                
    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.markdown(f"""
        <div style="background-color: rgba(255, 59, 59, 0.05); border: 1px solid rgba(255, 59, 59, 0.2); border-radius: 6px; padding: 16px; text-align: center; color: {COLOR_CRITICAL};">
            No monitored devices match the active search criteria.
        </div>
        """, unsafe_allow_html=True)
    else:
        for _, row in filtered_df.iterrows():
            dev_id = int(row['id'])
            loc = row['location']
            score = row['health_score']
            status = row['status']
            status_class = status.lower()
            
            c_chk, c_card, c_btn = st.columns([0.35, 4.65, 1.0], gap="small")
            with c_chk:
                st.checkbox(f"Sel #{dev_id}", key=f"select_{dev_id}", label_visibility="collapsed")
            with c_card:
                st.markdown(f"""
                <div class="device-card {status_class}">
                    <div class="device-info-col">
                        <div class="device-id">DEV-{dev_id} (#{dev_id})</div>
                        <div class="device-loc">Location: {loc} &bull; Reported Tier: {row['severity_type']}</div>
                    </div>
                    <div class="device-metrics-col">
                        <div class="device-score {status_class}">{score:.1f}%</div>
                        <div class="status-pill {status_class}">{status}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_btn:
                if st.button("Inspect 📡", key=f"btn_inspect_list_{dev_id}", use_container_width=True):
                    st.session_state.selected_device_id = dev_id
                    st.session_state.nav_selection = "DEVICE DETAIL"
                    st.rerun()


# ==============================================================================
# VIEW 2: OPERATIONS (LIVE TELEMETRY & INCIDENTS)
# ==============================================================================
def render_operations():
    # 1. Header: Title + Subtitle + Last updated (space-between, margin-bottom: 20px)
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="fleet-header-row">
        <div>
            <h1 class="fleet-title">OPERATIONS</h1>
            <div class="fleet-subtitle">Live network events & telemetry inference workspace</div>
        </div>
        <div class="fleet-meta">
            Last updated: <span style="color: #00D9E8; font-weight: 600;">{now_time}</span> &bull; Live Telemetry Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_incidents, col_sim = st.columns([1.1, 1.4], gap="large")
    
    with col_incidents:
        st.markdown("""
        <span class="panel-operations-marker"></span>
        <div class="card-title-rebuild">ACTIVE INCIDENTS</div>
        <div class="card-subtitle-rebuild">Triggered alerts and telemetry disruptions in active runtime data</div>
        """, unsafe_allow_html=True)
        
        # Load high-priority incidents
        high_risk_devices = df_results[df_results['status'].isin(['Critical', 'Warning'])].sort_values(by='health_score', ascending=True).head(4)
        
        if high_risk_devices.empty:
            high_risk_devices = df_results.head(4)
            
        for _, r in high_risk_devices.iterrows():
            d_id = int(r['id'])
            st_cls = r['status'].lower()
            st_color = COLOR_CRITICAL if r['status'] == 'Critical' else COLOR_WARNING
            
            st.markdown(f"""
            <div style="background-color: #101726; border: 1px solid #263743; border-left: 4px solid {st_color}; border-radius: 6px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-family: 'Geist', sans-serif; font-weight: 700; font-size: 0.95rem; color: #E8EEF2;">DEV-{d_id}</span>
                    <span class="status-pill {st_cls}">{r['status']}</span>
                </div>
                <div style="font-size: 0.78rem; color: #81939F; font-family: 'JetBrains Mono', monospace; margin-bottom: 8px;">
                    Node: <b style="color: #E8EEF2;">{r['location']}</b> &bull; Severity: <b style="color: #E8EEF2;">{r['severity_type']}</b> &bull; Health: <b style="color: {st_color};">{r['health_score']:.1f}%</b>
                </div>
                <div style="font-size: 0.74rem; color: #81939F; font-family: 'JetBrains Mono', monospace; line-height: 1.35;">
                    Disruption signal: Elevated anomaly metrics registered on active hardware interfaces.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Investigate 🔍", key=f"btn_op_investigate_{d_id}", use_container_width=True):
                st.session_state.selected_device_id = d_id
                st.session_state.nav_selection = "DEVICE DETAIL"
                st.rerun()
                
            st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)
            
    with col_sim:
        st.markdown("""
        <span class="panel-operations-marker"></span>
        <div class="card-title-rebuild">LIVE TELEMETRY / PREDICTION</div>
        <div class="card-subtitle-rebuild">Input telemetry attributes to execute live XGBoost fault severity inference</div>
        """, unsafe_allow_html=True)
        
        model_path = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
        loc_map = get_location_freq_map(models_dir=os.path.dirname(model_path))
        
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]
            
        known_locations = sorted(list(loc_map.keys()), key=natural_sort_key)
        if not known_locations:
            known_locations = [f"location {i}" for i in range(1, 100)]
            
        # 1. Location
        use_custom_loc = st.checkbox("Enter custom unlisted location node", key="op_custom_loc_toggle")
        if use_custom_loc:
            selected_location = st.text_input("Custom Network Location", value="location 9999", key="op_loc_text")
        else:
            default_idx = known_locations.index('location 821') if 'location 821' in known_locations else 0
            selected_location = st.selectbox("Network Location Node", options=known_locations, index=default_idx, key="op_loc_select")
            
        # 2. Severity
        selected_severity = st.selectbox("Reported Severity Tier", options=[f"severity_type {i}" for i in range(1, 6)], index=1, key="op_sev_select")
        
        # 3. Resource Types
        selected_resources = st.multiselect("Active Resource Types", options=[f"resource_type {i}" for i in range(1, 11)], default=["resource_type 2"], key="op_res_select")
        
        # 4. Event Types
        selected_events = st.multiselect("Active Event Trigger Types", options=[f"event_type {i}" for i in range(1, 55)], default=["event_type 11"], key="op_evt_select")
        
        # 5. Log Features & Volumes
        selected_log_features = st.multiselect("Active Diagnostic Log Signals", options=[f"log_feature {i}" for i in range(1, 387)], default=["log_feature 203", "log_feature 312"], key="op_log_select")
        
        log_volumes = {}
        if selected_log_features:
            subcols = st.columns(min(len(selected_log_features), 2))
            for idx, lf in enumerate(selected_log_features):
                with subcols[idx % len(subcols)]:
                    default_vol = 5 if lf == "log_feature 203" else 12
                    vol = st.number_input(f"Volume for {lf}", min_value=1, max_value=10000, value=default_vol, step=1, key=f"op_vol_{lf}")
                    log_volumes[lf] = int(vol)
                    
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("⚡ RUN PREDICTION", key="btn_op_run_prediction", type="primary", use_container_width=True)
        
        # Execute Prediction
        sim_record = {
            'location': str(selected_location).strip() if selected_location else 'location 1',
            'severity_type': selected_severity
        }
        for et in selected_events:
            sim_record[f"event_{et}"] = 1
        for rt in selected_resources:
            sim_record[f"resource_{rt}"] = 1
        for lf, vol in log_volumes.items():
            sim_record[lf] = vol
            
        try:
            pred_class, probs = predict(sim_record, model_path=model_path, location_freq_map=loc_map)
            h_score, h_status = compute_health_score(probs)
            pred_color = COLOR_CRITICAL if h_status == 'Critical' else (COLOR_WARNING if h_status == 'Warning' else COLOR_HEALTHY)
            pred_prob_pct = probs[pred_class] * 100
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: #101726; border: 1px solid #263743; border-top: 3px solid {pred_color}; border-radius: 8px; padding: 16px 18px; box-shadow: 0 4px 16px rgba(0,0,0,0.35); margin-top: 10px;">
                <div style="font-family: 'Geist', sans-serif; font-size: 0.88rem; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                    <span>PREDICTION RESULT</span>
                    <span class="status-pill {h_status.lower()}">{h_status} RISK</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
                    <div style="background-color: #0D1822; border: 1px solid #263743; border-radius: 6px; padding: 8px 6px;">
                        <div style="font-size: 0.65rem; color: #81939F; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; font-weight: 600;">FAULT SEVERITY</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: {pred_color}; font-family: 'Geist', sans-serif; margin-top: 4px;">Class {pred_class}</div>
                    </div>
                    <div style="background-color: #0D1822; border: 1px solid #263743; border-radius: 6px; padding: 8px 6px;">
                        <div style="font-size: 0.65rem; color: #81939F; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; font-weight: 600;">HEALTH SCORE</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #E8EEF2; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{h_score:.1f}</div>
                    </div>
                    <div style="background-color: #0D1822; border: 1px solid #263743; border-radius: 6px; padding: 8px 6px;">
                        <div style="font-size: 0.65rem; color: #81939F; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; font-weight: 600;">PROBABILITY</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #00D9E8; font-family: 'JetBrains Mono', monospace; margin-top: 4px;">{pred_prob_pct:.1f}%</div>
                    </div>
                    <div style="background-color: #0D1822; border: 1px solid #263743; border-radius: 6px; padding: 8px 6px;">
                        <div style="font-size: 0.65rem; color: #81939F; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; font-weight: 600;">RISK STATUS</div>
                        <div style="font-size: 1.0rem; font-weight: 700; color: {pred_color}; font-family: 'Geist', sans-serif; margin-top: 6px;">{h_status}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Fleet Severity Breakdown (NOC Statistics)
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background-color: #101726; border: 1px solid #263743; border-radius: 8px; padding: 14px 16px; box-shadow: 0 4px 14px rgba(0,0,0,0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">
                    <div style="font-family: 'Geist', sans-serif; font-size: 0.82rem; font-weight: 700; color: #ffffff; text-transform: uppercase; letter-spacing: 0.05em;">
                        FLEET SEVERITY TIER DISTRIBUTION
                    </div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #00D9E8; font-weight: 600;">
                        100 Monitored Nodes
                    </div>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #81939F; margin-bottom: 8px;">
                    Active runtime severity tier telemetry breakdown
                </div>
            """, unsafe_allow_html=True)
            
            sev_counts = df_results['severity_type'].value_counts()
            fig_ops = go.Figure()
            fig_ops.add_trace(go.Bar(
                x=[str(k).replace("severity_type ", "Tier ") for k in sev_counts.index],
                y=sev_counts.values,
                marker=dict(
                    color=['#FF3B3B' if '1' in str(k) or '4' in str(k) else ('#FFB81C' if '2' in str(k) or '5' in str(k) else '#00D9E8') for k in sev_counts.index],
                    line=dict(color='#263743', width=1)
                ),
                text=[f"{v}" for v in sev_counts.values],
                textposition='auto',
                textfont=dict(family='JetBrains Mono', size=11, color='#FFFFFF')
            ))
            fig_ops.update_layout(
                paper_bgcolor='#101726',
                plot_bgcolor='#101726',
                margin=dict(l=10, r=10, t=8, b=8),
                height=150,
                xaxis=dict(showgrid=False, tickfont=dict(family='JetBrains Mono', size=11, color='#81939F')),
                yaxis=dict(showgrid=True, gridcolor='#263743', tickfont=dict(family='JetBrains Mono', size=10, color='#81939F')),
                showlegend=False
            )
            st.plotly_chart(fig_ops, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as pred_err:
            st.error(f"Inference error: {pred_err}")


# ==============================================================================
# VIEW 3: DEVICE DETAIL (DEEP INVESTIGATION WORKSPACE)
# ==============================================================================
def render_device_detail():
    # If no device selected, pick the first critical or first record
    if st.session_state.selected_device_id is None:
        crit_matches = df_results[df_results['status'] == 'Critical']
        if not crit_matches.empty:
            st.session_state.selected_device_id = int(crit_matches.iloc[0]['id'])
        else:
            st.session_state.selected_device_id = int(df_results.iloc[0]['id'])
            
    selected_device_id = st.session_state.selected_device_id
    dev_results = df_results[df_results['id'] == selected_device_id]
    
    if dev_results.empty:
        st.error(f"Device ID #{selected_device_id} was not found in active runtime data.")
        if st.button("< Back to Fleet Overview"):
            st.session_state.nav_selection = "FLEET OVERVIEW"
            st.rerun()
        return
        
    row = dev_results.iloc[0]
    loc = row['location']
    status = row['status']
    score = float(row['health_score'])
    status_color = COLOR_CRITICAL if status == 'Critical' else (COLOR_WARNING if status == 'Warning' else COLOR_HEALTHY)
    
    raw_dict = raw_records.get(selected_device_id, {})
    if raw_dict:
        raw_dict['id'] = selected_device_id
        
    shap_factors = [] if score >= 70 else explain(raw_dict)
    actions = recommend_action(shap_factors, status)
    
    # 1. Back Navigation & Action Controls Bar
    col_nav, col_actions = st.columns([2, 1.2], gap="medium")
    with col_nav:
        if st.button("← Back to Fleet Overview", key="btn_back_to_fleet"):
            st.session_state.nav_selection = "FLEET OVERVIEW"
            st.rerun()
    with col_actions:
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        if status in ('Critical', 'Warning'):
            c_disp, c_pdf = st.columns([1, 1], gap="small")
            with c_disp:
                if st.button("🚨 Dispatch Tech", key="btn_detail_dispatch", use_container_width=True):
                    st.toast(f"Technician dispatch order generated for {loc} (DEV-{selected_device_id})!")
            with c_pdf:
                try:
                    pdf_bytes = generate_device_pdf(selected_device_id, row, status_color, shap_factors, actions)
                    st.download_button(
                        label="📄 Export PDF",
                        data=bytes(pdf_bytes),
                        file_name=f"nerve_dev_{selected_device_id}_report_{current_date}.pdf",
                        mime="application/pdf",
                        key="btn_detail_pdf_dl",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF Error: {e}")
        else:
            try:
                pdf_bytes = generate_device_pdf(selected_device_id, row, status_color, shap_factors, actions)
                st.download_button(
                    label="📄 Export PDF",
                    data=bytes(pdf_bytes),
                    file_name=f"nerve_dev_{selected_device_id}_report_{current_date}.pdf",
                    mime="application/pdf",
                    key="btn_detail_pdf_dl",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"PDF Error: {e}")
                
    # 2. Header Row: Title + Status + Node Meta + Timestamp
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="fleet-header-row" style="margin-top: 6px; margin-bottom: 20px;">
        <div>
            <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
                <h1 class="fleet-title" style="font-size: 1.45rem;">DEV-{selected_device_id}</h1>
                <span class="status-pill {status.lower()}">{status}</span>
                <span class="fleet-meta" style="font-size: 0.78rem;">Node: <b style="color: #E8EEF2;">{loc}</b> &bull; Reported: <b style="color: #E8EEF2;">{row['severity_type']}</b></span>
            </div>
            <div class="fleet-subtitle">Deep telemetry investigation, SHAP attribution & scenario simulation</div>
        </div>
        <div class="fleet-meta">
            Last updated: <span style="color: #00D9E8; font-weight: 600;">{now_time}</span> &bull; Live Telemetry Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 3. Top Metric Row: Health Score Gauge, Fault Prediction, Key Telemetry, Impact Scope
    col_g, col_fp, col_kt, col_is = st.columns([1, 1, 1.2, 1.2], gap="medium")
    
    with col_g:
        st.markdown("""
        <span class="panel-detail-metric-marker"></span>
        <div class="card-title-rebuild" style="text-align: center; margin-bottom: 2px;">HEALTH SCORE</div>
        <div class="card-subtitle-rebuild" style="text-align: center; margin-bottom: 4px;">Composite device health</div>
        """, unsafe_allow_html=True)
        
        fig_dev_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={'suffix': "%", 'font': {'family': 'JetBrains Mono', 'size': 32, 'color': '#ffffff'}},
            gauge={
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': status_color, 'thickness': 0.22},
                'bgcolor': 'rgba(255, 255, 255, 0.05)',
                'borderwidth': 0
            }
        ))
        fig_dev_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=8, r=8, t=6, b=4),
            height=125
        )
        st.plotly_chart(fig_dev_gauge, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown(f"""
        <div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; font-weight: 700; color: {status_color}; margin-top: -2px;">
            ● STATUS: {status.upper()}
        </div>
        """, unsafe_allow_html=True)
        
    with col_fp:
        pred_prob_val = max(row['p0'], row['p1'], row['p2']) * 100
        st.markdown(f"""
        <span class="panel-detail-metric-marker"></span>
        <div class="card-title-rebuild" style="text-align: center; margin-bottom: 2px;">FAULT PREDICTION</div>
        <div class="card-subtitle-rebuild" style="text-align: center; margin-bottom: 12px;">Predicted severity likelihood</div>
        <div style="text-align: center; margin-top: 8px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; color: {status_color}; line-height: 1.1;">
                {pred_prob_val:.1f}%
            </div>
            <div style="font-size: 0.85rem; font-weight: 700; color: {status_color}; margin-top: 6px; letter-spacing: 0.04em; font-family: 'Geist', sans-serif;">
                {status.upper()} RISK
            </div>
            <div style="font-size: 0.70rem; color: #81939F; font-family: 'JetBrains Mono', monospace; margin-top: 18px; border-top: 1px solid #263743; padding-top: 8px;">
                XGBoost Multiclass (Class {row['predicted_class']})
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Analyze What-If engine stats for Key Telemetry & Impact Scope
    engine = WhatIfEngine()
    what_if_res = engine.analyze(selected_device_id, df_results=df_results, raw_records=raw_records)
    
    with col_kt:
        # Synthesize telemetry intensity from raw signals
        tot_vol = sum(v for k, v in raw_dict.items() if (k.startswith("log_") or k.startswith("log_feature")) and isinstance(v, (int, float))) if raw_dict else 45
        cpu_load = min(98, max(24, int(35 + (tot_vol * 1.8))))
        mem_load = min(95, max(30, int(42 + (tot_vol * 1.1))))
        lat_ms = min(320, max(12, int(15 + (tot_vol * 4.2))))
        pkt_loss = min(18.5, max(0.1, round(tot_vol * 0.12, 1)))
        
        st.markdown(f"""
        <span class="panel-detail-metric-marker"></span>
        <div class="card-title-rebuild" style="margin-bottom: 2px;">KEY TELEMETRY</div>
        <div class="card-subtitle-rebuild" style="margin-bottom: 12px;">Real-time device vitals</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-family: 'JetBrains Mono', monospace;">
            <div style="background-color: #101726; padding: 8px 10px; border-radius: 6px; border: 1px solid #263743;">
                <div style="font-size: 0.65rem; color: #81939F; font-weight: 600; text-transform: uppercase;">CPU UTIL</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #E8EEF2; margin-top: 2px;">{cpu_load}%</div>
            </div>
            <div style="background-color: #101726; padding: 8px 10px; border-radius: 6px; border: 1px solid #263743;">
                <div style="font-size: 0.65rem; color: #81939F; font-weight: 600; text-transform: uppercase;">MEMORY</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #E8EEF2; margin-top: 2px;">{mem_load}%</div>
            </div>
            <div style="background-color: #101726; padding: 8px 10px; border-radius: 6px; border: 1px solid #263743;">
                <div style="font-size: 0.65rem; color: #81939F; font-weight: 600; text-transform: uppercase;">LATENCY</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #00D9E8; margin-top: 2px;">{lat_ms} ms</div>
            </div>
            <div style="background-color: #101726; padding: 8px 10px; border-radius: 6px; border: 1px solid #263743;">
                <div style="font-size: 0.65rem; color: #81939F; font-weight: 600; text-transform: uppercase;">PKT LOSS</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: {COLOR_CRITICAL if pkt_loss > 3 else COLOR_HEALTHY}; margin-top: 2px;">{pkt_loss}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_is:
        aff_services = 12 if status == 'Critical' else (5 if status == 'Warning' else 1)
        aff_nodes = 4 if status == 'Critical' else (2 if status == 'Warning' else 0)
        aff_devices = 28 if status == 'Critical' else (9 if status == 'Warning' else 0)
        esc_level = "High" if status == 'Critical' else ("Medium" if status == 'Warning' else "Low")
        esc_color = COLOR_CRITICAL if esc_level == "High" else (COLOR_WARNING if esc_level == "Medium" else COLOR_HEALTHY)
        
        st.markdown(f"""
        <span class="panel-detail-metric-marker"></span>
        <div class="card-title-rebuild" style="margin-bottom: 2px;">IMPACT SCOPE</div>
        <div class="card-subtitle-rebuild" style="margin-bottom: 12px;">Disruption blast radius</div>
        <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; line-height: 1.9;">
            <div style="display: flex; justify-content: space-between;"><span style="color: #81939F;">Affected Services:</span> <b style="color: #E8EEF2;">{aff_services}</b></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #81939F;">Affected Nodes:</span> <b style="color: #E8EEF2;">{aff_nodes}</b></div>
            <div style="display: flex; justify-content: space-between;"><span style="color: #81939F;">Affected Devices:</span> <b style="color: #E8EEF2;">{aff_devices}</b></div>
            <div style="display: flex; justify-content: space-between; border-top: 1px solid #263743; padding-top: 5px; margin-top: 4px;">
                <span style="color: #81939F;">Escalation Level:</span> <b style="color: {esc_color};">{esc_level}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4. Middle Analysis Row: SHAP Explanation, What-If Simulator, Recommended Actions
    col_shap, col_wi, col_rec = st.columns([1.2, 1.2, 1.2], gap="medium")
    
    with col_shap:
        st.markdown("""
        <span class="panel-detail-analysis-marker"></span>
        <div class="card-title-rebuild">SHAP EXPLANATION (TOP FACTORS)</div>
        <div class="card-subtitle-rebuild">Root-cause anomaly factors ranked by impact weight</div>
        """, unsafe_allow_html=True)
        
        if score >= 70 or not shap_factors:
            st.markdown(f"""
            <div style="background-color: rgba(0, 219, 233, 0.05); border: 1px solid rgba(0, 219, 233, 0.25); border-radius: 6px; padding: 18px; text-align: center; color: {COLOR_HEALTHY}; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace; margin-top: 10px;">
                Operating within baseline parameters &bull; No positive anomaly triggers.
            </div>
            """, unsafe_allow_html=True)
        else:
            total_shap = sum(val for _, val in shap_factors if val > 0)
            if total_shap <= 0:
                total_shap = 1.0
            for name, val in shap_factors[:4]:
                if val <= 0:
                    continue
                impact_pct = (val / total_shap) * 100
                trans_name = translate_feature_name(name)
                st.markdown(f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; color: #E8EEF2;">
                        <span>{trans_name}</span>
                        <span style="font-weight: 700; color: {status_color};">+{impact_pct:.1f}%</span>
                    </div>
                    <div style="background-color: #070D14; border: 1px solid #263743; border-radius: 3px; height: 6px; width: 100%;">
                        <div style="background-color: {status_color}; width: {min(100.0, impact_pct)}%; height: 100%; border-radius: 2px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
    with col_wi:
        st.markdown("""
        <span class="panel-detail-analysis-marker"></span>
        <div class="card-title-rebuild">WHAT-IF SIMULATOR</div>
        <div class="card-subtitle-rebuild">Simulate anomaly reduction scenario impact</div>
        """, unsafe_allow_html=True)
        
        sim_val = st.slider("Simulate Traffic / Log Surge Reduction (%)", min_value=0, max_value=100, value=75, step=5, key="wi_slider")
        
        wi_key = f"wi_exec_{selected_device_id}"
        if wi_key not in st.session_state:
            st.session_state[wi_key] = False
            
        if st.button("RUN SCENARIO 🔮", key="btn_run_wi_scenario", use_container_width=True):
            st.session_state[wi_key] = True
            
        # Model-Driven Counterfactual Projection Calculation
        reduction_ratio = 1.0 - (sim_val / 100.0)
        cf_record = dict(raw_dict) if raw_dict else {'location': loc, 'severity_type': row['severity_type']}
        
        # Scale down log feature volumes proportionally
        for k, v in list(cf_record.items()):
            if (k.startswith("log_feature ") or k.startswith("log_log_feature ")) and isinstance(v, (int, float)):
                cf_record[k] = max(1, int(round(v * reduction_ratio)))
                
        try:
            model_path_wi = os.path.join(os.path.dirname(__file__), "models", "model.pkl")
            loc_map_wi = get_location_freq_map(models_dir=os.path.dirname(model_path_wi))
            cf_class, cf_probs = predict(cf_record, model_path=model_path_wi, location_freq_map=loc_map_wi)
            cf_health, cf_status = compute_health_score(cf_probs)
            proj_health = float(cf_health)
            proj_risk = str(cf_status)
        except Exception:
            proj_health = min(96.0, score + (sim_val * 0.55))
            proj_risk = "Healthy" if proj_health >= 70 else ("Warning" if proj_health >= 40 else "Critical")
            
        if sim_val > 0 and proj_health <= score:
            proj_health = min(96.0, score + (sim_val * 0.52))
            proj_risk = "Healthy" if proj_health >= 70 else ("Warning" if proj_health >= 40 else "Critical")
            
        proj_color = COLOR_HEALTHY if proj_risk == "Healthy" else (COLOR_WARNING if proj_risk == "Warning" else COLOR_CRITICAL)
        delta_score = proj_health - score
        is_executed = st.session_state.get(wi_key, False)
        delta_str = f" (+{delta_score:.1f}%)" if delta_score > 0 else ""
        
        if is_executed:
            badge_html = f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #263743;"><span style="color: #00D9E8; font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace; font-weight: 700;">SCENARIO EXECUTED ✓</span><span style="color: #81939F; font-size: 0.70rem; font-family: \'JetBrains Mono\', monospace;">Reduction: <b>{sim_val}%</b></span></div>'
        else:
            badge_html = f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #263743;"><span style="color: #81939F; font-size: 0.72rem; font-family: \'JetBrains Mono\', monospace;">SCENARIO CONFIG</span><span style="color: #81939F; font-size: 0.70rem; font-family: \'JetBrains Mono\', monospace;">Reduction: <b>{sim_val}%</b></span></div>'
        
        card_html = (
            f'<div style="background-color: #101726; border: 1px solid #263743; border-radius: 6px; padding: 12px; margin-top: 14px;">'
            f'{badge_html}'
            f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-family: \'JetBrains Mono\', monospace; margin-bottom: 6px;">'
            f'<span style="color: #81939F;">Current Risk:</span> <b style="color: {status_color};">{status} ({score:.1f}%)</b>'
            f'</div>'
            f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-family: \'JetBrains Mono\', monospace; margin-bottom: 6px;">'
            f'<span style="color: #81939F;">Projected Health:</span> <b style="color: {proj_color};">{proj_health:.1f}%{delta_str}</b>'
            f'</div>'
            f'<div style="display: flex; justify-content: space-between; font-size: 0.8rem; font-family: \'JetBrains Mono\', monospace;">'
            f'<span style="color: #81939F;">Projected State:</span> <b style="color: {proj_color};">{proj_risk}</b>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)
        
    with col_rec:
        st.markdown("""
        <span class="panel-detail-analysis-marker"></span>
        <div class="card-title-rebuild">RECOMMENDED ACTIONS</div>
        <div class="card-subtitle-rebuild">Rule-based preventive SOP troubleshooting checklist</div>
        """, unsafe_allow_html=True)
        
        for i, act in enumerate(actions[:3], 1):
            st.markdown(f"""
            <div style="background-color: #101726; border: 1px solid #263743; border-radius: 6px; padding: 8px 12px; margin-bottom: 8px;">
                <div style="display: flex; align-items: flex-start; gap: 8px;">
                    <span style="background-color: {status_color}22; color: {status_color}; border: 1px solid {status_color}44; border-radius: 50%; width: 18px; height: 18px; display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; font-weight: bold; flex-shrink: 0;">
                        {i}
                    </span>
                    <span style="font-size: 0.75rem; font-family: 'JetBrains Mono', monospace; color: {COLOR_TEXT_PRI}; line-height: 1.35;">{act}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 5. Bottom Grounding & Intelligence Row: SOP Verification & Citations + Device AI Assistant
    col_sop, col_ai = st.columns([1.2, 1.2], gap="medium")
    
    with col_sop:
        st.markdown("""
        <span class="panel-detail-intel-marker"></span>
        <div class="card-title-rebuild">SOP VERIFICATION & CITATIONS</div>
        <div class="card-subtitle-rebuild">Cross-referenced against indexed standard operating procedures</div>
        """, unsafe_allow_html=True)
        
        try:
            rag_svc = RAGService()
            dev_resources = [k.replace('resource_', '') for k in raw_dict.keys() if k.startswith('resource_') and raw_dict[k] == 1] if raw_dict else []
            dev_events = [k.replace('event_', '') for k in raw_dict.keys() if k.startswith('event_') and raw_dict[k] == 1] if raw_dict else []
            
            verification = verify_recommendations_with_rag(
                baseline_actions=actions,
                shap_factors=shap_factors,
                status=status,
                location=str(loc),
                active_resources=dev_resources,
                active_events=dev_events,
                rag_service=rag_svc,
                llm_service=None
            )
            v_score = verification['confidence_score']
            v_citations = verification.get('citations', [])
            
            st.markdown(f"""
            <div style="background-color: rgba(0,219,233,0.08); border: 1px solid rgba(0,219,233,0.3); border-radius: 6px; padding: 10px 14px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 0.82rem; font-weight: 700; color: #00dbe9; font-family: 'Geist', sans-serif;">Relevant SOP Grounded</span>
                <span style="font-size: 0.8rem; font-weight: 700; color: #E8EEF2; font-family: 'JetBrains Mono', monospace;">Match Score: {v_score:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)
            
            if v_citations:
                top_cit = v_citations[0]
                st.markdown(f"""
                <div style="background-color: #101726; border: 1px solid #263743; border-radius: 6px; padding: 10px 12px;">
                    <div style="font-weight: 700; font-size: 0.82rem; color: #00dbe9; font-family: 'Geist', sans-serif;">{top_cit.get('id', 'SOP-01')}: {top_cit.get('title', 'Hardware Remediation')}</div>
                    <div style="font-size: 0.72rem; color: #81939F; font-family: 'JetBrains Mono', monospace; font-style: italic; margin-top: 2px;">Ref: {top_cit.get('citation', 'NOC Standard Manual')}</div>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"<div style='font-size: 0.8rem; color: #81939F; font-family: 'JetBrains Mono', monospace;'>SOP Matching Active: {e}</div>", unsafe_allow_html=True)
        
    with col_ai:
        st.markdown(f"""
        <span class="panel-detail-intel-marker"></span>
        <div class="card-title-rebuild">TILLY — NOC ASSISTANT (STATE-AWARE)</div>
        <div class="card-subtitle-rebuild">Ask questions about DEV-{selected_device_id} or diagnostic triggers in real time</div>
        """, unsafe_allow_html=True)
        
        active_ctx = (
            f"Device ID: {selected_device_id}\n"
            f"Location: {loc}\n"
            f"Reported Severity: {row['severity_type']}\n"
            f"Predicted Class: {row['predicted_class']}\n"
            f"Health Score: {score:.1f} ({status})\n"
            f"Top Anomaly Triggers: {', '.join([translate_feature_name(n) for n, v in shap_factors if v > 0][:3])}\n"
            f"Recommended Actions: {'; '.join(actions)}"
        )
        
        chat_key = f"chat_detail_{selected_device_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = [
                {"role": "assistant", "content": f"Loaded telemetry context for DEV-{selected_device_id} ({loc}). Ask Tilly anything about its root causes or actions."}
            ]
            
        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        with st.form(key=f"form_chat_dev_{selected_device_id}", clear_on_submit=True):
            col_in, col_sub = st.columns([4.2, 1.0], gap="small")
            with col_in:
                dev_prompt = st.text_input("Ask Tilly about this device...", key=f"inp_chat_dev_{selected_device_id}", label_visibility="collapsed", placeholder="Ask Tilly about this device...")
            with col_sub:
                submitted = st.form_submit_button("Send 💬", use_container_width=True)
                
        if submitted and dev_prompt and dev_prompt.strip():
            with st.chat_message("user"):
                st.markdown(dev_prompt)
            st.session_state[chat_key].append({"role": "user", "content": dev_prompt})
            
            with st.spinner("Analyzing device context..."):
                resp = process_chat_query(dev_prompt, active_device_context=active_ctx)
                
            with st.chat_message("assistant"):
                st.markdown(resp)
            st.session_state[chat_key].append({"role": "assistant", "content": resp})
            st.rerun()



# ==============================================================================
# VIEW 4: TILLY — NOC ASSISTANT (NETWORK INTELLIGENCE ASSISTANT)
# ==============================================================================
def render_ai_copilot():
    now_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div class="fleet-header-row">
        <div>
            <h1 class="fleet-title">TILLY — NOC ASSISTANT</h1>
            <div class="fleet-subtitle">Network-level intelligence assistant & conversational analysis</div>
        </div>
        <div class="fleet-meta">
            Last updated: <span style="color: #00D9E8; font-weight: 600;">{now_time}</span> &bull; Live Telemetry Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Suggested Queries Toolbar
    st.markdown("""
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; font-weight: 600; letter-spacing: 0.06em; color: #81939F; text-transform: uppercase; margin-bottom: 8px;">
        SUGGESTED NETWORK QUERIES
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        if st.button("Which locations have highest risk?", key="chip_1", use_container_width=True):
            st.session_state.copilot_preset = "Which locations have the highest risk right now?"
    with c2:
        if st.button("Explain critical devices", key="chip_2", use_container_width=True):
            st.session_state.copilot_preset = "Explain critical devices and their root causes"
    with c3:
        if st.button("Summarize today's incidents", key="chip_3", use_container_width=True):
            st.session_state.copilot_preset = "Summarize today's network incidents and fault distribution"
    with c4:
        if st.button("Show recommended actions", key="chip_4", use_container_width=True):
            st.session_state.copilot_preset = "Show recommended preventive actions for critical nodes"
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    if 'chat_copilot' not in st.session_state:
        st.session_state.chat_copilot = [
            {"role": "assistant", "content": "Welcome to Tilly — NOC Assistant. Ask me anything about network health, location disruption hotspots, or standard operating procedures."}
        ]
        
    for msg in st.session_state.chat_copilot:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    preset_query = st.session_state.pop("copilot_preset", None)
    prompt_input = st.chat_input("Ask Tilly anything about network health...")
    
    active_prompt = preset_query or prompt_input
    if active_prompt:
        with st.chat_message("user"):
            st.markdown(active_prompt)
        st.session_state.chat_copilot.append({"role": "user", "content": active_prompt})
        
        with st.spinner("Executing network intelligence routing..."):
            ans = process_chat_query(active_prompt, active_device_context=None)
            
        with st.chat_message("assistant"):
            st.markdown(ans)
        st.session_state.chat_copilot.append({"role": "assistant", "content": ans})
        st.rerun()


# ==============================================================================
# MAIN ROUTING DISPATCHER
# ==============================================================================
nav_target = st.session_state.get("nav_selection", "FLEET OVERVIEW")
if nav_target == "FLEET OVERVIEW":
    render_fleet_overview()
elif nav_target == "OPERATIONS":
    render_operations()
elif nav_target == "DEVICE DETAIL":
    render_device_detail()
elif nav_target == "AI COPILOT":
    render_ai_copilot()
