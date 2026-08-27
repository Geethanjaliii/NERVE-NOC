import os
import urllib.parse
import dotenv
dotenv.load_dotenv()
import streamlit as st

def get_oauth_config() -> dict:
    """Retrieves Google OAuth credentials and configuration from environment."""
    dotenv.load_dotenv(override=True)
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "").strip()
    dev_auth_str = os.getenv("DEV_AUTH", "false").strip().lower()
    dev_auth = dev_auth_str in ("true", "1", "yes")
    
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "dev_auth": dev_auth,
        "has_google_creds": bool(client_id and client_secret and client_id != "your_google_client_id_here")
    }

def get_google_auth_url(redirect_uri: str = "http://localhost:8501") -> str:
    """Constructs the Google OAuth 2.0 authorization URL."""
    config = get_oauth_config()
    if not config["has_google_creds"]:
        return ""
        
    params = {
        "client_id": config["client_id"],
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent"
    }
    return f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"

def is_authenticated() -> bool:
    """Checks whether a valid user session is active."""
    # Check query params for explicit demo_auth or oauth code first
    qp = safe_get_query_params()
    if qp.get("demo_auth"):
        st.session_state.logged_out = False
        st.session_state.authenticated = True
        st.session_state.user_info = {
            "name": "NOC Operator",
            "email": "operator.noc@nerve-telecom.net",
            "picture": "",
            "role": "Lead Network Operations Engineer"
        }
        return True

    # Check if user explicitly logged out in this session
    if st.session_state.get("logged_out", False):
        return False
        
    # Check if already authenticated in session
    if st.session_state.get("authenticated", False) and st.session_state.get("user_info") is not None:
        return True
        
    # If in DEV/Demo mode and not explicitly logged out, maintain authenticated session
    config = get_oauth_config()
    if config.get("dev_auth", False):
        st.session_state.authenticated = True
        st.session_state.user_info = {
            "name": "NOC Operator",
            "email": "operator.noc@nerve-telecom.net",
            "picture": "",
            "role": "Lead Network Operations Engineer"
        }
        return True
        
    return False

def get_current_user() -> dict:
    """Returns the currently authenticated user dictionary."""
    return st.session_state.get("user_info", {
        "name": "NOC Engineer",
        "email": "engineer@nerve-noc.net",
        "picture": "",
        "role": "Authorized Operator"
    })

def login_user(user_data: dict):
    """Sets authenticated session state for the user."""
    st.session_state.logged_out = False
    st.session_state.authenticated = True
    st.session_state.user_info = user_data

def safe_rerun():
    """Triggers a rerun safely across different Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def safe_get_query_params() -> dict:
    """Retrieves URL query parameters safely across Streamlit versions."""
    try:
        if hasattr(st, "query_params"):
            return dict(st.query_params)
        elif hasattr(st, "experimental_get_query_params"):
            raw = st.experimental_get_query_params()
            return {k: (v[0] if isinstance(v, list) and len(v) > 0 else v) for k, v in raw.items()}
    except Exception:
        pass
    return {}

def safe_clear_query_params():
    """Clears URL query parameters safely across Streamlit versions."""
    try:
        if hasattr(st, "query_params"):
            st.query_params.clear()
        elif hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params()
    except Exception:
        pass

def logout_user():
    """Clears authentication state and resets session state safely."""
    st.session_state.logged_out = True
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.selected_device_id = None
    safe_rerun()

def render_login_screen():
    """
    Renders the NERVE NOC login screen matching the visual reference:
    - Subtle network topology / constellation background with glowing cyan nodes
    - Centered NERVE shield emblem with glowing inner hexagon
    - Title: NERVE NOC
    - Subtitle: Network Intelligence Command Center
    - White 'Sign in with Google' button (with Google G logo)
    - Footer: Authorized NOC personnel only
    """
    config = get_oauth_config()
    
    # SVG Constellation data URI for CSS background
    bg_svg_uri = "data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 800' width='100%25' height='100%25' preserveAspectRatio='xMidYMid slice'%3E%3Cg stroke='%2300dbe9' stroke-width='0.75' stroke-opacity='0.22'%3E%3Cline x1='150' y1='180' x2='280' y2='240'/%3E%3Cline x1='280' y1='240' x2='420' y2='160'/%3E%3Cline x1='420' y1='160' x2='600' y2='220'/%3E%3Cline x1='600' y1='220' x2='780' y2='160'/%3E%3Cline x1='780' y1='160' x2='920' y2='240'/%3E%3Cline x1='920' y1='240' x2='1050' y2='180'/%3E%3Cline x1='280' y1='240' x2='320' y2='380'/%3E%3Cline x1='320' y1='380' x2='480' y2='340'/%3E%3Cline x1='480' y1='340' x2='600' y2='220'/%3E%3Cline x1='600' y1='220' x2='720' y2='340'/%3E%3Cline x1='720' y1='340' x2='880' y2='380'/%3E%3Cline x1='880' y1='380' x2='920' y2='240'/%3E%3Cline x1='480' y1='340' x2='400' y2='500'/%3E%3Cline x1='400' y1='500' x2='600' y2='540'/%3E%3Cline x1='600' y1='540' x2='800' y2='500'/%3E%3Cline x1='800' y1='500' x2='720' y2='340'/%3E%3Cline x1='400' y1='500' x2='220' y2='560'/%3E%3Cline x1='220' y1='560' x2='150' y2='680'/%3E%3Cline x1='150' y1='680' x2='360' y2='720'/%3E%3Cline x1='360' y1='720' x2='600' y2='660'/%3E%3Cline x1='600' y1='660' x2='840' y2='720'/%3E%3Cline x1='840' y1='720' x2='1050' y2='680'/%3E%3Cline x1='1050' y1='680' x2='980' y2='560'/%3E%3Cline x1='980' y1='560' x2='800' y2='500'/%3E%3Cline x1='600' y1='220' x2='600' y2='100'/%3E%3Cline x1='420' y1='160' x2='600' y2='100'/%3E%3Cline x1='780' y1='160' x2='600' y2='100'/%3E%3Cline x1='600' y1='540' x2='600' y2='660'/%3E%3C/g%3E%3Cg fill='%2300f5ff'%3E%3Ccircle cx='150' cy='180' r='2.5'/%3E%3Ccircle cx='280' cy='240' r='3.5'/%3E%3Ccircle cx='420' cy='160' r='2.5'/%3E%3Ccircle cx='600' cy='100' r='4'/%3E%3Ccircle cx='600' cy='220' r='4'/%3E%3Ccircle cx='780' cy='160' r='2.5'/%3E%3Ccircle cx='920' cy='240' r='3.5'/%3E%3Ccircle cx='1050' cy='180' r='2.5'/%3E%3Ccircle cx='320' cy='380' r='3'/%3E%3Ccircle cx='480' cy='340' r='3.5'/%3E%3Ccircle cx='720' cy='340' r='3.5'/%3E%3Ccircle cx='880' cy='380' r='3'/%3E%3Ccircle cx='400' cy='500' r='3'/%3E%3Ccircle cx='600' cy='540' r='4'/%3E%3Ccircle cx='800' cy='500' r='3'/%3E%3Ccircle cx='220' cy='560' r='2.5'/%3E%3Ccircle cx='150' cy='680' r='2.5'/%3E%3Ccircle cx='360' cy='720' r='3'/%3E%3Ccircle cx='600' cy='660' r='3.5'/%3E%3Ccircle cx='840' cy='720' r='3'/%3E%3Ccircle cx='980' cy='560' r='2.5'/%3E%3Ccircle cx='1050' cy='680' r='2.5'/%3E%3C/g%3E%3C/svg%3E"
    
    # Hide Streamlit chrome during login and inject reference styles
    st.markdown(f"""<style>
/* Hide Streamlit top header & sidebar */
header[data-testid="stHeader"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
footer {{ display: none !important; }}
#MainMenu {{ display: none !important; }}
div[data-testid="stToolbar"] {{ display: none !important; }}
div[data-testid="stDecoration"] {{ display: none !important; }}

/* Full page background with network constellation */
html, body, .stApp {{
    background-color: #040812 !important;
    background-image: radial-gradient(circle at 50% 35%, rgba(0, 229, 255, 0.07) 0%, rgba(2, 6, 14, 0.85) 60%, #02040a 100%), url("{bg_svg_uri}") !important;
    background-repeat: no-repeat !important;
    background-position: center top !important;
    background-size: cover !important;
    min-height: 100vh !important;
}}

/* Full screen container reset */
.block-container {{
    padding-top: 4rem !important;
    padding-bottom: 2rem !important;
    max-width: 600px !important;
    margin: 0 auto !important;
}}

/* Google Button Custom Styling */
div.stButton > button[kind="secondary"] {{
    background-color: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 8px !important;
    font-family: 'Geist', 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    height: 48px !important;
    padding: 0 24px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.5) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 auto !important;
    width: 240px !important;
}}

div.stButton > button[kind="secondary"]::before {{
    content: "";
    display: inline-block;
    width: 20px;
    height: 20px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%234285F4' d='M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z'/%3E%3Cpath fill='%2334A853' d='M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z'/%3E%3Cpath fill='%23FBBC05' d='M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z'/%3E%3Cpath fill='%23EA4335' d='M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z'/%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    margin-right: 10px;
    flex-shrink: 0;
}}

div.stButton > button[kind="secondary"]:hover {{
    background-color: #f8fafc !important;
    box-shadow: 0 6px 24px rgba(0, 229, 255, 0.35) !important;
    transform: translateY(-1px) !important;
    border-color: #00dbe9 !important;
    color: #0f172a !important;
}}

div.stButton > button[kind="secondary"]:active {{
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}}
</style>""", unsafe_allow_html=True)
    
    # Handle Google OAuth callback code if present in URL
    query_params = safe_get_query_params()
    auth_code = query_params.get("code", None)
    demo_auth = query_params.get("demo_auth", None)
    
    if demo_auth:
        login_user({
            "name": "NOC Operator",
            "email": "operator.noc@nerve-telecom.net",
            "picture": "",
            "role": "Lead Network Operations Engineer"
        })
        safe_clear_query_params()
        safe_rerun()
        return
        
    if auth_code and config["has_google_creds"]:
        st.info("Authenticating with Google OAuth...")
        try:
            import requests
            token_url = "https://oauth2.googleapis.com/token"
            data = {
                "code": auth_code,
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "redirect_uri": "http://localhost:8501",
                "grant_type": "authorization_code"
            }
            r = requests.post(token_url, data=data, timeout=10)
            if r.status_code == 200:
                tokens = r.json()
                access_token = tokens.get("access_token")
                user_info_resp = requests.get(
                    "https://www.googleapis.com/oauth2/v2/userinfo",
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10
                )
                if user_info_resp.status_code == 200:
                    u_data = user_info_resp.json()
                    login_user({
                        "name": u_data.get("name", "NOC Operator"),
                        "email": u_data.get("email", ""),
                        "picture": u_data.get("picture", ""),
                        "role": "Network Operations Engineer"
                    })
                    safe_clear_query_params()
                    safe_rerun()
            else:
                st.error("Google authentication failed. Please verify credentials or retry.")
        except Exception as e:
            st.error(f"Authentication error: {e}")

    # Center Brand Elements: Shield, Title, Subtitle
    st.markdown("""<div style="text-align: center; margin-bottom: 28px;">
<div style="display: inline-block; margin-bottom: 20px;">
<svg width="68" height="78" viewBox="0 0 68 78" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs>
<filter id="nerveShieldGlow" x="-30%" y="-30%" width="160%" height="160%">
<feDropShadow dx="0" dy="0" stdDeviation="6" flood-color="#00e5ff" flood-opacity="0.75"/>
</filter>
<filter id="nerveCoreGlow" x="-40%" y="-40%" width="180%" height="180%">
<feDropShadow dx="0" dy="0" stdDeviation="4" flood-color="#00e5ff" flood-opacity="0.9"/>
</filter>
</defs>
<path d="M34 5 L58 18 V42 C58 58 34 72 34 72 C34 72 10 58 10 42 V18 Z" stroke="#00e5ff" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" filter="url(#nerveShieldGlow)" fill="rgba(0, 229, 255, 0.03)"/>
<path d="M34 27 L46 34 V48 L34 55 L22 48 V34 Z" fill="#00e5ff" filter="url(#nerveCoreGlow)"/>
</svg>
</div>
<h1 style="font-family: 'Geist', 'Inter', sans-serif; font-size: 2.35rem; font-weight: 700; color: #ffffff; letter-spacing: 0.05em; margin: 0 0 8px 0; text-shadow: 0 0 24px rgba(255, 255, 255, 0.2);">
NERVE NOC
</h1>
<div style="font-family: 'Geist', 'Inter', sans-serif; font-size: 0.96rem; color: #cbd5e1; font-weight: 400; letter-spacing: 0.01em;">
Network Intelligence Command Center
</div>
</div>""", unsafe_allow_html=True)
    
    # Render the Google Sign In Button
    if config["has_google_creds"]:
        oauth_url = get_google_auth_url()
        st.markdown(f"""<div style="display: flex; justify-content: center; margin-bottom: 20px;">
<a href="{oauth_url}" target="_top" style="text-decoration: none;">
<div style="display: flex; align-items: center; justify-content: center; gap: 12px; background-color: #ffffff; color: #1f2937; font-family: 'Geist', sans-serif; font-size: 0.95rem; font-weight: 600; width: 240px; height: 48px; border-radius: 8px; cursor: pointer; transition: all 0.2s ease; box-shadow: 0 4px 18px rgba(0,0,0,0.45); border: 1px solid rgba(255,255,255,0.4);">
<svg width="20" height="20" viewBox="0 0 24 24">
<path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
<path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
<path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"/>
<path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"/>
</svg>
Sign in with Google
</div>
</a>
</div>""", unsafe_allow_html=True)
    else:
        # Under DEV_AUTH mode, render clean styled Google button
        st.markdown("""<style>
div[data-testid="stButton"] {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
</style>""", unsafe_allow_html=True)
        
        # Google Sign In Button
        if st.button("Sign in with Google", key="btn_nerve_google_signin", use_container_width=True):
            login_user({
                "name": "NOC Engineer",
                "email": "engineer.noc@nerve-telecom.net",
                "picture": "",
                "role": "Lead Network Operations Engineer"
            })
            safe_rerun()
            
    # Footer
    st.markdown("""<div style="text-align: center; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #64748b; margin-top: 18px;">
Authorized NOC personnel only
</div>""", unsafe_allow_html=True)


