"""Look & feel for DataPilot: CSS + plotly styling."""
import html

PALETTE = ["#2DE2C4", "#FFB84D", "#7C8CFF", "#FF6B9A", "#6EE7FF", "#B6F36B"]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');
:root{--accent:#2DE2C4;--amber:#FFB84D;--violet:#7C8CFF;--card:rgba(255,255,255,.04);--line:rgba(255,255,255,.10);--muted:#9AA6C7;}
html, body, .stApp, [data-testid="stMarkdownContainer"]{font-family:'Inter',system-ui,sans-serif;}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 560px at 88% -8%, rgba(124,140,255,.20), transparent 60%),
    radial-gradient(900px 520px at -8% 8%, rgba(45,226,196,.13), transparent 55%),
    #0B1020;}
[data-testid="stHeader"]{background:transparent;}
.block-container{padding-top:2.2rem;max-width:1080px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0E1630 0%,#0A0F22 100%);border-right:1px solid var(--line);}

/* brand + hero */
.brand{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.55rem;letter-spacing:-.02em;
  background:linear-gradient(90deg,var(--accent),var(--violet));-webkit-background-clip:text;background-clip:text;color:transparent;}
.brand small{display:block;font-family:'Inter';font-weight:500;font-size:.72rem;color:var(--muted);letter-spacing:.04em;-webkit-text-fill-color:var(--muted);}
.label{font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin:1.1rem 0 .3rem;font-weight:600;}
.hero h1{font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:3rem;line-height:1.05;letter-spacing:-.03em;margin:0 0 .6rem;}
.hero h1 span{background:linear-gradient(90deg,var(--accent) 0%,var(--violet) 55%,var(--amber) 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.hero p{color:var(--muted);font-size:1.05rem;max-width:640px;margin:0 0 1.4rem;}
.hero.compact h1{font-size:2rem;margin-bottom:.2rem}.hero.compact p{margin-bottom:1rem;font-size:.95rem}

/* pipeline + feature cards */
.pipe{display:flex;flex-wrap:wrap;gap:.45rem;align-items:center;margin:.2rem 0 1.6rem;}
.pipe .node{padding:.38rem .8rem;border:1px solid var(--line);background:var(--card);border-radius:999px;font-size:.82rem;color:#D7DEF5;}
.pipe .arrow{color:var(--accent);font-weight:700;}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.2rem;}
.fcard{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.2rem;}
.fcard .n{font-family:'Space Grotesk';font-size:.8rem;color:var(--accent);letter-spacing:.12em;}
.fcard h3{margin:.25rem 0 .3rem;font-size:1.05rem;font-family:'Space Grotesk';}
.fcard p{margin:0;color:var(--muted);font-size:.9rem;line-height:1.45;}
.hint{border:1px dashed rgba(45,226,196,.45);background:rgba(45,226,196,.06);border-radius:14px;padding:.9rem 1.1rem;color:#CFF7EF;}

/* data tiles */
[data-testid="stMetric"]{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.9rem 1.1rem;}
[data-testid="stMetricLabel"]{color:var(--muted);}
[data-testid="stMetricValue"]{font-family:'Space Grotesk';}

/* chat */
[data-testid="stChatMessage"]{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1rem 1.2rem;margin-bottom:.8rem;}
.insight{border-left:3px solid var(--amber);background:rgba(255,184,77,.09);border-radius:12px;padding:.8rem 1rem;margin:.3rem 0 .8rem;color:#FFE9C7;}
.insight b{color:var(--amber);letter-spacing:.1em;font-size:.7rem;display:block;margin-bottom:.2rem;}
.bigstat{font-family:'Space Grotesk';font-size:2.4rem;font-weight:700;color:var(--accent);margin:.2rem 0;}
.tags{display:flex;gap:.4rem;flex-wrap:wrap;margin:.2rem 0 .6rem;}
.tag{font-size:.72rem;padding:.18rem .6rem;border-radius:999px;border:1px solid var(--line);color:var(--muted);background:rgba(255,255,255,.03);}
.tag.ok{color:var(--accent);border-color:rgba(45,226,196,.4);}.tag.warn{color:var(--amber);border-color:rgba(255,184,77,.4);}

/* buttons / tabs / inputs */
.stButton>button,.stDownloadButton>button{border-radius:999px;border:1px solid var(--line);background:var(--card);color:#E8ECF8;transition:all .15s;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:var(--accent);color:var(--accent);transform:translateY(-1px);}
button[data-baseweb="tab"]{font-weight:600;}
[data-testid="stChatInput"]{border-radius:16px;}
a{color:var(--accent)!important;}
@media (max-width:800px){.grid{grid-template-columns:1fr}.hero h1{font-size:2.2rem}}
"""


def esc(x) -> str:
    return html.escape(str(x))


def style_fig(fig):
    """Make every plotly chart match the app theme."""
    fig.update_layout(
        template="plotly_dark", colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#D7DEF5"),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=20)),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)")
    return fig
