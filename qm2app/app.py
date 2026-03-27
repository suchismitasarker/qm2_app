#!/usr/bin/env python3
"""
QM2 Unified Data Analysis App  (with 1D Linecut)
"""

from flask import Flask, render_template_string, request
import os, io, time, base64, hashlib, threading, re, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import h5py
from nexusformat.nexus import nxload, NXdata, NXfield
from nxs_analysis_tools import plot_slice
import fabio, pyFAI
import pandas as pd

app = Flask(__name__)

ROOT            = "/nfs/chess/id4baux/2026-1"
PYFAI_IMG_ROOT  = "/nfs/chess/id4b/2026-1"
CACHE_DIR       = "/tmp/lslice_cache"
MAX_IO_WORKERS  = 4

os.makedirs(CACHE_DIR, exist_ok=True)

# ── CHESS logo (loaded once at startup, embedded as base64) ──────────────────
_CHESS_LOGO_B64 = None
try:
    with open("/nfs/chess/id4baux/chesslogo.png", "rb") as _f:
        _CHESS_LOGO_B64 = base64.b64encode(_f.read()).decode("utf-8")
except Exception:
    _CHESS_LOGO_B64 = None

_cache_lock     = threading.Lock()
_sv_file_cache  = {}
_sv_meta_cache  = {}
_sv_slice_cache = {}
_GRID_LINES     = None
_hb_ai_cache         = {}   # poni_path → AzimuthalIntegrator (caches LUT across requests)


# ── BASE HTML (sidebar has Linecut link added) ────────────────────────────────
BASE_HTML = """
<!DOCTYPE html><html><head>
    <title>Quantum Materials Beamline Data Analysis @CHESS</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root{--bg:#f5f7fb;--bg-card:#ffffff;--bg-header:#0d1520;--bg-sidebar:#111827;--text-main:#222;--text-muted:#555;--text-header:#e8f0f8;--border-subtle:#d0d4e0;--accent:#1b4f72;--accent-hover:#163f5a;--accent-glow:#00c8ff;}
        body.dark{--bg:#050816;--bg-card:#0d1520;--bg-header:#020617;--bg-sidebar:#020617;--text-main:#e5e7eb;--text-muted:#9ca3af;--text-header:#e5e7eb;--border-subtle:#1e3a5f;--accent:#00c8ff;--accent-hover:#0082b8;}
        *{box-sizing:border-box;}
        body{margin:0;font-family:'IBM Plex Sans',-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text-main);}
        .app-shell{display:flex;min-height:100vh;}
        .sidebar{width:240px;background:var(--bg-sidebar);color:var(--text-header);padding:0;display:flex;flex-direction:column;flex-shrink:0;}
        .sidebar-brand{padding:16px;display:flex;align-items:center;gap:12px;border-bottom:1px solid rgba(255,255,255,0.06);}
        .logo-circle{width:36px;height:36px;border-radius:50%;background:conic-gradient(from 0deg,#00c8ff 0%,#0047ab 45%,#00c8ff 100%);display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;color:#fff;box-shadow:0 0 14px rgba(0,200,255,0.3);flex-shrink:0;}
        .logo-text-main{font-size:14px;font-weight:600;line-height:1.2;}
        .logo-text-sub{font-size:11px;opacity:0.6;letter-spacing:0.04em;}
        .nav-section-title{font-size:10px;text-transform:uppercase;letter-spacing:0.1em;color:rgba(255,255,255,0.35);padding:14px 16px 4px;}
        .nav-link{display:flex;align-items:center;gap:8px;padding:8px 16px;color:rgba(255,255,255,0.65);text-decoration:none;font-size:13px;transition:background 0.15s,color 0.15s;border-left:2px solid transparent;}
        .nav-link:hover{background:rgba(255,255,255,0.06);color:#fff;}
        .nav-link.active{background:rgba(0,200,255,0.08);color:#00c8ff;border-left-color:#00c8ff;}
        .nav-icon{font-size:14px;width:18px;text-align:center;}
        .sidebar-footer{margin-top:auto;padding:12px 16px;font-size:11px;color:rgba(255,255,255,0.3);border-top:1px solid rgba(255,255,255,0.06);}
        .main-column{flex:1;display:flex;flex-direction:column;min-width:0;}
        header{background:var(--bg-header);color:var(--text-header);padding:12px 28px;display:flex;align-items:center;justify-content:space-between;box-shadow:0 2px 12px rgba(0,0,0,0.4);position:sticky;top:0;z-index:50;}
        header h1{margin:0;font-size:17px;font-weight:500;letter-spacing:0.01em;}
        header p{margin:2px 0 0;font-size:11px;opacity:0.6;}
        .header-badge{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#00c8ff;background:rgba(0,200,255,0.08);border:1px solid rgba(0,200,255,0.2);border-radius:20px;padding:3px 10px;}
        .theme-toggle{border-radius:999px;border:1px solid rgba(148,163,184,0.4);padding:4px 12px;background:transparent;color:var(--text-header);font-size:11px;cursor:pointer;transition:background 0.15s;}
        .theme-toggle:hover{background:rgba(148,163,184,0.15);}
        main{padding:24px 28px 36px;flex:1;overflow-y:auto;}
        .card{background:var(--bg-card);border-radius:8px;padding:18px 22px;margin-bottom:20px;box-shadow:0 1px 4px rgba(15,23,42,0.2);border:1px solid var(--border-subtle);}
        .card-header{display:flex;align-items:center;gap:8px;padding-bottom:12px;margin-bottom:14px;border-bottom:1px solid var(--border-subtle);}
        .card-dot{width:6px;height:6px;border-radius:50%;background:var(--accent-glow);box-shadow:0 0 8px var(--accent-glow);}
        .card-title{font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:var(--text-muted);}
        .section-title{margin:0 0 8px;font-size:18px;font-weight:600;}
        .subheading{font-size:13px;color:var(--text-muted);margin:0 0 10px;}
        a{color:var(--accent);text-decoration:none;}
        a:hover{text-decoration:underline;}
        ul{line-height:1.7;padding-left:20px;margin:0;}
        .path-label{font-family:'IBM Plex Mono',monospace;background:rgba(148,163,184,0.1);padding:2px 6px;border-radius:4px;font-size:12px;}
        .field-label{font-size:12px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:var(--text-muted);margin-bottom:5px;display:block;}
        .field-group{margin-bottom:14px;}
        select,input[type="number"],input[type="text"]{padding:6px 10px;border-radius:5px;border:1px solid var(--border-subtle);font-size:13px;font-family:'IBM Plex Sans',sans-serif;background:var(--bg-card);color:var(--text-main);outline:none;transition:border-color 0.15s,box-shadow 0.15s;}
        select:focus,input:focus{border-color:var(--accent);box-shadow:0 0 0 2px rgba(0,200,255,0.1);}
        select{min-width:220px;cursor:pointer;}
        input[type="number"]{width:120px;}
        input[type="text"]{width:100%;max-width:480px;}
        input[type="checkbox"],input[type="radio"]{accent-color:var(--accent);}
        select[multiple]{min-width:100%;max-width:540px;min-height:140px;padding:4px;}
        .inline-fields{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-end;}
        .inline-fields .field-group{margin-bottom:0;}
        .inline-form{display:inline-block;margin-right:8px;}
        button,.btn{background:var(--accent);color:#f9fafb;border:none;border-radius:5px;padding:7px 16px;cursor:pointer;font-size:13px;font-family:'IBM Plex Sans',sans-serif;font-weight:500;transition:background 0.15s,box-shadow 0.15s;}
        button:hover{background:var(--accent-hover);}
        .btn-primary{background:linear-gradient(135deg,#0082b8 0%,#00c8ff 100%);font-weight:600;letter-spacing:0.04em;box-shadow:0 2px 14px rgba(0,200,255,0.2);text-transform:uppercase;font-size:12px;padding:9px 22px;}
        .btn-primary:hover{opacity:0.88;box-shadow:0 4px 20px rgba(0,200,255,0.35);}
        #status{font-size:13px;color:var(--accent);margin-top:10px;display:none;}
        .progress-container{width:100%;background:var(--border-subtle);border-radius:3px;overflow:hidden;margin-top:8px;display:none;height:3px;}
        .progress-bar{height:100%;background:linear-gradient(90deg,var(--accent-hover),var(--accent-glow));animation:prog 1.8s ease-in-out infinite;}
        @keyframes prog{0%{margin-left:-40%;width:30%}50%{margin-left:30%;width:50%}100%{margin-left:100%;width:30%}}
        .missing{color:#f97373;}.ok{color:#4ade80;}.warn-text{color:#fbbf24;}
        .alert-warn{color:#92400e;background:rgba(240,180,41,0.08);border:1px solid rgba(240,180,41,0.25);border-radius:5px;padding:8px 12px;font-size:12px;margin-bottom:12px;}
        .alert-info{color:#1e40af;background:#eff6ff;border:1px solid #93c5fd;border-radius:5px;padding:10px 14px;font-size:12px;margin-bottom:16px;}
        .alert-error{color:#dc2626;background:rgba(255,79,79,0.06);border:1px solid rgba(255,79,79,0.2);border-radius:5px;padding:10px 12px;font-size:13px;}
        table{border-collapse:collapse;width:100%;font-size:13px;}
        th,td{border:1px solid var(--border-subtle);padding:7px 10px;}
        th{background:rgba(148,163,184,0.15);text-align:left;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:0.05em;}
        .log-table td{font-family:'IBM Plex Mono',monospace;font-size:12px;}
        img{max-width:900px;border-radius:5px;border:1px solid var(--border-subtle);}
        .badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;font-weight:600;margin-left:6px;}
        .badge-blue{background:rgba(56,189,248,0.15);color:#38bdf8;}
        .badge-green{background:rgba(74,222,128,0.15);color:#4ade80;}
        .badge-red{background:rgba(249,115,115,0.15);color:#f97373;}
        .badge-gold{background:rgba(240,180,41,0.12);color:#f0b429;}
        pre{background:#040810;border:1px solid var(--border-subtle);border-radius:5px;padding:12px 14px;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#9ca3af;max-height:220px;overflow-y:auto;line-height:1.6;}
        .two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
        hr{border:none;border-top:1px solid var(--border-subtle);margin:18px 0;}
        footer{font-size:11px;padding:10px 28px;color:var(--text-muted);border-top:1px solid var(--border-subtle);}
        .tab-bar{display:flex;border-bottom:1px solid var(--border-subtle);margin-bottom:18px;}
        .tab-btn{padding:8px 18px;font-size:12px;font-weight:600;font-family:inherit;letter-spacing:0.04em;color:var(--text-muted);background:transparent;border:none;border-bottom:2px solid transparent;cursor:pointer;transition:color 0.15s,border-color 0.15s;text-transform:uppercase;}
        .tab-btn:hover{color:var(--text-main);}
        .tab-btn.active{color:var(--accent);border-bottom-color:var(--accent);}
        .tab-panel{display:none;}.tab-panel.active{display:block;}
        .slice-grid{display:flex;gap:12px;flex-wrap:wrap;}
        .slice-box{flex:0 0 auto;min-width:200px;max-width:380px;}
        .single-box{max-width:380px;}
        .slice-filename{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--text-muted);margin-bottom:5px;display:flex;align-items:center;gap:6px;}
        .slice-timing{font-size:10px;color:#6b7280;margin-left:auto;}
        .slice-dot{width:6px;height:6px;background:var(--accent-glow);border-radius:50%;box-shadow:0 0 6px var(--accent-glow);flex-shrink:0;}
        .slice-img{width:100%;border:1px solid var(--border-subtle);border-radius:5px;display:block;}
        .dl-btn{display:inline-flex;align-items:center;gap:4px;margin-top:5px;font-size:11px;font-family:'IBM Plex Mono',monospace;color:var(--text-muted);text-decoration:none;padding:3px 9px;border:1px solid var(--border-subtle);border-radius:4px;background:rgba(0,0,0,0.04);transition:background 0.15s,color 0.15s;}
        .dl-btn:hover{background:rgba(0,200,255,0.08);color:#00c8ff;border-color:rgba(0,200,255,0.3);text-decoration:none;}
        .meta-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px;}
        .meta-chip{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:3px 10px;border-radius:20px;background:rgba(0,200,255,0.06);border:1px solid rgba(0,200,255,0.15);color:var(--text-muted);}
        .meta-chip b{color:var(--accent-glow);}
        ::-webkit-scrollbar{width:6px;height:6px;}
        ::-webkit-scrollbar-track{background:transparent;}
        ::-webkit-scrollbar-thumb{background:var(--border-subtle);border-radius:3px;}
    </style>
    <script>
        function applyThemeFromStorage(){if(localStorage.getItem('qm2_theme')==='dark')document.body.classList.add('dark');}
        function toggleTheme(){document.body.classList.toggle('dark');localStorage.setItem('qm2_theme',document.body.classList.contains('dark')?'dark':'light');}
        function showStatus(){var s=document.getElementById("status");if(s){s.style.display="block";s.innerText="⟳  Processing — please wait...";}var p=document.getElementById("progress-container");if(p)p.style.display="block";}
        function switchTab(name){document.querySelectorAll('.tab-btn').forEach(t=>t.classList.remove('active'));document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));var tb=document.getElementById('tab-'+name);var tp=document.getElementById('panel-'+name);var at=document.getElementById('active_tab');if(tb)tb.classList.add('active');if(tp)tp.classList.add('active');if(at)at.value=name;}
        var _skewDefaults = {"L":60,"K":90,"H":90};
        function updateSliceLabel(val){
            var el=document.getElementById("slice-label"); if(el) el.innerText=val;
            var sk=document.getElementById("skew-angle-inp");
            if(sk && sk.value==sk.dataset.lastDefault) { sk.value=_skewDefaults[val]||90; sk.dataset.lastDefault=sk.value; }
        }
        document.addEventListener('DOMContentLoaded',function(){applyThemeFromStorage();var at=document.getElementById('active_tab');if(at)switchTab(at.value||'single');});
    </script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
</head>
<body>
<div class="app-shell">
    <aside class="sidebar">
        <div class="sidebar-brand">
            {% if chess_logo %}
            <img src="data:image/png;base64,{{ chess_logo }}"
                 style="width:36px;height:36px;border-radius:50%;object-fit:contain;flex-shrink:0;background:#fff;padding:2px;">
            {% else %}
            <div class="logo-circle">QM</div>
            {% endif %}
            <div><div class="logo-text-main">QM2 Data</div><div class="logo-text-sub">CHESS · Cornell</div></div>
        </div>
        <div class="nav-section-title">Analysis</div>
        <a class="nav-link {{ 'active' if active_page == 'browse' else '' }}" href="/browse"><span class="nav-icon">📂</span> Browse &amp; Analyze</a>
        <a class="nav-link {{ 'active' if active_page == 'slices' else '' }}" href="/slices"><span class="nav-icon">🔬</span> NxRefine Viewer</a>
        <a class="nav-link {{ 'active' if active_page == 'pyfai' else '' }}" href="/pyfai"><span class="nav-icon">⚡</span> pyFAI Integration</a>
        <a class="nav-link {{ 'active' if active_page == 'sample_search' else '' }}" href="/sample_search"><span class="nav-icon">⚗️</span> Sample Search</a>

        <div class="nav-section-title">Info</div>
        <a class="nav-link {{ 'active' if active_page == 'help' else '' }}" href="/help"><span class="nav-icon">📖</span> Help / Docs</a>
        <div class="nav-section-title">About</div>
        <div style="font-size:12px;line-height:1.5;color:rgba(255,255,255,0.45);padding:0 16px;">QM2 Staff Scientist<br>Quantum Materials Beamline</div>
        <div class="sidebar-footer">&copy; {{ year }} QM2 · CHESS</div>
    </aside>
    <div class="main-column">
        <header>
            <div><h1>Quantum Materials Beamline Data Analysis @ CHESS</h1><p>Powder data · NxRefine slices · pyFAI integration · Temperature overlays</p></div>
            <div style="display:flex;align-items:center;gap:10px;">
                <span class="header-badge">QM2 Beamline</span>
                <button class="theme-toggle" onclick="toggleTheme()">☀ / ☾</button>
            </div>
        </header>
        <main>{{ content | safe }}</main>
        <footer>Quantum Materials Beamline (QM2) · Cornell High Energy Synchrotron Source (CHESS)</footer>
    </div>
</div>
</body></html>
"""

def render_base(content, active_page=""):
    import datetime
    return render_template_string(BASE_HTML, content=content, active_page=active_page,
                                  year=datetime.datetime.now().year,
                                  chess_logo=_CHESS_LOGO_B64)

# ── PAGE TEMPLATES (unchanged) ────────────────────────────────────────────────
BROWSER_CONTENT = """
<div class="card"><h2 class="section-title">Folder Browser</h2><p>Current folder: <span class="path-label">{{ current }}</span></p><p class="subheading">Navigate your experiment tree, inspect NXS files, or launch analysis tools.</p></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Subfolders</span></div><ul>{% for d in dirs %}<li><a href="/browse?path={{ d }}">{{ d.split('/')[-1] }}</a></li>{% endfor %}{% if not dirs %}<li><em>No subfolders</em></li>{% endif %}</ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">.nxs Files</span></div><ul>{% for f in files %}<li>{{ f.split('/')[-1] }} — <a href="/analyze_file?file={{ f }}">Powder Data</a></li>{% endfor %}{% if not files %}<li><em>No .nxs files in this folder</em></li>{% endif %}</ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Folder Tools</span></div>
<form class="inline-form" action="/select_temps" method="get"><input type="hidden" name="path" value="{{ current }}"><button type="submit">Temperature‑Dependent Powder Data</button></form>
<form class="inline-form" action="/nxprocess" method="get"><input type="hidden" name="path" value="{{ current }}"><button type="submit">NXprocess Status</button></form>
<form class="inline-form" action="/choose_folder_A" method="get"><input type="hidden" name="path" value="{{ current }}"><button type="submit">Compare Two Subfolders</button></form></div>
"""

ANALYSIS_CONTENT = """
<div class="card"><h2 class="section-title">Powder Data: Average Radial Sum</h2><p>File: <span class="path-label">{{ file }}</span></p><p class="subheading">View the averaged radial sum (f1+f2+f3)/3 and export as CSV.</p></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Plot Controls</span></div>
<form action="/plot" method="post"><input type="hidden" name="file" value="{{ file }}">
<div class="inline-fields"><div class="field-group"><label class="field-label">X-min</label><input type="number" step="0.1" name="xmin" value="{{ xmin }}"></div><div class="field-group"><label class="field-label">X-max</label><input type="number" step="0.1" name="xmax" value="{{ xmax }}"></div></div><br><button type="submit">Generate Plot</button></form></div>
{% if traces_json %}<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Average Radial Sum</span><span class="slice-timing" style="margin-left:auto;">scroll to zoom &middot; drag to pan &middot; box-select to zoom in</span></div>
<div id="powder-single-chart" style="width:100%;height:420px;"></div>
<script>(function(){
  var traces=JSON.parse({{ traces_json | tojson }});
  var pTraces=traces.map(function(t){return{x:t.x,y:t.y,type:'scatter',mode:'lines',name:t.name,line:{color:t.color,width:2}};});
  var layout={paper_bgcolor:'white',plot_bgcolor:'white',
    xaxis:{title:'Q (Å⁻¹)',range:[{{ xmin }},{{ xmax }}],showgrid:true,gridcolor:'#ddd',zeroline:false},
    yaxis:{title:'Intensity',autorange:true,showgrid:true,gridcolor:'#ddd',zeroline:false},
    margin:{t:30,r:20,b:60,l:70},font:{family:'Inter,sans-serif',size:12,color:'#222'},
    hovermode:'x unified',showlegend:false};
  var cfg={responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
    toImageButtonOptions:{format:'png',filename:'avg_radial_sum',scale:2}};
  Plotly.newPlot('powder-single-chart',pTraces,layout,cfg);
})();</script>
<br><form action="/export_csv" method="post"><input type="hidden" name="file" value="{{ file }}"><button type="submit">Download Averaged Data (CSV)</button></form></div>{% endif %}
<div class="card"><a href="/browse?path={{ parent }}">← Back to folder</a></div>
"""

SELECT_TEMPS_CONTENT = """
<div class="card"><h2 class="section-title">Temperature‑Dependent Powder Data</h2><p>Folder: <span class="path-label">{{ path }}</span></p><p class="subheading">Select temperature‑encoded NXS files to overlay their averaged radial sums. Optionally define a Q region of interest to track integrated intensity vs temperature.</p></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Select Temperatures</span></div>
<form action="/plot_temp" method="post"><input type="hidden" name="path" value="{{ path }}">

<div style="display:flex;gap:6px;margin-bottom:8px;align-items:center;">
  <span style="font-size:11px;color:var(--text-muted);">Quick select:</span>
  <button type="button" onclick="document.querySelectorAll('.temp-chk').forEach(c=>c.checked=true)"
    style="font-size:11px;padding:3px 10px;background:rgba(148,163,184,0.12);color:var(--text-main);border:1px solid var(--border-subtle);border-radius:4px;cursor:pointer;">All</button>
  <button type="button" onclick="document.querySelectorAll('.temp-chk').forEach(c=>c.checked=false)"
    style="font-size:11px;padding:3px 10px;background:rgba(148,163,184,0.12);color:var(--text-main);border:1px solid var(--border-subtle);border-radius:4px;cursor:pointer;">None</button>
  <span style="font-size:11px;color:var(--text-muted);margin-left:4px;">{{ temp_files|length }} file(s) found</span>
</div>
<div style="max-height:220px;overflow-y:auto;padding:8px 10px;border:1px solid var(--border-subtle);border-radius:5px;margin-bottom:12px;">
{% for T, fpath in temp_files %}<label style="display:block;margin-bottom:3px;font-size:13px;cursor:pointer;"><input class="temp-chk" type="checkbox" name="temps" value="{{ fpath }}" {% if not selected_files or fpath in selected_files %}checked{% endif %}> {{ T }} K — {{ fpath.split('/')[-1] }}</label>{% endfor %}
{% if not temp_files %}<p style="color:var(--text-muted);font-size:13px;margin:0;"><em>No temperature‑encoded .nxs files found.</em></p>{% endif %}
</div>

<div class="inline-fields">
  <div class="field-group"><label class="field-label">X-min (Q, Å⁻¹)</label><input type="number" step="0.1" name="xmin" value="{{ xmin }}"></div>
  <div class="field-group"><label class="field-label">X-max (Q, Å⁻¹)</label><input type="number" step="0.1" name="xmax" value="{{ xmax }}"></div>
</div>

<div style="margin-top:14px;padding:12px 16px;background:rgba(249,115,22,0.05);border:1px solid rgba(249,115,22,0.25);border-radius:8px;">
  <label style="display:flex;align-items:center;gap:7px;font-size:13px;cursor:pointer;margin-bottom:8px;">
    <input type="checkbox" name="roi_enabled" id="roi-enabled-chk" value="1"
           {% if roi_enabled %}checked{% endif %}
           onchange="document.getElementById('roi-grp').style.display=this.checked?'flex':'none'">
    <b style="color:#f97316;">&#9641; Region of Interest (Q integration)</b>
    <span style="font-size:11px;color:var(--text-muted);font-weight:400;">— shade Q band on chart &amp; compute ∫I dQ vs T</span>
  </label>
  <div id="roi-grp" style="display:{% if roi_enabled %}flex{% else %}none{% endif %};gap:16px;flex-wrap:wrap;align-items:flex-end;">
    <div class="field-group"><label class="field-label" style="color:#f97316;">ROI Q-min (Å⁻¹)</label>
      <input type="number" name="roi_qmin" value="{{ roi_qmin }}" step="0.01" style="width:110px;border-color:rgba(249,115,22,0.45);"></div>
    <div class="field-group"><label class="field-label" style="color:#f97316;">ROI Q-max (Å⁻¹)</label>
      <input type="number" name="roi_qmax" value="{{ roi_qmax }}" step="0.01" style="width:110px;border-color:rgba(249,115,22,0.45);"></div>
  </div>
</div>

<br><button type="submit">Generate Plot</button></form></div>

{% if traces_json %}
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Temperature-Dependent Overlay</span><span class="slice-timing" style="margin-left:auto;">scroll to zoom &middot; drag to pan &middot; box-select to zoom in</span></div>
<div id="powder-temp-chart" style="width:100%;height:480px;"></div>
<script>(function(){
  var traces=JSON.parse({{ traces_json | tojson }});
  var pTraces=traces.map(function(t){return{x:t.x,y:t.y,type:'scatter',mode:'lines',name:t.name,line:{color:t.color,width:1.8}};});
  var layout={paper_bgcolor:'white',plot_bgcolor:'white',
    xaxis:{title:'Q (Å⁻¹)',range:[{{ xmin }},{{ xmax }}],showgrid:true,gridcolor:'#ddd',zeroline:false},
    yaxis:{title:'Intensity',autorange:true,showgrid:true,gridcolor:'#ddd',zeroline:false},
    legend:{orientation:'v',x:1.02,y:1,xanchor:'left'},
    margin:{t:30,r:160,b:60,l:70},font:{family:'Inter,sans-serif',size:12,color:'#222'},
    hovermode:'x unified'};
  {% if roi_enabled %}
  layout.shapes=[{type:'rect',xref:'x',yref:'paper',x0:{{ roi_qmin }},x1:{{ roi_qmax }},y0:0,y1:1,
    fillcolor:'rgba(249,115,22,0.10)',line:{color:'rgba(249,115,22,0.55)',width:1.5,dash:'dot'}}];
  layout.annotations=[{x:({{ roi_qmin }}+{{ roi_qmax }})/2,xref:'x',yref:'paper',y:1.01,
    text:'ROI',showarrow:false,font:{color:'#f97316',size:11,weight:'bold'}}];
  {% endif %}
  var cfg={responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
    toImageButtonOptions:{format:'png',filename:'temp_overlay',scale:2}};
  Plotly.newPlot('powder-temp-chart',pTraces,layout,cfg);
})();</script>
<br><form action="/export_temp_csv" method="post"><input type="hidden" name="files" value="{{ selected_files_serialized }}"><button type="submit">&#11015; Download Overlay CSV</button></form></div>
{% endif %}

{% if roi_enabled and roi_data %}
<div class="card" style="border-color:rgba(249,115,22,0.4);">
  <div class="card-header">
    <div class="card-dot" style="background:#f97316;box-shadow:0 0 8px #f97316;"></div>
    <span class="card-title" style="color:#f97316;">ROI Integrated Intensity vs Temperature</span>
    <span class="slice-timing" style="margin-left:auto;">Q = [{{ roi_qmin }} – {{ roi_qmax }}] Å⁻¹ &nbsp;·&nbsp; trapezoid rule</span>
  </div>
  <div id="roi-int-chart" style="width:100%;height:320px;"></div>
  <script>(function(){
    var temps={{ roi_temps_json | safe }};
    var intens={{ roi_intens_json | safe }};
    var colors={{ roi_colors_json | safe }};
    var traces=[{x:temps,y:intens,type:'scatter',mode:'markers+lines',
      marker:{color:colors,size:10,line:{color:'#333',width:1}},
      line:{color:'#f97316',width:2},
      hovertemplate:'<b>%{x} K</b><br>\u222bI dQ = %{y:.4g}<extra></extra>'}];
    var layout={paper_bgcolor:'white',plot_bgcolor:'white',
      xaxis:{title:'Temperature (K)',showgrid:true,gridcolor:'#ddd',zeroline:false},
      yaxis:{title:'\u222b I(Q) dQ  [ROI]',autorange:true,showgrid:true,gridcolor:'#ddd',zeroline:false},
      margin:{t:20,r:30,b:60,l:80},font:{family:'Inter,sans-serif',size:12,color:'#222'},
      hovermode:'closest'};
    var cfg={responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
      toImageButtonOptions:{format:'png',filename:'roi_integral_vs_T',scale:2}};
    Plotly.newPlot('roi-int-chart',traces,layout,cfg);
  })();</script>
  <hr style="margin:14px 0 10px;">
  <table style="font-size:12px;max-width:360px;margin-bottom:12px;">
    <tr><th>Temperature (K)</th><th>∫ I(Q) dQ</th></tr>
    {% for d in roi_data %}<tr><td>{{ d.T }} K</td><td>{{ "%.5g"|format(d.integral) }}</td></tr>{% endfor %}
  </table>
  <form action="/export_roi_csv" method="post">
    <input type="hidden" name="roi_csv_data" value="{{ roi_csv_serialized }}">
    <input type="hidden" name="roi_qmin" value="{{ roi_qmin }}">
    <input type="hidden" name="roi_qmax" value="{{ roi_qmax }}">
    <button type="submit">&#11015; Download ROI Integration CSV</button>
  </form>
</div>
{% endif %}

<div class="card"><a href="/browse?path={{ path }}">← Back to folder</a></div>
"""

NXPROCESS_CONTENT = """
<div class="card"><h2 class="section-title">NXS Processing Status</h2><p>Folder: <span class="path-label">{{ path }}</span></p></div>
<div class="card"><table><tr><th>File</th><th>Entry</th><th>NXprocess Steps</th></tr>
{% for row in rows %}<tr><td>{{ row.file }}</td><td>{{ row.entry }}</td><td>{% if row.processes %}<span class="ok">{{ row.processes|join(', ') }}</span>{% else %}<span class="missing">None</span>{% endif %}</td></tr>{% endfor %}
{% if not rows %}<tr><td colspan="3"><em>No .nxs files found.</em></td></tr>{% endif %}</table></div>
<div class="card"><a href="/browse?path={{ path }}">← Back to folder</a></div>
"""

CHOOSE_FOLDER_A_CONTENT = """
<div class="card"><h2 class="section-title">Compare Two Subfolders — Step 1</h2><p>Select <b>Folder A</b> inside: <span class="path-label">{{ path }}</span></p></div>
<div class="card"><form action="/choose_folder_B" method="post"><input type="hidden" name="path" value="{{ path }}">{% for d in subdirs %}<label><input type="radio" name="folderA" value="{{ d }}" required> {{ d.split('/')[-1] }}</label><br>{% endfor %}{% if not subdirs %}<p><em>No subfolders available.</em></p>{% endif %}<br><button type="submit">Next: Choose Folder B</button></form></div>
<div class="card"><a href="/browse?path={{ path }}">← Back to folder</a></div>
"""

CHOOSE_FOLDER_B_CONTENT = """
<div class="card"><h2 class="section-title">Compare Two Subfolders — Step 2</h2><p>Folder A: <span class="path-label">{{ folderA.split('/')[-1] }}</span></p><p>Select <b>Folder B</b> inside: <span class="path-label">{{ path }}</span></p></div>
<div class="card"><form action="/choose_temps_compare" method="post"><input type="hidden" name="folderA" value="{{ folderA }}"><input type="hidden" name="path" value="{{ path }}">{% for d in subdirs %}<label><input type="radio" name="folderB" value="{{ d }}" required> {{ d.split('/')[-1] }}</label><br>{% endfor %}{% if not subdirs %}<p><em>No subfolders available.</em></p>{% endif %}<br><button type="submit">Next: Choose Temperatures</button></form></div>
<div class="card"><a href="/browse?path={{ path }}">← Back to folder</a></div>
"""

CHOOSE_TEMPS_COMPARE_CONTENT = """
<div class="card"><h2 class="section-title">Compare Two Subfolders — Step 3</h2><p>Folder A: <span class="path-label">{{ folderA.split('/')[-1] }}</span> &nbsp;|&nbsp; Folder B: <span class="path-label">{{ folderB.split('/')[-1] }}</span></p></div>
<div class="card"><form action="/plot_temp_compare" method="post">
<input type="hidden" name="folderA" value="{{ folderA }}"><input type="hidden" name="folderB" value="{{ folderB }}"><input type="hidden" name="parent" value="{{ parent }}">
<h4 style="margin:0 0 8px;">{{ folderA.split('/')[-1] }}</h4>
{% for T, fpath in tempsA %}<label><input type="checkbox" name="tempsA" value="{{ fpath }}" {% if not selectedA or fpath in selectedA %}checked{% endif %}> {{ T }} K — {{ fpath.split('/')[-1] }}</label><br>{% endfor %}
{% if not tempsA %}<p><em>No temperature‑encoded files in Folder A.</em></p>{% endif %}
<h4 style="margin:16px 0 8px;">{{ folderB.split('/')[-1] }}</h4>
{% for T, fpath in tempsB %}<label><input type="checkbox" name="tempsB" value="{{ fpath }}" {% if not selectedB or fpath in selectedB %}checked{% endif %}> {{ T }} K — {{ fpath.split('/')[-1] }}</label><br>{% endfor %}
{% if not tempsB %}<p><em>No temperature‑encoded files in Folder B.</em></p>{% endif %}<br>
<div class="inline-fields"><div class="field-group"><label class="field-label">X-min</label><input type="number" step="0.1" name="xmin" value="{{ xmin }}"></div><div class="field-group"><label class="field-label">X-max</label><input type="number" step="0.1" name="xmax" value="{{ xmax }}"></div></div><br>
<button type="submit">Generate Comparison Plot</button></form></div>
{% if traces_json %}<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">Comparison Plot</span><span class="slice-timing" style="margin-left:auto;">scroll to zoom &middot; drag to pan &middot; box-select to zoom in</span></div>
<div id="powder-compare-chart" style="width:100%;height:500px;"></div>
<script>(function(){
  var traces=JSON.parse({{ traces_json | tojson }});
  var pTraces=traces.map(function(t){return{x:t.x,y:t.y,type:'scatter',mode:'lines',name:t.name,line:{color:t.color,width:1.8,dash:t.dash||'solid'}};});
  var layout={paper_bgcolor:'white',plot_bgcolor:'white',
    xaxis:{title:'Q (Å⁻¹)',range:[{{ xmin }},{{ xmax }}],showgrid:true,gridcolor:'#ddd',zeroline:false},
    yaxis:{title:'Intensity',autorange:true,showgrid:true,gridcolor:'#ddd',zeroline:false},
    legend:{orientation:'v',x:1.02,y:1,xanchor:'left'},
    margin:{t:30,r:200,b:60,l:70},font:{family:'Inter,sans-serif',size:12,color:'#222'},
    hovermode:'x unified'};
  var cfg={responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
    toImageButtonOptions:{format:'png',filename:'folder_comparison',scale:2}};
  Plotly.newPlot('powder-compare-chart',pTraces,layout,cfg);
})();</script>
<br><form action="/export_compare_csv" method="post"><input type="hidden" name="folderA" value="{{ folderA }}"><input type="hidden" name="folderB" value="{{ folderB }}"><input type="hidden" name="selA" value="{{ selectedA_serialized }}"><input type="hidden" name="selB" value="{{ selectedB_serialized }}"><button type="submit">Download Comparison Data (CSV)</button></form></div>{% endif %}
<div class="card"><a href="/browse?path={{ parent }}">← Back to folder</a></div>
"""

# ── pyFAI TEMPLATES ───────────────────────────────────────────────────────────
PYFAI_CONTENT = """
<div class="card"><h2 class="section-title">pyFAI Integration <span class="badge badge-blue">Pilatus 6M</span></h2><p class="subheading">1D integration (Q or 2&theta;), caking (2D), batch processing, and height-scan preview of .cbf detector images using a PONI calibration file.</p></div>
<div class="card"><form action="/pyfai/run" method="post" id="pyfai-form">
<div class="field-group"><label class="field-label">Processing Mode</label>
<label><input type="radio" name="mode" value="single" checked onclick="toggleMode(this.value)"> Single image</label>&nbsp;&nbsp;
<label><input type="radio" name="mode" value="batch" onclick="toggleMode(this.value)"> Batch folder</label>&nbsp;&nbsp;
<label><input type="radio" name="mode" value="height_batch" onclick="toggleMode(this.value)"> &#128200; Height Batch <span style="font-size:11px;color:var(--text-muted);">(preview only)</span></label></div>
<div id="single-path-row" class="field-group"><label class="field-label">Image File (.cbf / .tif / .edf)</label><div style="display:flex;gap:8px;align-items:center;"><input type="text" name="img_path" id="img_path" value="{{ img_path or '' }}" placeholder="/nfs/chess/.../image.cbf"><a href="/pyfai/browse?field=img_path&pick=file&ext=.cbf,.tif,.tiff,.edf&img_path={{ img_path|urlencode }}&folder_path={{ folder_path|urlencode }}&poni_path={{ poni_path|urlencode }}&mask_path={{ mask_path|urlencode }}&output_path={{ output_path|urlencode }}&mode={{ mode }}"><button type="button">&#128194; Browse</button></a></div></div>
<div id="batch-path-row" class="field-group" style="display:none;"><label class="field-label">Image Folder (.cbf files)</label><div style="display:flex;gap:8px;align-items:center;"><input type="text" name="folder_path" id="folder_path" value="{{ folder_path or '' }}" placeholder="/nfs/chess/.../images/"><a href="/pyfai/browse?field=folder_path&pick=folder&img_path={{ img_path|urlencode }}&folder_path={{ folder_path|urlencode }}&poni_path={{ poni_path|urlencode }}&mask_path={{ mask_path|urlencode }}&output_path={{ output_path|urlencode }}&mode={{ mode }}"><button type="button">&#128194; Browse</button></a></div></div>
<div class="field-group"><label class="field-label">PONI Calibration File</label><div style="display:flex;gap:8px;align-items:center;"><input type="text" name="poni_path" id="poni_path" value="{{ poni_path or '' }}" placeholder="/nfs/chess/.../detector.poni"><a href="/pyfai/browse?field=poni_path&pick=file&ext=.poni&img_path={{ img_path|urlencode }}&folder_path={{ folder_path|urlencode }}&poni_path={{ poni_path|urlencode }}&mask_path={{ mask_path|urlencode }}&output_path={{ output_path|urlencode }}&mode={{ mode }}"><button type="button">&#128194; Browse</button></a></div></div>
<div class="field-group"><label class="field-label">Mask File (optional)</label><div style="display:flex;gap:8px;align-items:center;"><input type="text" name="mask_path" id="mask_path" value="{{ mask_path or '' }}" placeholder="Leave blank for no mask"><a href="/pyfai/browse?field=mask_path&pick=file&ext=.tif,.tiff,.cbf,.edf,.npy&img_path={{ img_path|urlencode }}&folder_path={{ folder_path|urlencode }}&poni_path={{ poni_path|urlencode }}&mask_path={{ mask_path|urlencode }}&output_path={{ output_path|urlencode }}&mode={{ mode }}"><button type="button">&#128194; Browse</button></a></div></div>
<div id="output-row" class="field-group"><label class="field-label">Output Folder</label><div style="display:flex;gap:8px;align-items:center;"><input type="text" name="output_path" id="output_path" value="{{ output_path or '' }}" placeholder="/nfs/chess/.../output/"><a href="/pyfai/browse?field=output_path&pick=folder&img_path={{ img_path|urlencode }}&folder_path={{ folder_path|urlencode }}&poni_path={{ poni_path|urlencode }}&mask_path={{ mask_path|urlencode }}&output_path={{ output_path|urlencode }}&mode={{ mode }}"><button type="button">&#128194; Browse</button></a></div></div>
<div id="hb-note" class="alert-info" style="display:none;">&#128200; <b>Height Batch mode</b> &mdash; raw detector image + 1D Q integration shown one-by-one. No CSV or output files are saved.</div>
<hr>
<div id="integration-opts">
<div class="field-group"><label class="field-label">Integration Options</label>
<label><input type="checkbox" name="do_1d" checked> 1D integration (Q)</label><br>
<label><input type="checkbox" name="do_tth" checked> Convert Q &rarr; 2&theta;</label><br>
<label><input type="checkbox" name="do_cake"> Caking (2D integration)</label></div>
<div class="inline-fields">
<div class="field-group"><label class="field-label">1D bins</label><input type="number" name="thbin" value="10000" min="100" max="50000"></div>
<div class="field-group"><label class="field-label">Cake azimuthal bins</label><input type="number" name="nazim" value="360" min="10" max="720"></div></div>
</div>
<div id="hb-bins" style="display:none;">
<div class="inline-fields"><div class="field-group"><label class="field-label">1D bins <span style="font-weight:400;text-transform:none;font-size:11px;color:var(--text-muted);">(2000 = fast preview)</span></label><input type="number" name="thbin_hb" value="2000" min="100" max="50000"></div></div>
</div>
<br><button type="submit" class="btn-primary">Run pyFAI Integration</button></form></div>
<script>(function(){var mode="{{ mode or 'single' }}";document.querySelectorAll('input[name="mode"]').forEach(function(r){if(r.value===mode)r.checked=true;});toggleMode(mode);})();
function toggleMode(val){
  var isHB=(val==='height_batch');
  document.getElementById('single-path-row').style.display=(val==='single')?'':'none';
  document.getElementById('batch-path-row').style.display=(val==='batch'||isHB)?'':'none';
  document.getElementById('output-row').style.display=isHB?'none':'';
  document.getElementById('integration-opts').style.display=isHB?'none':'';
  document.getElementById('hb-bins').style.display=isHB?'':'none';
  document.getElementById('hb-note').style.display=isHB?'':'none';
}</script>
"""

PYFAI_BROWSE_CONTENT = """
<div class="card"><h2 class="section-title">{% if pick == 'file' %}Pick a File{% else %}Pick a Folder{% endif %} <span class="badge badge-blue">{{ field }}</span></h2><p>Current path: <span class="path-label">{{ current }}</span></p></div>
<div class="card" style="padding:10px 16px;font-size:13px;">{% for crumb_label, crumb_path in breadcrumbs %}<a href="/pyfai/browse?{{ browse_qs(crumb_path) }}">{{ crumb_label }}</a>{% if not loop.last %} / {% endif %}{% endfor %}</div>
{% if pick == 'folder' %}<div class="card"><form action="/pyfai/pick" method="post"><input type="hidden" name="field" value="{{ field }}"><input type="hidden" name="value" value="{{ current }}"><input type="hidden" name="img_path" value="{{ state.img_path }}"><input type="hidden" name="folder_path" value="{{ state.folder_path }}"><input type="hidden" name="poni_path" value="{{ state.poni_path }}"><input type="hidden" name="mask_path" value="{{ state.mask_path }}"><input type="hidden" name="output_path" value="{{ state.output_path }}"><input type="hidden" name="mode" value="{{ state.mode }}"><button type="submit">✅ Select: <span class="path-label">{{ current }}</span></button></form></div>{% endif %}
{% if subdirs %}<div class="card"><h3 class="subheading">Subfolders</h3><ul>{% for d in subdirs %}<li><a href="/pyfai/browse?{{ browse_qs(d) }}">📁 {{ d.split('/')[-1] }}</a></li>{% endfor %}</ul></div>{% endif %}
{% if pick == 'file' and files %}<div class="card"><h3 class="subheading">Files{% if ext %} ({{ ext }}){% endif %}</h3><ul>{% for f in files %}<li><form action="/pyfai/pick" method="post" style="display:inline;"><input type="hidden" name="field" value="{{ field }}"><input type="hidden" name="value" value="{{ f }}"><input type="hidden" name="img_path" value="{{ state.img_path }}"><input type="hidden" name="folder_path" value="{{ state.folder_path }}"><input type="hidden" name="poni_path" value="{{ state.poni_path }}"><input type="hidden" name="mask_path" value="{{ state.mask_path }}"><input type="hidden" name="output_path" value="{{ state.output_path }}"><input type="hidden" name="mode" value="{{ state.mode }}"><button type="submit" style="background:none;color:var(--accent);border:none;padding:0;cursor:pointer;font-size:13px;text-decoration:underline;">{{ f.split('/')[-1] }}</button></form></li>{% endfor %}</ul></div>{% endif %}
{% if pick == 'file' and not files and not subdirs %}<div class="card"><em>No matching files or subfolders found.</em></div>{% endif %}
<div class="card"><a href="{{ back_url }}">← Cancel, back to pyFAI setup</a></div>
"""

PYFAI_RESULT_CONTENT = """
<script>
var _pfd=[{% for r in results %}{f:{{ r.filename|tojson }},ok:{{ 'true' if r.ok else 'false' }},q:{{ r.q_json|safe if r.q_json else 'null' }},I:{{ r.I_json|safe if r.I_json else 'null' }},tth:{{ r.tth_json|safe if r.tth_json else 'null' }},Itth:{{ r.I_tth_json|safe if r.I_tth_json else 'null' }},cake:{{ r.plot_cake|tojson }},qpts:{{ r.q_points|tojson }},cshape:{{ r.cake_shape|tojson }},err:{{ r.error|tojson }}}{% if not loop.last %},{% endif %}{% endfor %}];
var _pi=0,_pN=_pfd.length;
function _pShow(i){
  if(i<0||i>=_pN)return; _pi=i; var d=_pfd[i];
  document.getElementById('pfi-fname').textContent=d.f;
  document.getElementById('pfi-ctr').textContent=(i+1)+' / '+_pN;
  document.getElementById('pfi-prev').disabled=(i===0);
  document.getElementById('pfi-next').disabled=(i===_pN-1);
  document.getElementById('pfi-status').innerHTML=d.ok?'<span class="badge badge-green">OK</span>':'<span class="badge badge-red">Error</span>';
  document.getElementById('pfi-meta').textContent='Q points: '+d.qpts+' | Cake: '+d.cshape;
  var eEl=document.getElementById('pfi-err'); eEl.style.display=d.err?'':'none'; eEl.textContent=d.err||'';
  document.getElementById('pfi-plots').style.display=d.ok?'':'none';
  var qD=document.getElementById('pfi-q'); Plotly.purge(qD);
  var tD=document.getElementById('pfi-tth'); Plotly.purge(tD);
  if(d.q&&d.I){
    qD.style.display='';
    Plotly.newPlot(qD,[{x:d.q,y:d.I,type:'scatter',mode:'lines',name:'Q',line:{color:'#1e88e5',width:1.5}}],
      {paper_bgcolor:'white',plot_bgcolor:'white',xaxis:{title:'Q (\u00c5\u207b\u00b9)',autorange:true,showgrid:true,gridcolor:'#e0e0e0',zeroline:false},
       yaxis:{title:'Intensity',autorange:true,showgrid:true,gridcolor:'#e0e0e0',zeroline:false},
       margin:{t:30,b:50,l:60,r:20},hovermode:'x unified'},
      {responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,toImageButtonOptions:{format:'png',filename:'pyfai_1d',scale:2}});
  } else { qD.style.display='none'; }
  if(d.tth&&d.Itth){
    tD.style.display='';
    Plotly.newPlot(tD,[{x:d.tth,y:d.Itth,type:'scatter',mode:'lines',name:'2\u03b8',line:{color:'#e53935',width:1.5}}],
      {paper_bgcolor:'white',plot_bgcolor:'white',xaxis:{title:'2\u03b8 (deg)',autorange:true,showgrid:true,gridcolor:'#e0e0e0',zeroline:false},
       yaxis:{title:'Intensity',autorange:true,showgrid:true,gridcolor:'#e0e0e0',zeroline:false},
       margin:{t:30,b:50,l:60,r:20},hovermode:'x unified'},
      {responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,toImageButtonOptions:{format:'png',filename:'pyfai_tth',scale:2}});
  } else { tD.style.display='none'; }
  var cImg=document.getElementById('pfi-cake');
  if(d.cake){cImg.src='data:image/png;base64,'+d.cake; cImg.style.display='';}
  else{cImg.style.display='none';}
  var cWrap=document.getElementById('pfi-cake-wrap'); cWrap.style.display=d.cake?'':'none';
}
</script>
<div class="card">
  <h2 class="section-title">pyFAI Integration Results {% if errors %}<span class="badge badge-red">{{ errors|length }} error(s)</span>{% else %}<span class="badge badge-green">Success</span>{% endif %}</h2>
  <p class="subheading">Processed {{ results|length }} image(s). Output: <span class="path-label">{{ output_path }}</span></p>
</div>
{% if errors %}<div class="card"><h3 class="subheading" style="color:#f97373;">Errors</h3><ul>{% for e in errors %}<li class="missing">{{ e }}</li>{% endfor %}</ul></div>{% endif %}
{% if results %}
<div class="card">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--border-subtle);">
    <button id="pfi-prev" onclick="_pShow(_pi-1)" disabled>&#8592; Prev</button>
    <span id="pfi-ctr" style="font-size:13px;font-weight:600;min-width:64px;text-align:center;">1 / {{ results|length }}</span>
    <button id="pfi-next" onclick="_pShow(_pi+1)" {% if results|length <= 1 %}disabled{% endif %}>Next &#8594;</button>
    <span id="pfi-fname" style="font-size:13px;font-weight:600;color:var(--accent);margin-left:8px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
    <span id="pfi-status" style="flex-shrink:0;"></span>
  </div>
  <p id="pfi-err" class="missing" style="display:none;margin:0 0 8px;"></p>
  <div id="pfi-plots">
    <div class="two-col" style="margin-bottom:10px;">
      <div><p style="font-size:12px;margin:0 0 4px;"><b>1D Integration (Q)</b></p><div id="pfi-q" style="width:100%;height:340px;background:white;"></div></div>
      <div><p style="font-size:12px;margin:0 0 4px;"><b>Q &#8594; 2&#952;</b></p><div id="pfi-tth" style="width:100%;height:340px;background:white;"></div></div>
    </div>
    <div id="pfi-cake-wrap" style="display:none;margin-top:8px;">
      <p style="font-size:12px;margin:4px 0;"><b>Caked Image</b></p>
      <img id="pfi-cake" style="max-width:100%;">
    </div>
    <p id="pfi-meta" style="font-size:12px;margin-top:10px;color:var(--text-muted);"></p>
  </div>
</div>
<script>_pShow(0);</script>
{% endif %}
{% if log_rows %}<div class="card"><h3 class="subheading">Batch Log</h3><table class="log-table"><tr><th>File</th><th>Q points</th><th>Cake shape</th><th>Status</th></tr>{% for row in log_rows %}<tr><td>{{ row.filename }}</td><td>{{ row.q_points }}</td><td>{{ row.cake_shape }}</td><td>{% if row.ok %}<span class="ok">OK</span>{% else %}<span class="missing">{{ row.error }}</span>{% endif %}</td></tr>{% endfor %}</table></div>{% endif %}
<div class="card"><a href="/pyfai">&#8592; Back to pyFAI Integration</a></div>
"""

PYFAI_HEIGHT_VIEWER_CONTENT = """
<script>
var _hbFiles={{ img_paths_json | safe }};
var _hbPoni={{ poni_path | tojson }};
var _hbMask={{ mask_path | tojson }};
var _hbThbin={{ thbin }};
var _hbCache={};     // integration results: {q, I, q_points, filename, ok}
var _hbImgCache={};  // raw images (fetched on demand): raw_b64 strings
var _hi=0, _hN=_hbFiles.length;

// ── Integration fetch (default, fast — no raw image) ─────────────────────────
function _hFetch(i, cb){
  if(_hbCache[i]!==undefined){cb(_hbCache[i]);return;}
  var fd=new FormData();
  fd.append('img_path',_hbFiles[i]);
  fd.append('poni_path',_hbPoni);
  fd.append('mask_path',_hbMask||'');
  fd.append('thbin',_hbThbin);
  fd.append('show_raw','0');
  fetch('/pyfai/height_one_ajax',{method:'POST',body:fd})
    .then(function(r){return r.json();})
    .then(function(d){_hbCache[i]=d; cb(d);})
    .catch(function(e){cb({ok:false,error:String(e),filename:_hbFiles[i].split('/').pop()});});
}

// ── On-demand raw image fetch ─────────────────────────────────────────────────
function _hFetchImg(i){
  var btn=document.getElementById('hb-img-btn');
  var rawCol=document.getElementById('hb-raw-col');
  var layout=document.getElementById('hb-layout');
  // Already cached — just reveal
  if(_hbImgCache[i]!==undefined){
    document.getElementById('hb-raw-img').src='data:image/png;base64,'+_hbImgCache[i];
    rawCol.style.display='';
    layout.style.gridTemplateColumns='1fr 1fr';
    btn.textContent='🖼 Hide Image';
    btn.onclick=function(){_hHideImg();};
    return;
  }
  btn.disabled=true; btn.textContent='Loading image\u2026';
  var _imgT0=Date.now();
  var fd=new FormData();
  fd.append('img_path',_hbFiles[i]);
  fd.append('poni_path',_hbPoni);
  fd.append('mask_path',_hbMask||'');
  fd.append('thbin',_hbThbin);
  fd.append('show_raw','1');
  fetch('/pyfai/height_one_ajax',{method:'POST',body:fd})
    .then(function(r){return r.json();})
    .then(function(d){
      btn.disabled=false;
      if(d.ok&&d.raw_b64){
        _hbImgCache[i]=d.raw_b64;
        if(i===_hi){
          document.getElementById('hb-raw-img').src='data:image/png;base64,'+d.raw_b64;
          rawCol.style.display='';
          layout.style.gridTemplateColumns='1fr 1fr';
          btn.textContent='🖼 Hide Image';
          btn.onclick=function(){_hHideImg();};
          // Append image timing to meta bar
          var metaEl=document.getElementById('hb-meta');
          var cur=metaEl.textContent;
          var imgStats='';
          if(d.t_render_s!=null) imgStats+='  \u00b7  Render: '+d.t_render_s+' s';
          if(d.t_total_s!=null)  imgStats+='  \u00b7  Img total: '+d.t_total_s+' s';
          metaEl.textContent=cur+imgStats;
        }
      } else {
        btn.textContent='🖼 Show Image';
        btn.onclick=function(){_hFetchImg(_hi);};
      }
    })
    .catch(function(){
      btn.disabled=false;
      btn.textContent='🖼 Show Image';
      btn.onclick=function(){_hFetchImg(_hi);};
    });
}

function _hHideImg(){
  document.getElementById('hb-raw-col').style.display='none';
  document.getElementById('hb-layout').style.gridTemplateColumns='1fr';
  var btn=document.getElementById('hb-img-btn');
  btn.textContent='🖼 Show Image';
  btn.onclick=function(){_hFetchImg(_hi);};
}

// ── Live elapsed timer while waiting ─────────────────────────────────────────
var _hbTimer=null;
function _hbStartTimer(){
  var t0=Date.now();
  var el=document.getElementById('hb-elapsed');
  if(el) el.textContent='0.0 s';
  _hbStopTimer();
  _hbTimer=setInterval(function(){
    var s=((Date.now()-t0)/1000).toFixed(1);
    if(el) el.textContent=s+' s';
  },100);
}
function _hbStopTimer(){ if(_hbTimer){clearInterval(_hbTimer);_hbTimer=null;} }

// ── Build stat bar string ─────────────────────────────────────────────────────
function _hbMetaStr(d){
  var parts=[];
  if(d.img_shape)       parts.push('Image: '+d.img_shape+' px');
  if(d.q_points!=='—')  parts.push('Q pts: '+d.q_points);
  if(d.t_load_s!=null)  parts.push('Load: '+d.t_load_s+' s');
  if(d.t_integrate_s!=null) parts.push('Integrate: '+d.t_integrate_s+' s');
  if(d.t_render_s!=null)    parts.push('Render: '+d.t_render_s+' s');
  if(d.t_total_s!=null) parts.push('Total: '+d.t_total_s+' s');
  return parts.join('  \u00b7  ');
}

// ── Navigate to image i ───────────────────────────────────────────────────────
function _hShow(i){
  if(i<0||i>=_hN)return; _hi=i;
  var fname=_hbFiles[i].split('/').pop();
  document.getElementById('hb-ctr').textContent=(i+1)+' / '+_hN;
  document.getElementById('hb-prev').disabled=(i===0);
  document.getElementById('hb-next').disabled=(i===_hN-1);
  document.getElementById('hb-fname').textContent=fname;
  document.getElementById('hb-status').innerHTML='';
  document.getElementById('hb-loading').style.display='';
  document.getElementById('hb-plots').style.display='none';
  document.getElementById('hb-err').style.display='none';
  document.getElementById('hb-meta').textContent='';
  // Reset image panel for this index
  var rawCol=document.getElementById('hb-raw-col');
  var layout=document.getElementById('hb-layout');
  var btn=document.getElementById('hb-img-btn');
  rawCol.style.display='none';
  layout.style.gridTemplateColumns='1fr';
  btn.disabled=false;
  btn.textContent='🖼 Show Image';
  btn.onclick=function(){_hFetchImg(_hi);};
  // Start live timer (only if not already cached)
  if(_hbCache[i]===undefined) _hbStartTimer();
  _hFetch(i,function(d){
    _hbStopTimer();
    document.getElementById('hb-loading').style.display='none';
    document.getElementById('hb-status').innerHTML=d.ok
      ?'<span class="badge badge-green">OK</span>'
      :'<span class="badge badge-red">Error</span>';
    if(!d.ok){
      var eEl=document.getElementById('hb-err');
      eEl.style.display=''; eEl.textContent=d.error||'Unknown error'; return;
    }
    document.getElementById('hb-plots').style.display='';
    document.getElementById('hb-meta').textContent=_hbMetaStr(d);
    // If image already cached for this index, auto-show it
    if(_hbImgCache[i]!==undefined){
      document.getElementById('hb-raw-img').src='data:image/png;base64,'+_hbImgCache[i];
      rawCol.style.display='';
      layout.style.gridTemplateColumns='1fr 1fr';
      btn.textContent='🖼 Hide Image';
      btn.onclick=function(){_hHideImg();};
    }
    var qD=document.getElementById('hb-q'); Plotly.purge(qD);
    if(d.q&&d.q.length){
      Plotly.newPlot(qD,[{x:d.q,y:d.I,type:'scatter',mode:'lines',name:'Q',
        line:{color:'#1e88e5',width:1.5}}],
        {paper_bgcolor:'white',plot_bgcolor:'white',
         xaxis:{title:'Q (\u00c5\u207b\u00b9)',autorange:true,showgrid:true,gridcolor:'#e0e0e0',zeroline:false},
         yaxis:{title:'Intensity',autorange:true,showgrid:true,gridcolor:'#e0e0e0',zeroline:false},
         margin:{t:30,b:50,l:60,r:20},hovermode:'x unified'},
        {responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:true,
         toImageButtonOptions:{format:'png',filename:'hb_1d',scale:2}});
    }
    // Preload next image integration only (no raw — keeps it fast)
    if(i+1<_hN) _hFetch(i+1,function(){});
  });
}
window.addEventListener('DOMContentLoaded',function(){if(_hN>0)_hShow(0);});
</script>
<div class="card">
  <h2 class="section-title">Height Batch Preview <span class="badge badge-blue">{{ n_files }} image(s)</span></h2>
  <p class="subheading">1D Q integration loaded on demand &mdash; no files saved. Raw detector image available on request.</p>
</div>
<div class="card">
  <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--border-subtle);">
    <button id="hb-prev" onclick="_hShow(_hi-1)" disabled>&#8592; Prev</button>
    <span id="hb-ctr" style="font-size:13px;font-weight:600;min-width:64px;text-align:center;">— / {{ n_files }}</span>
    <button id="hb-next" onclick="_hShow(_hi+1)" {% if n_files <= 1 %}disabled{% endif %}>Next &#8594;</button>
    <span id="hb-fname" style="font-size:13px;font-weight:600;color:var(--accent);margin-left:8px;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"></span>
    <span id="hb-status" style="flex-shrink:0;"></span>
  </div>
  <div id="hb-loading" style="text-align:center;padding:40px 0;display:none;">
    <div class="progress-bar-wrap" style="max-width:300px;margin:0 auto 12px;"><div class="progress-bar"></div></div>
    <p style="font-size:13px;color:var(--text-muted);">Integrating&hellip; <span id="hb-elapsed" style="font-family:'IBM Plex Mono',monospace;font-weight:600;color:var(--accent);">0.0 s</span></p>
  </div>
  <p id="hb-err" class="missing" style="display:none;margin:0 0 8px;"></p>
  <div id="hb-plots" style="display:none;">
    <div id="hb-layout" style="display:grid;grid-template-columns:1fr;gap:16px;">
      <div id="hb-raw-col" style="display:none;">
        <p style="font-size:12px;margin:0 0 6px;font-weight:600;">Raw Detector Image (log scale)</p>
        <img id="hb-raw-img" style="width:100%;border-radius:4px;border:1px solid var(--border-subtle);">
      </div>
      <div>
        <p style="font-size:12px;margin:0 0 6px;font-weight:600;">1D Integration (Q)</p>
        <div id="hb-q" style="width:100%;height:360px;background:white;border-radius:4px;"></div>
      </div>
    </div>
    <div style="margin-top:10px;display:flex;align-items:center;gap:12px;">
      <button id="hb-img-btn" onclick="_hFetchImg(_hi)"
        style="font-size:12px;padding:5px 14px;background:rgba(148,163,184,0.12);color:var(--text-main);border:1px solid var(--border-subtle);border-radius:5px;">
        &#128444; Show Image</button>
      <p id="hb-meta" style="font-size:12px;margin:0;color:var(--text-muted);"></p>
    </div>
  </div>
</div>
<div class="card"><a href="/pyfai">&#8592; Back to pyFAI Integration</a></div>
"""

HELP_CONTENT = """
<div class="card"><h2 class="section-title">Help / Documentation</h2><p class="subheading">Quick reference for the QM2 Data Analysis interface.</p></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">1. Browse &amp; Analyze</span></div><ul><li>Navigate experiment tree via sidebar → Browse &amp; Analyze.</li><li>Click <b>Powder Data</b> next to a file for averaged radial sum plot.</li><li>Adjust X-min / X-max to focus Q-range. Export as CSV.</li></ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">2. NxRefine Viewer</span></div><ul><li>Click <b>NxRefine Viewer</b> in sidebar to view H/K/L slices.</li><li>Choose Single, Compare Two, or Compare Multiple files.</li><li>Set slice axis, values, colormap, vmin/vmax, and plot range.</li></ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">3. Linecut Tool</span></div><ul><li>After generating slices, click <b>Open Linecut Tool</b> or use the sidebar link.</li><li>Choose a file, axis, slice value, then specify start (X₁,Y₁) and end (X₂,Y₂) in r.l.u.</li><li>The tool returns an overlay image showing the cut and a 1D intensity profile (log scale).</li><li>Export profile as CSV for further analysis.</li></ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">4. Temperature Overlays</span></div><ul><li>From a folder, click <b>Temperature‑Dependent Powder Data</b>.</li><li>Select temperature-encoded files → generate overlay → export CSV.</li></ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">5. pyFAI Integration</span></div><ul><li>Single image or batch folder of .cbf files.</li><li>Supply PONI file; optionally a mask (.tif/.cbf/.npy).</li><li>Enable 1D (Q), Q→2θ, and/or Caking independently.</li></ul></div>
<div class="card"><div class="card-header"><div class="card-dot"></div><span class="card-title">6. Dark / Light Theme</span></div><ul><li>Use ☀ / ☾ button in the header. Preference stored in browser.</li></ul></div>
"""

# ── SLICE VIEWER TEMPLATE (with linecut launcher added) ───────────────────────
SLICE_VIEWER_CONTENT = """
<div class="card">
    <h2 class="section-title">NxRefine Data Viewer <span class="badge badge-gold">H / K / L Slices</span></h2>
    <p class="subheading">View single or multiple NXS files with flexible H, K, or L slice selection, parallel I/O caching, and shared colour scaling.</p>
</div>
{% if dirs or current %}
<div class="card">
    <div class="card-header"><div class="card-dot"></div><span class="card-title">Navigate</span></div>
    <p style="font-size:12px;margin:0 0 8px;">Current: <span class="path-label">{{ current }}</span></p>
    {% if dirs %}<ul>{% for d in dirs %}<li><a href="/slices?path={{ d }}">📁 {{ d.split('/')[-1] }}</a></li>{% endfor %}</ul>
    {% else %}<p style="font-size:12px;color:var(--text-muted);">No subfolders.</p>{% endif %}
</div>
{% endif %}
<div class="card">
    <div class="card-header"><div class="card-dot"></div><span class="card-title">Slice Configuration</span></div>
    <form method="POST" action="/slices" onsubmit="showStatus()">
        <input type="hidden" name="path" value="{{ current }}">
        <input type="hidden" name="active_tab" id="active_tab" value="{{ active_tab }}">
        {% if not files %}<p style="color:var(--text-muted);">No .nxs files in this folder.</p>{% else %}
        <div class="tab-bar">
            <button type="button" class="tab-btn {% if active_tab=='single' %}active{% endif %}" id="tab-single" onclick="switchTab('single')">Single File</button>
            <button type="button" class="tab-btn {% if active_tab=='compare' %}active{% endif %}" id="tab-compare" onclick="switchTab('compare')">Compare Two</button>
            <button type="button" class="tab-btn {% if active_tab=='multi' %}active{% endif %}" id="tab-multi" onclick="switchTab('multi')">Compare Multiple</button>
            <button type="button" class="tab-btn {% if active_tab=='linecut' %}active{% endif %}" id="tab-linecut" onclick="switchTab('linecut')" style="{% if active_tab=='linecut' %}color:#f0b429;border-bottom-color:#f0b429;{% endif %}">&#128208; 1D Linecut</button>
            <button type="button" class="tab-btn {% if active_tab=='thinfilm' %}active{% endif %}" id="tab-thinfilm" onclick="switchTab('thinfilm')" style="{% if active_tab=='thinfilm' %}color:#a78bfa;border-bottom-color:#a78bfa;{% endif %}">&#127910; Thin-Film</button>
            <button type="button" class="tab-btn {% if active_tab=='orderpar' %}active{% endif %}" id="tab-orderpar" onclick="switchTab('orderpar')" style="{% if active_tab=='orderpar' %}color:#ef4444;border-bottom-color:#ef4444;{% endif %}">&#128200; Order Parameter</button>
        </div>
        <div class="tab-panel {% if active_tab=='single' %}active{% endif %}" id="panel-single">
            <div class="field-group"><label class="field-label">NXS File</label><select name="file_a">{% for f in files %}<option value="{{ f }}" {% if f==file_a %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select></div>
        </div>
        <div class="tab-panel {% if active_tab=='compare' %}active{% endif %}" id="panel-compare">
            <div class="inline-fields">
                <div class="field-group"><label class="field-label">File A</label><select name="file_a_cmp">{% for f in files %}<option value="{{ f }}" {% if f==file_a_cmp %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select></div>
                <div class="field-group"><label class="field-label">File B</label><select name="file_b_cmp">{% for f in files %}<option value="{{ f }}" {% if f==file_b_cmp %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select></div>
            </div>
        </div>
        <div class="tab-panel {% if active_tab=='multi' %}active{% endif %}" id="panel-multi">
            <div class="field-group">
                <label class="field-label">Select Files &nbsp;<span style="font-weight:400;text-transform:none;font-size:11px;color:var(--text-muted);">Hold Ctrl / ⌘ for multiple</span></label>
                <select name="files_multi" multiple>{% for f in files %}<option value="{{ f }}" {% if f in multi_selected %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select>
                <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">{{ multi_selected|length }} selected</div>
            </div>
        </div>
        <div class="tab-panel {% if active_tab=='linecut' %}active{% endif %}" id="panel-linecut">
            <!-- Mode selector + shared options -->
            <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap;align-items:center;">
                <span style="font-size:12px;font-weight:600;color:var(--text-muted);">Mode:</span>
                <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;padding:5px 12px;border-radius:6px;border:1px solid rgba(0,200,255,0.3);background:rgba(0,200,255,0.05);">
                    <input type="radio" name="lc_mode" value="single"
                           {% if lc_mode=='single' %}checked{% endif %}
                           onchange="switchLcMode('single')"> Single
                </label>
                <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;padding:5px 12px;border-radius:6px;border:1px solid rgba(240,180,41,0.35);background:rgba(240,180,41,0.05);">
                    <input type="radio" name="lc_mode" value="compare"
                           {% if lc_mode=='compare' %}checked{% endif %}
                           onchange="switchLcMode('compare')"> &#128293; Compare Two
                </label>
                <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;padding:5px 12px;border-radius:6px;border:1px solid rgba(74,222,128,0.35);background:rgba(74,222,128,0.05);">
                    <input type="radio" name="lc_mode" value="multi"
                           {% if lc_mode=='multi' %}checked{% endif %}
                           onchange="switchLcMode('multi')"> &#127777; Multi-Temperature
                </label>
                <div class="field-group" style="margin:0 0 0 8px;"><label class="field-label">Sample pts</label>
                    <input type="number" name="lc_npts" value="{{ lc_npts }}" min="50" max="2000" style="width:90px;">
                </div>
                <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;padding:5px 10px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);">
                    <input type="checkbox" name="lc_autoscale" id="lc_autoscale_chk" value="1"
                           {% if lc_autoscale %}checked{% endif %}> Auto-scale
                </label>
            </div>
            <!-- ── SHARED: Projection Settings (always visible, used by all modes) ── -->
            <div id="lc-shared-row" style="background:rgba(0,200,255,0.04);border:1px solid rgba(0,200,255,0.2);border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <div style="font-size:11px;font-weight:700;color:#00c8ff;letter-spacing:.08em;margin-bottom:8px;">
                    &#9632; PROJECTION SETTINGS
                    <span style="font-weight:400;text-transform:none;font-size:10px;color:var(--text-muted);">&nbsp;— shared across all modes</span>
                </div>
                <!-- Transform path -->
                <div class="inline-fields" style="margin-bottom:8px;">
                    <div class="field-group"><label class="field-label">Transform path</label>
                        <input type="text" name="lc_transform_path" value="{{ lc_transform_path }}" placeholder="entry/transform" style="width:220px;">
                    </div>
                    <label style="display:flex;align-items:center;gap:5px;font-size:12px;cursor:pointer;padding-top:14px;">
                        <input type="checkbox" name="lc_plot_lines" value="1" {% if lc_plot_lines %}checked{% endif %}>
                        <span style="color:#00c8ff;font-weight:600;">Plot Lines</span>
                    </label>
                </div>
                <!-- Slice Axis H/K/L + X/Y-Axis dropdowns -->
                <div style="display:flex;gap:6px;align-items:center;margin-bottom:10px;flex-wrap:wrap;">
                    <span style="font-size:12px;font-weight:700;color:#00c8ff;min-width:68px;">Slice Axis:</span>
                    <label style="display:flex;align-items:center;gap:5px;font-size:13px;cursor:pointer;padding:4px 14px;border-radius:6px;border:1px solid rgba(0,200,255,0.4);background:{% if lc_xaxis=='Qh' %}rgba(0,200,255,0.2){% else %}rgba(0,200,255,0.05){% endif %};">
                        <input type="radio" name="lc_slice_axis_btn" value="H" {% if lc_xaxis=='Qh' %}checked{% endif %}
                               onchange="lcSetSliceAxis('H')"><span style="font-weight:600;">H</span>
                    </label>
                    <label style="display:flex;align-items:center;gap:5px;font-size:13px;cursor:pointer;padding:4px 14px;border-radius:6px;border:1px solid rgba(0,200,255,0.4);background:{% if lc_xaxis=='Qk' %}rgba(0,200,255,0.2){% else %}rgba(0,200,255,0.05){% endif %};">
                        <input type="radio" name="lc_slice_axis_btn" value="K" {% if lc_xaxis=='Qk' %}checked{% endif %}
                               onchange="lcSetSliceAxis('K')"><span style="font-weight:600;">K</span>
                    </label>
                    <label style="display:flex;align-items:center;gap:5px;font-size:13px;cursor:pointer;padding:4px 14px;border-radius:6px;border:1px solid rgba(0,200,255,0.4);background:{% if lc_xaxis=='Ql' %}rgba(0,200,255,0.2){% else %}rgba(0,200,255,0.05){% endif %};">
                        <input type="radio" name="lc_slice_axis_btn" value="L" {% if lc_xaxis=='Ql' %}checked{% endif %}
                               onchange="lcSetSliceAxis('L')"><span style="font-weight:600;">L</span>
                    </label>
                    <span style="font-size:11px;color:var(--text-muted);margin-left:6px;">→ sets X-Axis</span>
                    <div style="margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
                        <div class="field-group" style="margin:0;"><label class="field-label" style="color:#00c8ff;margin-bottom:0;font-size:10px;">X-Axis</label>
                            <select name="lc_xaxis" id="lc_xaxis_sel" style="min-width:90px;font-size:12px;" onchange="lcSyncAxisBtns()">
                                <option value="Ql" {% if lc_xaxis=='Ql' %}selected{% endif %}>Ql</option>
                                <option value="Qk" {% if lc_xaxis=='Qk' %}selected{% endif %}>Qk</option>
                                <option value="Qh" {% if lc_xaxis=='Qh' %}selected{% endif %}>Qh</option>
                                <option value="None" {% if lc_xaxis=='None' %}selected{% endif %}>None</option>
                            </select>
                        </div>
                        <div class="field-group" style="margin:0;"><label class="field-label" style="color:#00c8ff;margin-bottom:0;font-size:10px;">Y-Axis (2D)</label>
                            <select name="lc_yaxis" id="lc_yaxis_sel" style="min-width:90px;font-size:12px;">
                                <option value="None" {% if lc_yaxis=='None' %}selected{% endif %}>None</option>
                                <option value="Ql" {% if lc_yaxis=='Ql' %}selected{% endif %}>Ql</option>
                                <option value="Qk" {% if lc_yaxis=='Qk' %}selected{% endif %}>Qk</option>
                                <option value="Qh" {% if lc_yaxis=='Qh' %}selected{% endif %}>Qh</option>
                            </select>
                        </div>
                    </div>
                </div>
                <!-- ROI grid: Ql/Qk/Qh min/max + Lock -->
                <div style="display:grid;grid-template-columns:50px 1fr 1fr 40px;gap:4px 8px;align-items:center;font-size:12px;">
                    <span style="font-weight:600;color:var(--text-muted);">Axis</span>
                    <span style="font-weight:600;color:var(--text-muted);">Minimum</span>
                    <span style="font-weight:600;color:var(--text-muted);">Maximum</span>
                    <span style="font-weight:600;color:var(--text-muted);">Lock</span>
                    <span style="font-weight:600;color:#00c8ff;">Ql</span>
                    <input type="number" step="0.01" name="lc_ql_min" value="{{ lc_ql_min }}" style="width:100%;">
                    <input type="number" step="0.01" name="lc_ql_max" value="{{ lc_ql_max }}" style="width:100%;">
                    <input type="checkbox" name="lc_ql_lock" value="1" {% if lc_ql_lock %}checked{% endif %} title="Lock Ql range">
                    <span style="font-weight:600;color:#00c8ff;">Qk</span>
                    <input type="number" step="0.01" name="lc_qk_min" value="{{ lc_qk_min }}" style="width:100%;">
                    <input type="number" step="0.01" name="lc_qk_max" value="{{ lc_qk_max }}" style="width:100%;">
                    <input type="checkbox" name="lc_qk_lock" value="1" {% if lc_qk_lock %}checked{% endif %} title="Lock Qk range">
                    <span style="font-weight:600;color:#00c8ff;">Qh</span>
                    <input type="number" step="0.01" name="lc_qh_min" value="{{ lc_qh_min }}" style="width:100%;">
                    <input type="number" step="0.01" name="lc_qh_max" value="{{ lc_qh_max }}" style="width:100%;">
                    <input type="checkbox" name="lc_qh_lock" value="1" {% if lc_qh_lock %}checked{% endif %} title="Lock Qh range">
                </div>
            </div>
            <!-- ── FILE A (single/compare mode only) ── -->
            <div id="lc-single-row" style="{% if lc_mode=='multi' %}display:none;{% endif %}background:rgba(0,200,255,0.06);border:1px solid rgba(0,200,255,0.25);border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <div style="font-size:11px;font-weight:700;color:#00c8ff;letter-spacing:.08em;margin-bottom:8px;">
                    &#9632; {% if lc_mode=='compare' %}FILE A{% else %}NXS FILE{% endif %}
                </div>
                <div class="inline-fields" style="margin-bottom:0;">
                    <div class="field-group"><label class="field-label">NXS File{% if lc_mode=='compare' %} A{% endif %}</label>
                        <select name="lc_file">{% for f in files %}<option value="{{ f }}" {% if f==lc_file %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select>
                    </div>
                    <div class="field-group"><label class="field-label">Transform path{% if lc_mode=='compare' %} A (override){% endif %}</label>
                        <input type="text" name="lc_file_a_transform_path" value="{{ lc_file_a_transform_path }}" placeholder="(use shared above)" style="width:200px;">
                        <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">Leave blank to use shared transform path</div>
                    </div>
                </div>
            </div>
            <!-- ── FILE B (compare mode only) ── -->
            <div id="lc-compare-row" style="{% if lc_mode!='compare' %}display:none;{% endif %}background:rgba(240,180,41,0.07);border:1px solid rgba(240,180,41,0.3);border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <div style="font-size:11px;font-weight:700;color:#f0b429;letter-spacing:.08em;margin-bottom:8px;">
                    &#9632; FILE B &nbsp;<span style="font-weight:400;text-transform:none;font-size:10px;color:var(--text-muted);">Slice axis &amp; ROI shared from above</span>
                </div>
                <div class="inline-fields" style="margin-bottom:0;">
                    <div class="field-group"><label class="field-label" style="color:#f0b429;">NXS File B</label>
                        <select name="lc_file_b">{% for f in files %}<option value="{{ f }}" {% if f==lc_file_b %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select>
                    </div>
                    <div class="field-group"><label class="field-label" style="color:#f0b429;">Transform path B (override)</label>
                        <input type="text" name="lc_file_b_transform_path" value="{{ lc_file_b_transform_path }}" placeholder="(use shared above)" style="width:200px;">
                        <div style="font-size:10px;color:var(--text-muted);margin-top:2px;">Leave blank to use shared transform path</div>
                    </div>
                </div>
            </div>
            <!-- ── MULTI-TEMPERATURE ── -->
            <div id="lc-multi-row" style="{% if lc_mode!='multi' %}display:none;{% endif %}background:rgba(74,222,128,0.06);border:1px solid rgba(74,222,128,0.3);border-radius:8px;padding:10px 14px;margin-bottom:8px;">
                <div style="font-size:11px;font-weight:700;color:#4ade80;letter-spacing:.08em;margin-bottom:8px;">
                    &#127777; MULTI-TEMPERATURE &nbsp;<span style="font-weight:400;text-transform:none;font-size:11px;color:var(--text-muted);">Hold Ctrl / ⌘ to select multiple files · Slice axis &amp; ROI from above</span>
                </div>
                <div class="inline-fields" style="margin-bottom:0;align-items:flex-start;">
                    <div class="field-group">
                        <label class="field-label" style="color:#4ade80;">NXS Files (select multiple)</label>
                        <select name="lc_files_multi" multiple style="min-width:340px;min-height:130px;font-size:12px;">
                            {% for f in files %}<option value="{{ f }}" {% if f in lc_files_multi %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}
                        </select>
                        <div style="font-size:11px;color:var(--text-muted);margin-top:3px;">{{ lc_files_multi|length }} selected</div>
                    </div>
                </div>
                <!-- Analysis options -->
                <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(74,222,128,0.2);display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">
                    <label style="display:flex;align-items:center;gap:7px;font-size:13px;cursor:pointer;padding:6px 12px;border-radius:6px;border:1px solid rgba(74,222,128,0.3);background:rgba(74,222,128,0.06);">
                        <input type="checkbox" name="lc_show_overlays" value="1"
                               {% if lc_show_overlays %}checked{% endif %}>
                        <span style="color:#4ade80;font-weight:600;">&#128444; Slice Overlays</span>
                    </label>
                    <div style="display:flex;flex-direction:column;gap:8px;">
                    <label style="display:flex;align-items:center;gap:7px;font-size:13px;cursor:pointer;padding:6px 12px;border-radius:6px;border:1px solid rgba(167,139,250,0.35);background:rgba(167,139,250,0.06);">
                        <input type="checkbox" name="lc_show_heatmap" value="1"
                               {% if lc_show_heatmap %}checked{% endif %}
                               onchange="document.getElementById('lc-heatmap-opts').style.display=this.checked?'flex':'none'">
                        <span style="color:#a78bfa;font-weight:600;">&#127782; T vs Q Heatmap</span>
                    </label>
                    <div id="lc-heatmap-opts" style="display:{% if lc_show_heatmap %}flex{% else %}none{% endif %};flex-wrap:wrap;gap:12px;align-items:flex-end;padding:8px 12px;background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.3);border-radius:6px;">
                        <label style="display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer;">
                            <input type="checkbox" name="hm_autoscale" value="1"
                                   {% if hm_autoscale %}checked{% endif %}
                                   onchange="var d=document.getElementById('hm-range-grp');d.style.display=this.checked?'none':'flex';">
                            <span style="color:#a78bfa;font-weight:500;">Auto-scale</span>
                        </label>
                        <div id="hm-range-grp" style="display:{% if hm_autoscale %}none{% else %}flex{% endif %};gap:10px;align-items:flex-end;">
                            <div class="field-group" style="margin:0;">
                                <label class="field-label" style="color:#a78bfa;">vmin</label>
                                <input type="number" step="any" name="hm_vmin" value="{{ hm_vmin }}" style="width:100px;">
                            </div>
                            <div class="field-group" style="margin:0;">
                                <label class="field-label" style="color:#a78bfa;">vmax</label>
                                <input type="number" step="any" name="hm_vmax" value="{{ hm_vmax }}" style="width:100px;">
                            </div>
                        </div>
                    </div>
                    </div>
                    <div style="display:flex;flex-direction:column;gap:8px;">
                        <label style="display:flex;align-items:center;gap:7px;font-size:13px;cursor:pointer;padding:6px 12px;border-radius:6px;border:1px solid rgba(251,146,60,0.35);background:rgba(251,146,60,0.06);">
                            <input type="checkbox" name="lc_show_peakfit" value="1"
                                   {% if lc_show_peakfit %}checked{% endif %}
                                   onchange="document.getElementById('lc-peakfit-roi').style.display=this.checked?'flex':'none'">
                            <span style="color:#fb923c;font-weight:600;">&#128200; Peak Fit (Gaussian + Linear bg)</span>
                        </label>
                        <div id="lc-peakfit-roi" style="display:{% if lc_show_peakfit %}flex{% else %}none{% endif %};flex-wrap:wrap;gap:12px;align-items:flex-end;padding:8px 12px;background:rgba(251,146,60,0.07);border:1px solid rgba(251,146,60,0.3);border-radius:6px;">
                            <div class="field-group" style="margin:0;">
                                <label class="field-label" style="color:#fb923c;">Fit Q-min (r.l.u.)</label>
                                <input type="number" step="0.01" name="lc_fit_qmin" value="{{ lc_fit_qmin }}" style="width:90px;">
                            </div>
                            <div class="field-group" style="margin:0;">
                                <label class="field-label" style="color:#fb923c;">Fit Q-max (r.l.u.)</label>
                                <input type="number" step="0.01" name="lc_fit_qmax" value="{{ lc_fit_qmax }}" style="width:90px;">
                            </div>
                            <div class="field-group" style="margin:0;">
                                <label class="field-label" style="color:#fb923c;">Peak center init</label>
                                <input type="number" step="0.001" name="lc_fit_center" value="{{ lc_fit_center }}" style="width:100px;">
                            </div>
                        </div>
                        {# ── 4th option: Peak Parameters vs T ── #}
                        <label style="display:flex;align-items:center;gap:7px;font-size:13px;cursor:pointer;padding:6px 12px;border-radius:6px;border:1px solid rgba(56,189,248,0.35);background:rgba(56,189,248,0.06);">
                            <input type="checkbox" name="lc_show_peakplot" value="1"
                                   {% if lc_show_peakplot %}checked{% endif %}>
                            <span style="color:#38bdf8;font-weight:600;">&#128202; Peak Parameters vs T &nbsp;<span style="font-weight:400;font-size:11px;opacity:0.7;">(requires Peak Fit)</span></span>
                        </label>
                    </div>
                </div>
            </div>
            <!-- Hidden linecut endpoints (kept for multi/compare backward compat) -->
            <input type="hidden" name="lc_x1" value="{{ lc_x1 }}">
            <input type="hidden" name="lc_y1" value="{{ lc_y1 }}">
            <input type="hidden" name="lc_x2" value="{{ lc_x2 }}">
            <input type="hidden" name="lc_y2" value="{{ lc_y2 }}">
        </div>
        <div class="tab-panel {% if active_tab=='thinfilm' %}active{% endif %}" id="panel-thinfilm">
            <div style="background:rgba(167,139,250,0.08);border:1px solid rgba(167,139,250,0.3);border-radius:8px;padding:12px 16px;">
                <div style="font-size:11px;font-weight:700;color:#a78bfa;letter-spacing:.08em;margin-bottom:10px;">&#127910; THIN-FILM SLICE VIEWER (NxRefine)</div>
                <div class="inline-fields">
                    <div class="field-group"><label class="field-label">NXS File</label>
                        <select name="tf_file">{% for f in files %}<option value="{{ f }}" {% if f==tf_file %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select>
                    </div>
                    <div class="field-group"><label class="field-label">Slice Axis</label>
                        <select name="tf_slice_axis" id="tf-slice-axis-sel" onchange="tfUpdateAxisLabel()" style="min-width:200px;">
                            <option value="Ql" {% if tf_slice_axis=='Ql' %}selected{% endif %}>Ql — fix L, view H-K</option>
                            <option value="Qk" {% if tf_slice_axis=='Qk' %}selected{% endif %}>Qk — fix K, view H-L</option>
                            <option value="Qh" {% if tf_slice_axis=='Qh' %}selected{% endif %}>Qh — fix H, view K-L</option>
                        </select>
                    </div>
                </div>
                <div class="inline-fields" style="margin-top:10px;">
                    <div class="field-group"><label class="field-label">Transform path</label>
                        <input type="text" name="tf_transform_path" value="{{ tf_transform_path }}" style="width:180px;"></div>
                    <div class="field-group"><label class="field-label">Signal name</label>
                        <input type="text" name="tf_signal" value="{{ tf_signal }}" style="width:90px;"></div>
                    <div class="field-group"><label class="field-label">Qh axis</label>
                        <input type="text" name="tf_qh" value="{{ tf_qh }}" style="width:70px;"></div>
                    <div class="field-group"><label class="field-label">Qk axis</label>
                        <input type="text" name="tf_qk" value="{{ tf_qk }}" style="width:70px;"></div>
                    <div class="field-group"><label class="field-label">Ql axis</label>
                        <input type="text" name="tf_ql" value="{{ tf_ql }}" style="width:70px;"></div>
                </div>
                <!-- Row 3: axis values + rotation + colormap + mode -->
                <div class="inline-fields" style="margin-top:10px;">
                    <div class="field-group"><label class="field-label"><span id="tf-axis-label">{{ tf_slice_axis }}</span> values (Å⁻¹, comma-separated)</label>
                        <input type="text" name="tf_qls" value="{{ tf_qls }}" placeholder="e.g. 0, 0.5, 1, 1.5" style="width:240px;"></div>
                    <div class="field-group"><label class="field-label">Rotation (°)</label>
                        <input type="number" name="tf_rotation" value="{{ tf_rotation }}" step="1" style="width:100px;"></div>
                    <div class="field-group"><label class="field-label">Colormap</label>
                        <select name="tf_cmap" style="min-width:130px;">
                            {% for cm in ['viridis','plasma','inferno','RdBu_r','coolwarm','hot','jet'] %}
                            <option value="{{ cm }}" {% if tf_cmap==cm %}selected{% endif %}>{{ cm }}</option>
                            {% endfor %}
                        </select></div>
                    <div class="field-group"><label class="field-label">Mode</label>
                        <select name="tf_mode" style="min-width:120px;">
                            <option value="linear" {% if tf_mode=='linear' %}selected{% endif %}>Linear</option>
                            <option value="log"    {% if tf_mode=='log'    %}selected{% endif %}>log₁₀(I+1)</option>
                            <option value="sqrt"   {% if tf_mode=='sqrt'   %}selected{% endif %}>√I  (sqrt)</option>
                        </select></div>
                    <div class="field-group"><label class="field-label">Max px/side <span style="font-weight:400;text-transform:none;">(speed)</span></label>
                        <input type="number" name="tf_maxpx" value="{{ tf_maxpx }}" min="100" max="2400" step="100" style="width:90px;"></div>
                </div>
                <!-- Row 4: vmin/vmax — autoscale toggle -->
                <div class="inline-fields" style="margin-top:10px;align-items:flex-end;">
                    <div class="field-group" style="align-self:center;padding-bottom:4px;">
                        <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;">
                            <input type="checkbox" id="tf-autoscale-chk" name="tf_autoscale" value="1"
                                   {% if tf_autoscale %}checked{% endif %}
                                   onchange="tfToggleScale(this.checked)"> Autoscale (percentile)
                        </label>
                    </div>
                    <!-- percentile group (visible when autoscale ON) -->
                    <div id="tf-pct-grp" style="display:{% if tf_autoscale %}flex{% else %}none{% endif %};gap:16px;">
                        <div class="field-group"><label class="field-label">vmin %ile</label>
                            <input type="number" name="tf_vmin_pct" value="{{ tf_vmin_pct }}" step="0.5" min="0" max="50" style="width:80px;"></div>
                        <div class="field-group"><label class="field-label">vmax %ile</label>
                            <input type="number" name="tf_vmax_pct" value="{{ tf_vmax_pct }}" step="0.5" min="50" max="100" style="width:80px;"></div>
                    </div>
                    <!-- explicit group (visible when autoscale OFF) -->
                    <div id="tf-manual-grp" style="display:{% if not tf_autoscale %}flex{% else %}none{% endif %};gap:16px;">
                        <div class="field-group"><label class="field-label">vmin</label>
                            <input type="number" name="tf_vmin" value="{{ tf_vmin }}" step="any" style="width:110px;"></div>
                        <div class="field-group"><label class="field-label">vmax</label>
                            <input type="number" name="tf_vmax" value="{{ tf_vmax }}" step="any" style="width:110px;"></div>
                    </div>
                </div>
                <!-- Row 5: X / Y axis range -->
                <div class="inline-fields" style="margin-top:10px;align-items:flex-end;">
                    <div class="field-group" style="align-self:center;padding-bottom:4px;">
                        <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;">
                            <input type="checkbox" id="tf-axauto-chk" name="tf_axauto" value="1"
                                   {% if tf_axauto %}checked{% endif %}
                                   onchange="tfToggleAxRange(this.checked)"> Auto X/Y range
                        </label>
                    </div>
                    <div id="tf-axrange-grp" style="display:{% if not tf_axauto %}flex{% else %}none{% endif %};gap:16px;flex-wrap:wrap;">
                        <div class="field-group">
                            <label class="field-label" style="color:#a78bfa;">X min (<span class="tf-xax-lbl">{% if tf_slice_axis=='Ql' %}Qh{% elif tf_slice_axis=='Qk' %}Qh{% else %}Qk{% endif %}'</span>)</label>
                            <input type="number" name="tf_xmin" value="{{ tf_xmin }}" step="0.01" style="width:100px;">
                        </div>
                        <div class="field-group">
                            <label class="field-label" style="color:#a78bfa;">X max (<span class="tf-xax-lbl">{% if tf_slice_axis=='Ql' %}Qh{% elif tf_slice_axis=='Qk' %}Qh{% else %}Qk{% endif %}'</span>)</label>
                            <input type="number" name="tf_xmax" value="{{ tf_xmax }}" step="0.01" style="width:100px;">
                        </div>
                        <div class="field-group">
                            <label class="field-label" style="color:#a78bfa;">Y min (<span class="tf-yax-lbl">{% if tf_slice_axis=='Ql' %}Qk{% elif tf_slice_axis=='Qk' %}Ql{% else %}Ql{% endif %}'</span>)</label>
                            <input type="number" name="tf_ymin" value="{{ tf_ymin }}" step="0.01" style="width:100px;">
                        </div>
                        <div class="field-group">
                            <label class="field-label" style="color:#a78bfa;">Y max (<span class="tf-yax-lbl">{% if tf_slice_axis=='Ql' %}Qk{% elif tf_slice_axis=='Qk' %}Ql{% else %}Ql{% endif %}'</span>)</label>
                            <input type="number" name="tf_ymax" value="{{ tf_ymax }}" step="0.01" style="width:100px;">
                        </div>
                    </div>
                </div>
                <button type="submit" formaction="/slices/thinfilm" onclick="showStatus()"
                        style="margin-top:14px;background:linear-gradient(135deg,#6d28d9 0%,#a78bfa 100%);color:#fff;border:none;border-radius:5px;padding:9px 22px;font-weight:600;font-size:12px;letter-spacing:0.04em;cursor:pointer;text-transform:uppercase;box-shadow:0 2px 14px rgba(167,139,250,0.25);">
                    &#9654; Generate Maps
                </button>
            </div>
            {% if tf_error %}<div class="alert-error" style="margin-top:12px;"><b>Error:</b><pre style="margin:6px 0 0;max-height:280px;overflow-y:auto;">{{ tf_error }}</pre></div>{% endif %}
            {% if tf_info %}
            <div style="margin-top:8px;font-size:12px;color:var(--text-muted);font-family:'IBM Plex Mono',monospace;background:rgba(0,0,0,0.12);padding:8px 12px;border-radius:5px;white-space:pre;">{{ tf_info }}{% if tf_elapsed is not none %}
&#9201; Generated {{ tf_rows|length if tf_rows else 0 }} map(s) in {{ "%.2f"|format(tf_elapsed) }} s{% endif %}</div>
            {% endif %}
            {% if tf_rows %}
            <div class="slice-grid" style="margin-top:16px;">
                {% for row in tf_rows %}
                <div class="slice-box" style="flex:0 0 auto;min-width:200px;max-width:380px;">
                    <div class="slice-filename">
                        <div class="slice-dot" style="background:#a78bfa;box-shadow:0 0 6px #a78bfa;"></div>
                        {{ row.axis_name }} = {{ "%.4f"|format(row.axis_val) }} Å⁻¹
                        {% if row.error %}<span class="badge badge-red">error</span>{% endif %}
                    </div>
                    {% if row.img %}
                    <img class="slice-img" src="data:image/png;base64,{{ row.img }}">
                    <a class="dl-btn" href="data:image/png;base64,{{ row.img }}"
                       download="thinfilm_{{ row.axis_name }}_{{ '%.4f'|format(row.axis_val) }}.png">&#11015; PNG</a>
                    {% else %}<div class="alert-error" style="font-size:12px;">{{ row.error }}</div>{% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
        {# ═══ ORDER PARAMETER TAB ═══ #}
        <div class="tab-panel {% if active_tab=='orderpar' %}active{% endif %}" id="panel-orderpar">
            <div style="background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:14px 18px;">
                <div style="font-size:11px;font-weight:700;color:#ef4444;letter-spacing:.08em;margin-bottom:12px;">&#128200; ORDER PARAMETER — Temperature-Dependent 1D Linecut Analysis</div>
                <p style="font-size:12px;color:var(--text-muted);margin:0 0 12px;">
                    Select <b>NXS files</b> at different temperatures.
                    Define a <b>Q-space ROI</b> (Qh, Qk, Ql min/max), pick a <b>cut axis</b> for 1D linecuts, and track the integrated intensity vs temperature.
                </p>
                <!-- NXS files + transform path -->
                <div class="inline-fields" style="align-items:flex-start;margin-bottom:12px;">
                    <div class="field-group">
                        <label class="field-label" style="color:#ef4444;">NXS Files (select multiple) &nbsp;<span style="font-weight:400;text-transform:none;font-size:11px;color:var(--text-muted);">Hold Ctrl / &#8984;</span></label>
                        <select name="op_files" multiple style="min-width:360px;min-height:120px;font-size:12px;">
                            {% for f in files %}<option value="{{ f }}" {% if f in op_files %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}
                        </select>
                        <div style="font-size:11px;color:var(--text-muted);margin-top:3px;">{{ op_files|length }} selected</div>
                    </div>
                    <div class="field-group">
                        <label class="field-label" style="color:#ef4444;">Transform path inside NXS</label>
                        <input type="text" name="op_transform_path" value="{{ op_transform_path }}" placeholder="entry/transform" style="width:260px;">
                    </div>
                </div>
                <!-- 1D Linecut: Qh/Qk/Ql min/max + cut axis -->
                <div style="margin-bottom:14px;padding:10px 14px;background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.25);border-radius:8px;">
                    <div style="font-size:11px;font-weight:600;color:#3b82f6;letter-spacing:.06em;margin-bottom:8px;">&#128208; Q-SPACE ROI &amp; 1D LINECUT</div>
                    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px;">
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:12px;font-weight:600;color:#3b82f6;min-width:28px;">Qh:</span>
                            <div class="field-group" style="margin:0;"><label class="field-label">min</label><input type="number" step="0.01" name="op_qh_min" value="{{ op_qh_min }}" style="width:85px;"></div>
                            <div class="field-group" style="margin:0;"><label class="field-label">max</label><input type="number" step="0.01" name="op_qh_max" value="{{ op_qh_max }}" style="width:85px;"></div>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:12px;font-weight:600;color:#3b82f6;min-width:28px;">Qk:</span>
                            <div class="field-group" style="margin:0;"><label class="field-label">min</label><input type="number" step="0.01" name="op_qk_min" value="{{ op_qk_min }}" style="width:85px;"></div>
                            <div class="field-group" style="margin:0;"><label class="field-label">max</label><input type="number" step="0.01" name="op_qk_max" value="{{ op_qk_max }}" style="width:85px;"></div>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:12px;font-weight:600;color:#3b82f6;min-width:28px;">Ql:</span>
                            <div class="field-group" style="margin:0;"><label class="field-label">min</label><input type="number" step="0.01" name="op_ql_min" value="{{ op_ql_min }}" style="width:85px;"></div>
                            <div class="field-group" style="margin:0;"><label class="field-label">max</label><input type="number" step="0.01" name="op_ql_max" value="{{ op_ql_max }}" style="width:85px;"></div>
                        </div>
                    </div>
                    <div class="inline-fields">
                        <div class="field-group" style="margin:0;">
                            <label class="field-label" style="color:#3b82f6;">Cut Axis (1D profile along)</label>
                            <select name="op_cut_axis" style="min-width:190px;">
                                <option value="H" {% if op_cut_axis=='H' %}selected{% endif %}>Qh (sum over Qk, Ql)</option>
                                <option value="K" {% if op_cut_axis=='K' %}selected{% endif %}>Qk (sum over Qh, Ql)</option>
                                <option value="L" {% if op_cut_axis=='L' %}selected{% endif %}>Ql (sum over Qh, Qk)</option>
                            </select>
                        </div>
                        <label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;padding-top:16px;">
                            <input type="checkbox" name="op_lc_auto" value="1" {% if op_lc_auto %}checked{% endif %}> Auto-scale Y
                        </label>
                        <div class="field-group" style="margin:0;"><label class="field-label">LC vmin</label><input type="number" step="any" name="op_lc_vmin" value="{{ op_lc_vmin }}" style="width:80px;"></div>
                        <div class="field-group" style="margin:0;"><label class="field-label">LC vmax</label><input type="number" step="any" name="op_lc_vmax" value="{{ op_lc_vmax }}" style="width:80px;"></div>
                    </div>
                </div>
                <!-- Integrated Intensity axis controls -->
                <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin-bottom:12px;padding:8px 12px;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.25);border-radius:6px;">
                    <span style="font-size:11px;font-weight:600;color:#ef4444;">&#128200; Integ. Intensity Plot</span>
                    <label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;">
                        <input type="checkbox" name="op_int_auto" value="1" {% if op_int_auto %}checked{% endif %}> Auto-scale
                    </label>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#ef4444;">I min</label><input type="number" step="any" name="op_int_ymin" value="{{ op_int_ymin }}" style="width:100px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#ef4444;">I max</label><input type="number" step="any" name="op_int_ymax" value="{{ op_int_ymax }}" style="width:100px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#ef4444;">T min (K)</label><input type="number" step="any" name="op_int_tmin" value="{{ op_int_tmin }}" style="width:90px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#ef4444;">T max (K)</label><input type="number" step="any" name="op_int_tmax" value="{{ op_int_tmax }}" style="width:90px;"></div>
                </div>
                <!-- Heatmap options -->
                <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin-bottom:12px;padding:8px 12px;background:rgba(167,139,250,0.06);border:1px solid rgba(167,139,250,0.25);border-radius:6px;">
                    <span style="font-size:11px;font-weight:600;color:#a78bfa;">&#127782; Heatmap</span>
                    <label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;">
                        <input type="checkbox" name="op_show_heatmap" value="1" {% if op_show_heatmap %}checked{% endif %}> Show heatmap
                    </label>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#a78bfa;">vmin</label><input type="number" step="any" name="op_hm_vmin" value="{{ op_hm_vmin }}" style="width:100px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#a78bfa;">vmax</label><input type="number" step="any" name="op_hm_vmax" value="{{ op_hm_vmax }}" style="width:100px;"></div>
                    <label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;">
                        <input type="checkbox" name="op_hm_auto" value="1" {% if op_hm_auto %}checked{% endif %}> Auto-scale
                    </label>
                </div>
                <!-- Peak fit options -->
                <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px;padding:8px 12px;background:rgba(251,146,60,0.06);border:1px solid rgba(251,146,60,0.25);border-radius:6px;">
                    <span style="font-size:11px;font-weight:600;color:#fb923c;">&#128200; Peak Fit</span>
                    <label style="font-size:12px;cursor:pointer;display:flex;align-items:center;gap:5px;">
                        <input type="checkbox" name="op_do_fit" value="1" {% if op_do_fit %}checked{% endif %}> Enable peak fitting
                    </label>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#fb923c;">Fit Q-min</label><input type="number" step="0.01" name="op_fit_qmin" value="{{ op_fit_qmin }}" style="width:90px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#fb923c;">Fit Q-max</label><input type="number" step="0.01" name="op_fit_qmax" value="{{ op_fit_qmax }}" style="width:90px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#fb923c;">Center init</label><input type="number" step="0.001" name="op_fit_center" value="{{ op_fit_center }}" style="width:100px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#fb923c;">Center min</label><input type="number" step="0.01" name="op_fit_cmin" value="{{ op_fit_cmin }}" style="width:90px;"></div>
                    <div class="field-group" style="margin:0;"><label class="field-label" style="color:#fb923c;">Center max</label><input type="number" step="0.01" name="op_fit_cmax" value="{{ op_fit_cmax }}" style="width:90px;"></div>
                </div>
                <button type="submit" formaction="/slices/orderpar" onclick="showStatus()"
                        style="background:linear-gradient(135deg,#b91c1c 0%,#ef4444 100%);color:#fff;border:none;border-radius:5px;padding:9px 22px;font-weight:600;font-size:12px;letter-spacing:0.04em;cursor:pointer;text-transform:uppercase;box-shadow:0 2px 14px rgba(239,68,68,0.25);">
                    &#9654; Compute Order Parameter
                </button>
            </div>
            {% if op_error %}<div class="alert-error" style="margin-top:12px;"><b>Error:</b><pre style="margin:6px 0 0;max-height:280px;overflow-y:auto;">{{ op_error }}</pre></div>{% endif %}
            {% if op_scan_info %}<div style="margin-top:10px;font-size:12px;color:var(--text-muted);font-family:'IBM Plex Mono',monospace;background:rgba(0,0,0,0.08);padding:8px 12px;border-radius:5px;">{{ op_scan_info }}{% if op_elapsed is not none %} &#9201; {{ "%.2f"|format(op_elapsed) }}s{% endif %}</div>{% endif %}
            {# ── PRIMARY: Integrated Intensity vs Temperature ── #}
            {% if op_integ_json %}
            <div class="card" style="border-color:rgba(239,68,68,0.5);margin-top:16px;">
                <div class="card-header"><div class="card-dot" style="background:#ef4444;box-shadow:0 0 10px #ef4444;"></div>
                    <span class="card-title" style="color:#ef4444;">&#128200; Integrated Intensity vs Temperature</span>
                    <span class="slice-timing" style="margin-left:auto;">{{ op_ntemps }} temperatures</span>
                </div>
                <div id="op-plotly-integ" style="width:100%;height:420px;"></div>
                <script>(function(){
                    var d = {{ op_integ_json | safe }};
                    var trace = {x:d.temps,y:d.integ,type:'scatter',mode:'markers+lines',
                        marker:{color:d.colors,size:11,line:{color:'rgba(0,0,0,0.3)',width:1}},
                        line:{color:'#ef4444',width:2.5},
                        hovertemplate:'T=%{x} K<br>I=%{y:.4e}<extra></extra>'};
                    var layout = {paper_bgcolor:'white',plot_bgcolor:'white',
                        {% if op_int_auto %}
                        xaxis:{title:'Temperature (K)',showgrid:true,gridcolor:'#eee',zeroline:false,autorange:true},
                        yaxis:{title:'Integrated Intensity (arb.)',showgrid:true,gridcolor:'#eee',zeroline:false,autorange:true},
                        {% else %}
                        xaxis:{title:'Temperature (K)',showgrid:true,gridcolor:'#eee',zeroline:false,range:[{{ op_int_tmin }},{{ op_int_tmax }}]},
                        yaxis:{title:'Integrated Intensity (arb.)',showgrid:true,gridcolor:'#eee',zeroline:false,range:[{{ op_int_ymin }},{{ op_int_ymax }}]},
                        {% endif %}
                        margin:{t:30,r:40,b:65,l:90},font:{family:'Inter,sans-serif',size:12,color:'#222'}};
                    Plotly.newPlot('op-plotly-integ',[trace],layout,
                        {responsive:true,displaylogo:false,scrollZoom:true,
                         toImageButtonOptions:{format:'png',filename:'integ_vs_T',scale:2}});
                })();</script>
                {% if op_integ_csv %}
                <div style="margin-top:12px;"><form method="POST" action="/slices/orderpar/csv">
                    <input type="hidden" name="csv_data" value="{{ op_integ_csv }}">
                    <button type="submit" style="background:linear-gradient(135deg,#b91c1c,#ef4444);color:#fff;border:none;border-radius:4px;padding:7px 18px;font-size:11px;font-weight:600;cursor:pointer;">&#11015; Download Integrated Intensity CSV</button>
                </form></div>
                {% endif %}
            </div>
            {% endif %}
            {# ── 1D linecut chart ── #}
            {% if op_traces_json %}
            <div class="card" style="border-color:rgba(59,130,246,0.4);margin-top:14px;">
                <div class="card-header"><div class="card-dot" style="background:#3b82f6;box-shadow:0 0 10px #3b82f6;"></div>
                    <span class="card-title" style="color:#3b82f6;">&#128208; 1D Linecuts — All Temperatures</span>
                    <span class="slice-timing" style="margin-left:auto;">{{ op_ntemps }} temperatures</span>
                </div>
                <div id="op-plotly-linecuts" style="width:100%;height:460px;"></div>
                <script>(function(){
                    var traces = {{ op_traces_json | safe }};
                    var layout = {
                        paper_bgcolor:'white', plot_bgcolor:'white',
                        xaxis:{ title:'Q{{ op_cut_axis|lower }} (r.l.u.)', showgrid:true, gridcolor:'#eee', zeroline:false },
                        {% if op_lc_auto %}
                        yaxis:{ title:'Intensity', showgrid:true, gridcolor:'#eee', zeroline:false, autorange:true },
                        {% else %}
                        yaxis:{ title:'Intensity', showgrid:true, gridcolor:'#eee', zeroline:false, range:[{{ op_lc_vmin }},{{ op_lc_vmax }}] },
                        {% endif %}
                        legend:{ orientation:'v', x:1.02, y:1, xanchor:'left' },
                        margin:{t:30,r:180,b:65,l:80},
                        font:{family:'Inter,sans-serif',size:12,color:'#222'},
                        hovermode:'x unified'
                    };
                    Plotly.newPlot('op-plotly-linecuts',traces,layout,
                        {responsive:true,displaylogo:false,scrollZoom:true,
                         toImageButtonOptions:{format:'png',filename:'order_par_linecuts',scale:2}});
                })();</script>
            </div>
            {% endif %}
            {# ── T vs distance heatmap ── #}
            {% if op_heatmap_json %}
            <div class="card" style="border-color:rgba(167,139,250,0.45);margin-top:14px;">
                <div class="card-header"><div class="card-dot" style="background:#a78bfa;box-shadow:0 0 10px #a78bfa;"></div>
                    <span class="card-title" style="color:#a78bfa;">&#127782; Temperature vs Distance Heatmap</span></div>
                <div id="op-plotly-heatmap" style="width:100%;height:460px;"></div>
                <script>(function(){
                    var hd = {{ op_heatmap_json | safe }};
                    var trace = { z:hd.z, x:hd.x, y:hd.y, type:'heatmap', colorscale:'Viridis',
                        colorbar:{title:{text:'Intensity',side:'right'},thickness:18,len:0.9},
                        hovertemplate:'Q={{ op_cut_axis|lower }}=%{x:.4f}<br>T=%{y} K<br>I=%{z:.4e}<extra></extra>',
                        zsmooth:'best',
                        {% if op_hm_auto %}zauto:true{% else %}zauto:false,zmin:{{ op_hm_vmin }},zmax:{{ op_hm_vmax }}{% endif %}
                    };
                    var layout = { paper_bgcolor:'white',plot_bgcolor:'white',
                        xaxis:{title:'Q{{ op_cut_axis|lower }} (r.l.u.)',showgrid:true,gridcolor:'#eee',zeroline:false},
                        yaxis:{title:'Temperature (K)',showgrid:true,gridcolor:'#eee',zeroline:false,
                               tickvals:hd.y, ticktext:hd.y.map(function(v){return v+' K';})},
                        margin:{t:30,r:120,b:65,l:80}, font:{family:'Inter,sans-serif',size:12,color:'#222'}};
                    Plotly.newPlot('op-plotly-heatmap',[trace],layout,
                        {responsive:true,displaylogo:false,scrollZoom:true,
                         toImageButtonOptions:{format:'png',filename:'order_par_heatmap',scale:2}});
                })();</script>
            </div>
            {% endif %}
            {# ── peak fit + order parameter ── #}
            {% if op_fit_json %}
            <div class="card" style="border-color:rgba(251,146,60,0.45);margin-top:14px;">
                <div class="card-header"><div class="card-dot" style="background:#fb923c;box-shadow:0 0 10px #fb923c;"></div>
                    <span class="card-title" style="color:#fb923c;">&#128200; Peak Height vs Temperature (Order Parameter)</span>
                    <span class="slice-timing" style="margin-left:auto;">Fit range: {{ "%.3f"|format(op_fit_qmin) }} – {{ "%.3f"|format(op_fit_qmax) }} r.l.u.</span></div>
                <div id="op-plotly-orderpar" style="width:100%;height:380px;"></div>
                <script>(function(){
                    var fd = {{ op_fit_json | safe }};
                    var traces = [{
                        x:fd.temps,y:fd.heights,type:'scatter',mode:'markers+lines',name:'Peak Height',
                        marker:{color:fd.colors,size:10,line:{color:'rgba(0,0,0,0.25)',width:1}},
                        line:{color:'#ef4444',width:2},
                        hovertemplate:'T=%{x} K<br>Height=%{y:.4e}<extra></extra>'
                    }];
                    if(fd.amplitudes){traces.push({
                        x:fd.temps,y:fd.amplitudes,type:'scatter',mode:'markers+lines',name:'Amplitude (area)',
                        marker:{color:fd.colors,size:7,symbol:'diamond'},
                        line:{color:'#fb923c',width:1.5,dash:'dot'},yaxis:'y2',
                        hovertemplate:'T=%{x} K<br>Amp=%{y:.4e}<extra></extra>'});}
                    var layout = { paper_bgcolor:'white',plot_bgcolor:'white',
                        xaxis:{title:'Temperature (K)',showgrid:true,gridcolor:'#eee',zeroline:false},
                        yaxis:{title:'Peak Height (arb.)',showgrid:true,gridcolor:'#eee',zeroline:false},
                        yaxis2:{title:'Amplitude (area)',overlaying:'y',side:'right',showgrid:false},
                        legend:{orientation:'h',yanchor:'bottom',y:1.02,xanchor:'left',x:0},
                        margin:{t:40,r:80,b:65,l:85}, font:{family:'Inter,sans-serif',size:12,color:'#222'}};
                    Plotly.newPlot('op-plotly-orderpar',traces,layout,
                        {responsive:true,displaylogo:false,scrollZoom:true,
                         toImageButtonOptions:{format:'png',filename:'order_parameter',scale:2}});
                })();</script>
                {% if op_fit_table %}
                <div style="overflow-x:auto;margin-top:14px;">
                <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace;">
                    <thead><tr style="background:rgba(251,146,60,0.12);border-bottom:2px solid rgba(251,146,60,0.4);">
                        <th style="padding:6px 10px;text-align:left;color:#fb923c;">T (K)</th>
                        <th style="padding:6px 10px;text-align:right;">Height</th>
                        <th style="padding:6px 10px;text-align:right;">Amplitude</th>
                        <th style="padding:6px 10px;text-align:right;">Center</th>
                        <th style="padding:6px 10px;text-align:right;">FWHM</th>
                        <th style="padding:6px 10px;text-align:right;">BG slope</th>
                        <th style="padding:6px 10px;text-align:right;">&#967;&#178;&#8320;</th>
                        <th style="padding:6px 10px;text-align:left;">Status</th>
                    </tr></thead>
                    <tbody>{% for r in op_fit_table %}
                    <tr style="border-bottom:1px solid var(--border-subtle);{% if loop.index is odd %}background:rgba(0,0,0,0.015);{% endif %}">
                        <td style="padding:5px 10px;color:#ef4444;font-weight:600;">{{ r.T }}</td>
                        <td style="padding:5px 10px;text-align:right;">{{ "%.4e"|format(r.height) if r.height is not none else "&mdash;"|safe }}</td>
                        <td style="padding:5px 10px;text-align:right;">{{ "%.4e"|format(r.amplitude) if r.amplitude is not none else "&mdash;"|safe }}</td>
                        <td style="padding:5px 10px;text-align:right;">{{ "%.5f"|format(r.center) if r.center is not none else "&mdash;"|safe }}</td>
                        <td style="padding:5px 10px;text-align:right;">{{ "%.5f"|format(r.fwhm) if r.fwhm is not none else "&mdash;"|safe }}</td>
                        <td style="padding:5px 10px;text-align:right;">{{ "%.3e"|format(r.slope) if r.slope is not none else "&mdash;"|safe }}</td>
                        <td style="padding:5px 10px;text-align:right;">{{ "%.4f"|format(r.redchi) if r.redchi is not none else "&mdash;"|safe }}</td>
                        <td style="padding:5px 10px;font-size:11px;color:{% if r.success %}#4ade80{% else %}#f97373{% endif %};">{{ r.status }}</td>
                    </tr>{% endfor %}</tbody>
                </table></div>
                {% endif %}
                {% if op_fit_csv %}
                <div style="margin-top:12px;"><form method="POST" action="/slices/orderpar/csv">
                    <input type="hidden" name="csv_data" value="{{ op_fit_csv }}">
                    <button type="submit" style="background:linear-gradient(135deg,#ea580c,#fb923c);">&#11015; Download Order Parameter CSV</button>
                </form></div>
                {% endif %}
            </div>
            {% endif %}
        </div>
        <div id="tf-shared-controls">
        <hr>
        <div class="inline-fields">
            <div class="field-group"><label class="field-label">Slice Axis</label>
                <select name="slice_axis" onchange="updateSliceLabel(this.value)">
                    <option value="L" {% if slice_axis=="L" %}selected{% endif %}>L — fix L, view H-K</option>
                    <option value="K" {% if slice_axis=="K" %}selected{% endif %}>K — fix K, view H-L</option>
                    <option value="H" {% if slice_axis=="H" %}selected{% endif %}>H — fix H, view K-L</option>
                </select>
            </div>
            <div class="field-group"><label class="field-label"><span id="slice-label">{{ slice_axis }}</span> Values (comma-separated)</label>
                <input type="text" name="Ls" value="{{ L_values }}" placeholder="e.g. 0, 0.5, 1, 1.5" style="width:220px;"></div>
            <div class="field-group"><label class="field-label">Colormap</label>
                <select name="cmap">{% for cm in ["inferno","viridis","plasma","magma","turbo"] %}<option value="{{ cm }}" {% if cm==cmap %}selected{% endif %}>{{ cm }}</option>{% endfor %}</select></div>
            <div class="field-group"><label class="field-label">Skew angle (°) <span style="font-weight:400;text-transform:none;font-size:11px;">(0 = rect, 60 = hex)</span></label>
                <input type="number" name="skew_angle" id="skew-angle-inp" value="{{ skew_angle }}" min="0" max="180" step="1" style="width:90px;"></div>
            <div class="field-group" style="align-self:center;padding-top:18px;">
                <label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;">
                    <input type="checkbox" name="show_grid" value="1" {% if show_grid %}checked{% endif %}> Show grid
                </label>
            </div>
        </div>
        <div class="inline-fields" style="margin-bottom:14px;">
            <div class="field-group"><label class="field-label">vmin (log &gt; 0)</label><input type="number" step="0.0001" name="vmin" value="{{ vmin }}" min="0.0001"></div>
            <div class="field-group"><label class="field-label">vmax</label><input type="number" step="0.0001" name="vmax" value="{{ vmax }}" min="0.0001"></div>
            <div class="field-group" style="align-self:center;padding-top:18px;"><label style="display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer;"><input type="checkbox" name="autoscale" value="1" {% if autoscale %}checked{% endif %}> Auto-scale (shared)</label></div>
        </div>
        <div class="inline-fields" style="margin-bottom:18px;">
            <div class="field-group"><label class="field-label">X-min</label><input type="number" step="0.1" name="xmin" value="{{ xmin }}"></div>
            <div class="field-group"><label class="field-label">X-max</label><input type="number" step="0.1" name="xmax" value="{{ xmax }}"></div>
            <div class="field-group"><label class="field-label">Y-min</label><input type="number" step="0.1" name="ymin" value="{{ ymin }}"></div>
            <div class="field-group"><label class="field-label">Y-max</label><input type="number" step="0.1" name="ymax" value="{{ ymax }}"></div>
        </div>
        {% if vmin_warning %}<div class="alert-warn">⚠ {{ vmin_warning }}</div>{% endif %}
        <button type="submit" class="btn-primary" id="submit-btn">{% if active_tab=='linecut' %}Extract Linecut{% else %}Generate Plots{% endif %}</button>
        <div id="status"></div>
        <div id="progress-container" class="progress-container"><div class="progress-bar"></div></div>
        </div>{# /tf-shared-controls #}
        {% endif %}
    </form>
</div>
<script>
(function(){
    var _sv_orig = window.switchTab;
    window.switchTab = function(name){
        _sv_orig(name);
        // shared controls visibility: hide for thinfilm + orderpar (they have own submit)
        var _hideShared = (name==='thinfilm' || name==='orderpar');
        var sc = document.getElementById('tf-shared-controls');
        if(sc) sc.style.display = _hideShared ? 'none' : '';
        // shared submit button
        var btn = document.getElementById('submit-btn');
        if(btn){
            btn.style.display = _hideShared ? 'none' : '';
            btn.textContent = (name==='linecut') ? 'Extract Linecut' : 'Generate Plots';
        }
        // linecut tab accent
        var tl = document.getElementById('tab-linecut');
        if(tl){ tl.style.color=(name==='linecut')?'#f0b429':''; tl.style.borderBottomColor=(name==='linecut')?'#f0b429':''; }
        // thinfilm tab accent
        var ttf = document.getElementById('tab-thinfilm');
        if(ttf){ ttf.style.color=(name==='thinfilm')?'#a78bfa':''; ttf.style.borderBottomColor=(name==='thinfilm')?'#a78bfa':''; }
        // order parameter tab accent
        var top = document.getElementById('tab-orderpar');
        if(top){ top.style.color=(name==='orderpar')?'#ef4444':''; top.style.borderBottomColor=(name==='orderpar')?'#ef4444':''; }
        // axis labels (linecut)
        var sl = document.getElementById('lc-axis-label');
        var axSel = document.querySelector('select[name="slice_axis"]');
        if(sl && axSel) sl.textContent = axSel.value;
        var slb = document.getElementById('lc-axis-label-b');
        if(slb && axSel) slb.textContent = axSel.value;
        // thin-film axis label
        tfUpdateAxisLabel();
    };
    var axSel = document.querySelector('select[name="slice_axis"]');
    if(axSel) axSel.addEventListener('change', function(){
        var sl=document.getElementById('lc-axis-label'); if(sl) sl.textContent=this.value;
        var slb=document.getElementById('lc-axis-label-b'); if(slb) slb.textContent=this.value;
    });
})();
function tfUpdateAxisLabel(){
    var sel = document.getElementById('tf-slice-axis-sel');
    var lbl = document.getElementById('tf-axis-label');
    if(!sel) return;
    if(lbl) lbl.textContent = sel.value;
    // update X/Y axis name hints in the range row
    var ax = sel.value; // Ql, Qk, or Qh
    var xName = (ax==='Qh') ? "Qk'" : "Qh'";
    var yName = (ax==='Ql') ? "Qk'" : "Ql'";
    document.querySelectorAll('.tf-xax-lbl').forEach(function(s){ s.textContent = xName; });
    document.querySelectorAll('.tf-yax-lbl').forEach(function(s){ s.textContent = yName; });
}
function tfToggleScale(autoscale){
    var pg = document.getElementById('tf-pct-grp');
    var mg = document.getElementById('tf-manual-grp');
    if(pg) pg.style.display = autoscale ? 'flex' : 'none';
    if(mg) mg.style.display = autoscale ? 'none' : 'flex';
}
function tfToggleAxRange(auto){
    var g = document.getElementById('tf-axrange-grp');
    if(g) g.style.display = auto ? 'none' : 'flex';
}
function switchLcMode(val){
    var sr = document.getElementById('lc-single-row');   // File A (single/compare only)
    var cr = document.getElementById('lc-compare-row');  // File B (compare only)
    var mr = document.getElementById('lc-multi-row');    // multi-temp file list
    if(sr) sr.style.display = (val==='multi') ? 'none' : '';
    if(cr) cr.style.display = (val==='compare') ? '' : 'none';
    if(mr) mr.style.display = (val==='multi') ? '' : 'none';
    // Keep lc-shared-row always visible
}
function toggleLcCompare(on){ switchLcMode(on ? 'compare' : 'single'); }

// ── Slice Axis H/K/L → auto-sets X-Axis dropdown and clears Y-Axis ──
function lcSetSliceAxis(letter) {
    var _map = { H: 'Qh', K: 'Qk', L: 'Ql' };
    var xSel = document.getElementById('lc_xaxis_sel');
    var ySel = document.getElementById('lc_yaxis_sel');
    if (xSel) xSel.value = _map[letter] || 'Ql';
    if (ySel) ySel.value = 'None';
}
// ── Sync H/K/L radio buttons when X-Axis dropdown changes ──
function lcSyncAxisBtns() {
    var xSel = document.getElementById('lc_xaxis_sel');
    if (!xSel) return;
    var _rmap = { Qh: 'H', Qk: 'K', Ql: 'L' };
    var letter = _rmap[xSel.value];
    var btns = document.querySelectorAll('input[name="lc_slice_axis_btn"]');
    btns.forEach(function(b) { b.checked = (b.value === letter); });
}
</script>

{% if active_tab == 'linecut' %}
{% if lc_error %}<div class="card"><div class="alert-error">&#9888; {{ lc_error }}</div></div>{% endif %}

{% if lc_multi_traces_json %}
{# ═══ MULTI-TEMPERATURE MODE ═══ #}
<div class="card" style="border-color:rgba(74,222,128,0.45);">
    <div class="card-header">
        <div class="card-dot" style="background:#4ade80;box-shadow:0 0 10px #4ade80;"></div>
        <span class="card-title" style="color:#4ade80;">&#127777; Multi-Temperature Linecuts — {{ lc_xaxis }} (r.l.u.) vs Intensity</span>
        <span class="slice-timing" style="margin-left:auto;">{{ "%.2f"|format(lc_elapsed) }}s &middot; Slice: {{ lc_xaxis }} | ROI: Ql[{{ lc_ql_min }},{{ lc_ql_max }}] Qk[{{ lc_qk_min }},{{ lc_qk_max }}] Qh[{{ lc_qh_min }},{{ lc_qh_max }}]</span>
    </div>
    <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
        {% for item in lc_multi_items %}
        <div style="font-size:12px;padding:3px 10px;background:rgba(74,222,128,0.08);border-radius:4px;border:1px solid rgba(74,222,128,0.3);">
            <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{{ item.color }};margin-right:4px;"></span>
            <b style="color:#4ade80;">{{ item.label }}</b>
        </div>
        {% endfor %}
    </div>
    <div id="lc-plotly-multi" style="width:100%;height:500px;"></div>
    <script>
    (function(){
      var traces = {{ lc_multi_traces_json | safe }};
      {% if lc_autoscale %}
      var yaxis = { title: 'Intensity', autorange: true,
                    showgrid: true, gridcolor: '#ddd', zeroline: false };
      {% else %}
      var yaxis = { title: 'Intensity',
                    range: [{{ "%.6f"|format(lc_vmin_log) }}, {{ "%.6f"|format(lc_vmax_log) }}],
                    showgrid: true, gridcolor: '#ddd', zeroline: false };
      {% endif %}
      var layout = {
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        xaxis: { title: '{{ lc_xaxis }} (r.l.u.)',
                 showgrid: true, gridcolor: '#ddd', zeroline: false, autorange: true },
        yaxis: yaxis,
        legend: { orientation: 'v', x: 1.02, y: 1, xanchor: 'left' },
        margin: { t: 30, r: 200, b: 65, l: 75 },
        font: { family: 'Inter,sans-serif', size: 12, color: '#222' },
        hovermode: 'x unified'
      };
      var cfg = { responsive: true, displaylogo: false, scrollZoom: true,
                  displayModeBar: true,
                  modeBarButtonsToAdd: ['toggleSpikelines'],
                  toImageButtonOptions: { format: 'png', filename: 'linecut_multi_temp', scale: 2 } };
      Plotly.newPlot('lc-plotly-multi', traces, layout, cfg);
    })();
    </script>
</div>
{% if lc_show_overlays and lc_multi_overlays %}
<div class="card" style="border-color:rgba(74,222,128,0.25);">
    <div class="card-header">
        <div class="card-dot" style="background:#4ade80;box-shadow:none;"></div>
        <span class="card-title">&#128444; 2D Projection Overlays — per Temperature</span>
        <span class="slice-timing" style="margin-left:auto;font-size:11px;color:var(--text-muted);">Sum over {{ lc_xaxis }} axis | ROI: Ql[{{ lc_ql_min }},{{ lc_ql_max }}] Qk[{{ lc_qk_min }},{{ lc_qk_max }}] Qh[{{ lc_qh_min }},{{ lc_qh_max }}]</span>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:16px;">
        {% for item in lc_multi_overlays %}
        <div style="flex:0 0 auto;min-width:280px;max-width:420px;background:rgba(0,0,0,0.02);border:1px solid rgba(255,255,255,0.08);border-radius:8px;padding:8px;">
            <div style="font-size:11px;font-weight:600;color:{{ item.color }};margin-bottom:6px;display:flex;align-items:center;gap:6px;">
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{{ item.color }};box-shadow:0 0 5px {{ item.color }};"></span>
                {{ item.label }}
            </div>
            {% if item.proj_json %}
            <div id="ov-plot-{{ loop.index }}" style="width:100%;height:240px;"></div>
            <script>
            (function(){
              var _pd = {{ item.proj_json | safe }};
              var _t = { z: _pd.z, x: _pd.x, y: _pd.y, type: 'heatmap',
                         colorscale: 'Viridis', showscale: false,
                         hovertemplate: _pd.xaxis+'=%{x:.3f}<br>'+_pd.yaxis+'=%{y:.3f}<br>I=%{z:.3e}<extra></extra>' };
              var _l = { paper_bgcolor:'white', plot_bgcolor:'white',
                         xaxis:{ title:{ text:_pd.xaxis+' (r.l.u.)', font:{size:10} }, tickfont:{size:9} },
                         yaxis:{ title:{ text:_pd.yaxis+' (r.l.u.)', font:{size:10} }, tickfont:{size:9} },
                         margin:{ t:10, r:10, b:45, l:55 },
                         font:{ family:'Inter,sans-serif', size:10 } };
              Plotly.newPlot('ov-plot-{{ loop.index }}', [_t], _l,
                { responsive:true, displaylogo:false, scrollZoom:true,
                  modeBarButtonsToRemove:['toImage','sendDataToCloud'] });
            })();
            </script>
            {% else %}
            <div style="font-size:11px;color:#f97373;padding:8px;">{{ item.label }}</div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
</div>
{% endif %}
{% if lc_multi_csv %}
<div class="card" style="padding:12px 18px;">
    <form method="POST" action="/slices/linecut/csv">
        <input type="hidden" name="csv_data" value="{{ lc_multi_csv }}">
        <input type="hidden" name="lc_axis"  value="{{ slice_axis }}">
        <input type="hidden" name="lc_val"   value="{{ lc_val_actual }}">
        <button type="submit">&#11015; Download Multi-Temperature Linecut CSV</button>
    </form>
</div>
{% endif %}

{# ══ T vs Q HEATMAP ══ #}
{% if lc_heatmap_json %}
<div class="card" style="border-color:rgba(167,139,250,0.45);">
    <div class="card-header">
        <div class="card-dot" style="background:#a78bfa;box-shadow:0 0 10px #a78bfa;"></div>
        <span class="card-title" style="color:#a78bfa;">&#127782; Temperature vs Q Heatmap — Intensity Map</span>
        <span class="slice-timing" style="margin-left:auto;">{{ lc_heatmap_ntemps }} temperatures · {{ slice_axis }} = {{ "%.3f"|format(lc_val_actual) }}</span>
    </div>
    <div id="lc-plotly-heatmap" style="width:100%;height:480px;"></div>
    <script>
    (function(){
      var hdata = {{ lc_heatmap_json | safe }};
      var trace = {
        z: hdata.z, x: hdata.x, y: hdata.y,
        type: 'heatmap',
        colorscale: 'Viridis',
        colorbar: { title: { text: 'Intensity', side: 'right' }, thickness: 18, len: 0.9 },
        hovertemplate: 'Q=%{x:.4f}<br>T=%{y} K<br>I=%{z:.4e}<extra></extra>',
        zsmooth: 'best',
        {% if hm_autoscale %}
        zauto: true
        {% else %}
        zauto: false, zmin: {{ hm_vmin }}, zmax: {{ hm_vmax }}
        {% endif %}
      };
      var layout = {
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        xaxis: { title: 'Distance along cut (r.l.u.) \u2014 [{{ lc_n1 }},{{ lc_n2 }}]',
                 showgrid: true, gridcolor: '#eee', zeroline: false, autorange: true },
        yaxis: { title: 'Temperature (K)', showgrid: true, gridcolor: '#eee',
                 zeroline: false, autorange: true, dtick: 1,
                 tickvals: hdata.y,
                 ticktext: hdata.y.map(function(v){ return v + ' K'; }) },
        margin: { t: 30, r: 120, b: 65, l: 80 },
        font: { family: 'Inter,sans-serif', size: 12, color: '#222' }
      };
      Plotly.newPlot('lc-plotly-heatmap', [trace], layout,
        { responsive: true, displaylogo: false, scrollZoom: true,
          toImageButtonOptions: { format: 'png', filename: 'temp_vs_Q_heatmap', scale: 2 } });
    })();
    </script>
</div>
{% endif %}

{# ══ PEAK FIT: Height vs Temperature ══ #}
{% if lc_fit_json %}
<div class="card" style="border-color:rgba(251,146,60,0.45);">
    <div class="card-header">
        <div class="card-dot" style="background:#fb923c;box-shadow:0 0 10px #fb923c;"></div>
        <span class="card-title" style="color:#fb923c;">&#128200; Peak Height vs Temperature &mdash; Gaussian + Linear Fit</span>
        <span class="slice-timing" style="margin-left:auto;">
            Fit range: {{ "%.3f"|format(lc_fit_qmin) }} &ndash; {{ "%.3f"|format(lc_fit_qmax) }} r.l.u.
            &nbsp;&middot;&nbsp; center init: {{ "%.3f"|format(lc_fit_center) }}
        </span>
    </div>
    <div id="lc-plotly-peakfit" style="width:100%;height:380px;"></div>
    <script>
    (function(){
      var fd = {{ lc_fit_json | safe }};
      var trace = {
        x: fd.temps, y: fd.heights,
        type: 'scatter', mode: 'markers+lines',
        marker: { color: fd.colors, size: 10,
                  line: { color: 'rgba(0,0,0,0.25)', width: 1 } },
        line: { color: '#fb923c', width: 1.5, dash: 'dot' },
        hovertemplate: 'T=%{x} K<br>Height=%{y:.4e}<extra></extra>',
        name: 'Peak Height'
      };
      var layout = {
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        xaxis: { title: 'Temperature (K)', showgrid: true, gridcolor: '#eee', zeroline: false },
        yaxis: { title: 'Peak Height (arb. units)', showgrid: true, gridcolor: '#eee', zeroline: false },
        margin: { t: 30, r: 30, b: 65, l: 90 },
        font: { family: 'Inter,sans-serif', size: 12, color: '#222' }
      };
      Plotly.newPlot('lc-plotly-peakfit', [trace], layout,
        { responsive: true, displaylogo: false, scrollZoom: true,
          toImageButtonOptions: { format: 'png', filename: 'peak_height_vs_T', scale: 2 } });
    })();
    </script>
    {% if lc_fit_table %}
    <div style="overflow-x:auto;margin-top:16px;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace;">
        <thead>
            <tr style="background:rgba(251,146,60,0.12);border-bottom:2px solid rgba(251,146,60,0.4);">
                <th style="padding:6px 12px;text-align:left;color:#fb923c;">T (K)</th>
                <th style="padding:6px 12px;text-align:right;">Height</th>
                <th style="padding:6px 12px;text-align:right;">Center (r.l.u.)</th>
                <th style="padding:6px 12px;text-align:right;">FWHM</th>
                <th style="padding:6px 12px;text-align:right;">Amplitude</th>
                <th style="padding:6px 12px;text-align:right;">BG slope</th>
                <th style="padding:6px 12px;text-align:right;">&#967;&#178;&#8320;</th>
                <th style="padding:6px 12px;text-align:left;">Status</th>
            </tr>
        </thead>
        <tbody>
            {% for row in lc_fit_table %}
            <tr style="border-bottom:1px solid var(--border-subtle);{% if loop.index is odd %}background:rgba(0,0,0,0.015);{% endif %}">
                <td style="padding:5px 12px;color:#fb923c;font-weight:600;">{{ row.T }}</td>
                <td style="padding:5px 12px;text-align:right;">{{ "%.4e"|format(row.height) if row.height is not none else "&mdash;" | safe }}</td>
                <td style="padding:5px 12px;text-align:right;">{{ "%.5f"|format(row.center) if row.center is not none else "&mdash;" | safe }}</td>
                <td style="padding:5px 12px;text-align:right;">{{ "%.5f"|format(row.fwhm) if row.fwhm is not none else "&mdash;" | safe }}</td>
                <td style="padding:5px 12px;text-align:right;">{{ "%.4e"|format(row.amplitude) if row.amplitude is not none else "&mdash;" | safe }}</td>
                <td style="padding:5px 12px;text-align:right;">{{ "%.3e"|format(row.slope) if row.slope is not none else "&mdash;" | safe }}</td>
                <td style="padding:5px 12px;text-align:right;">{{ "%.4f"|format(row.redchi) if row.redchi is not none else "&mdash;" | safe }}</td>
                <td style="padding:5px 12px;font-size:11px;color:{% if row.success %}#4ade80{% else %}#f97373{% endif %};">{{ row.status }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    </div>
    {% endif %}
    {% if lc_fit_csv %}
    <div style="margin-top:12px;">
        <form method="POST" action="/slices/linecut/fitcsv">
            <input type="hidden" name="csv_data" value="{{ lc_fit_csv }}">
            <input type="hidden" name="slice_axis" value="{{ slice_axis }}">
            <input type="hidden" name="lc_val_actual" value="{{ lc_val_actual }}">
            <button type="submit" style="background:linear-gradient(135deg,#ea580c,#fb923c);">&#11015; Download Fit Results CSV</button>
        </form>
    </div>
    {% endif %}
</div>
{% endif %}

{# ══ PEAK PARAMETERS vs TEMPERATURE ══ #}
{% if lc_peakplot_json %}
{% set _pp = lc_peakplot_json | tojson %}
<div class="card" style="border-color:rgba(56,189,248,0.45);margin-top:10px;">
    <div class="card-header">
        <div class="card-dot" style="background:#38bdf8;box-shadow:0 0 10px #38bdf8;"></div>
        <span class="card-title" style="color:#38bdf8;">&#128202; Peak Parameters vs Temperature</span>
        <span style="margin-left:auto;font-size:11px;color:var(--text-muted);font-family:'IBM Plex Mono',monospace;">
            Slice axis: <b>{{ lc_xaxis }}</b> &nbsp;|&nbsp;
            Ql [{{ lc_ql_min }}, {{ lc_ql_max }}] &nbsp;
            Qk [{{ lc_qk_min }}, {{ lc_qk_max }}] &nbsp;
            Qh [{{ lc_qh_min }}, {{ lc_qh_max }}]
        </span>
    </div>
    {# Sub-panel selector #}
    <div style="display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap;">
        <button id="pp-btn-h"  onclick="ppSwitch('height')"
                style="font-size:12px;padding:4px 14px;cursor:pointer;border-radius:5px;
                       background:rgba(56,189,248,0.2);border:1px solid rgba(56,189,248,0.5);
                       color:#38bdf8;font-weight:600;">
            &#128200; Peak Height
        </button>
        <button id="pp-btn-c"  onclick="ppSwitch('center')"
                style="font-size:12px;padding:4px 14px;cursor:pointer;border-radius:5px;
                       background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);
                       color:var(--text-muted);">
            &#127919; Peak Center
        </button>
        <button id="pp-btn-fw" onclick="ppSwitch('fwhm')"
                style="font-size:12px;padding:4px 14px;cursor:pointer;border-radius:5px;
                       background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);
                       color:var(--text-muted);">
            &#8596; FWHM
        </button>
        <button id="pp-btn-all" onclick="ppSwitch('all')"
                style="font-size:12px;padding:4px 14px;cursor:pointer;border-radius:5px;
                       background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);
                       color:var(--text-muted);">
            &#9783; All Three
        </button>
        <span style="margin-left:auto;font-size:11px;color:var(--text-muted);align-self:center;">
            Fit ROI: {{ "%.3f"|format(lc_fit_qmin) }} &ndash; {{ "%.3f"|format(lc_fit_qmax) }} r.l.u.
        </span>
    </div>

    {# Single-panel view #}
    <div id="pp-single" style="width:100%;height:380px;"></div>
    {# All-three stacked view #}
    <div id="pp-all" style="display:none;">
        <div style="font-size:11px;color:#38bdf8;font-weight:600;margin:4px 0 2px;font-family:'IBM Plex Mono',monospace;">Peak Height</div>
        <div id="pp-all-h"  style="width:100%;height:300px;"></div>
        <div style="font-size:11px;color:#38bdf8;font-weight:600;margin:10px 0 2px;font-family:'IBM Plex Mono',monospace;">Peak Center (r.l.u.)</div>
        <div id="pp-all-c"  style="width:100%;height:280px;"></div>
        <div style="font-size:11px;color:#38bdf8;font-weight:600;margin:10px 0 2px;font-family:'IBM Plex Mono',monospace;">FWHM (r.l.u.)</div>
        <div id="pp-all-fw" style="width:100%;height:280px;"></div>
    </div>

    <script>
    (function(){
        var _pp = {{ lc_peakplot_json | safe }};
        var _cfg = { responsive:true, displaylogo:false, scrollZoom:true,
                     displayModeBar:'hover', uirevision:'pp-fixed',
                     toImageButtonOptions:{ format:'png', scale:2 } };

        function _ppLayout(ytitle, yformat) {
            return {
                paper_bgcolor:'white', plot_bgcolor:'white',
                xaxis:{ title:'Temperature (K)', showgrid:true, gridcolor:'#eee',
                        zeroline:false, dtick:1, tickvals:_pp.T,
                        ticktext:_pp.T.map(function(v){ return v+' K'; }) },
                yaxis:{ title:ytitle, showgrid:true, gridcolor:'#eee', zeroline:false,
                        tickformat: yformat || '' },
                margin:{ t:30, r:30, b:70, l:90 },
                font:{ family:'Inter,sans-serif', size:12, color:'#222' },
                uirevision:'pp-fixed'
            };
        }

        function _ppTrace(yvals, name, mode) {
            return {
                x: _pp.T, y: yvals,
                type: 'scatter',
                mode: mode || 'markers+lines',
                marker:{ color:_pp.colors, size:10,
                         line:{ color:'rgba(0,0,0,0.2)', width:1 } },
                line:{ color:'#38bdf8', width:1.5, dash:'dot' },
                hovertemplate: 'T=%{x} K<br>'+name+'=%{y:.5g}<extra></extra>',
                name: name
            };
        }

        var _ppCurrent = 'height';
        var _ppAllInit = false;

        function ppDraw(divId, yvals, ytitle, yformat) {
            Plotly.react(divId, [_ppTrace(yvals, ytitle)], _ppLayout(ytitle, yformat), _cfg);
        }

        window['ppSwitch'] = function(key) {
            _ppCurrent = key;
            var btns = { height:'pp-btn-h', center:'pp-btn-c', fwhm:'pp-btn-fw', all:'pp-btn-all' };
            Object.keys(btns).forEach(function(k) {
                var b = document.getElementById(btns[k]);
                if (k === key) {
                    b.style.background = 'rgba(56,189,248,0.2)';
                    b.style.border     = '1px solid rgba(56,189,248,0.5)';
                    b.style.color      = '#38bdf8';
                    b.style.fontWeight = '600';
                } else {
                    b.style.background = 'rgba(255,255,255,0.04)';
                    b.style.border     = '1px solid rgba(255,255,255,0.15)';
                    b.style.color      = 'var(--text-muted)';
                    b.style.fontWeight = 'normal';
                }
            });
            var single = document.getElementById('pp-single');
            var all    = document.getElementById('pp-all');
            if (key === 'all') {
                single.style.display = 'none';
                all.style.display    = '';
                if (!_ppAllInit) {
                    _ppAllInit = true;
                    ppDraw('pp-all-h',  _pp.height, 'Peak Height (arb. units)', '.3e');
                    ppDraw('pp-all-c',  _pp.center, 'Peak Center (r.l.u.)');
                    ppDraw('pp-all-fw', _pp.fwhm,   'FWHM (r.l.u.)');
                } else {
                    Plotly.Plots.resize('pp-all-h');
                    Plotly.Plots.resize('pp-all-c');
                    Plotly.Plots.resize('pp-all-fw');
                }
            } else {
                single.style.display = '';
                all.style.display    = 'none';
                var ymap = { height:[_pp.height,'Peak Height (arb. units)','.3e'],
                             center:[_pp.center,'Peak Center (r.l.u.)',''],
                             fwhm:  [_pp.fwhm,  'FWHM (r.l.u.)',''] };
                var args = ymap[key];
                ppDraw('pp-single', args[0], args[1], args[2]);
            }
        };

        // Initial render: Peak Height
        ppSwitch('height');
    })();
    </script>
</div>
{% endif %}

{% elif lc_overlay and lc_compare and lc_overlay not in ('PROJ1D','PROJ2D') %}
{# ═══ LEGACY COMPARE MODE (pixel-based linecut): combined 1-D chart ═══ #}
<div class="card" style="border-color:rgba(240,180,41,0.45);">
    <div class="card-header">
        <div class="card-dot" style="background:#f0b429;box-shadow:0 0 10px #f0b429;"></div>
        <span class="card-title" style="color:#f0b429;">&#128293; Temperature Comparison — 1D Intensity vs Distance</span>
        <span class="slice-timing" style="margin-left:auto;">{{ "%.2f"|format(lc_elapsed) }}s &middot; {{ lc_npts }} pts</span>
    </div>
    <div style="display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap;">
        <div style="font-size:12px;color:#1565c0;padding:4px 10px;background:rgba(21,101,192,0.08);border-radius:4px;border:1px solid rgba(21,101,192,0.3);">
            &#9632; A &nbsp;·&nbsp; <b>{{ lc_file_label_a }}</b> &nbsp;·&nbsp; {{ slice_axis }} = {{ "%.3f"|format(lc_val_actual) }}
        </div>
        <div style="font-size:12px;color:#c62828;padding:4px 10px;background:rgba(198,40,40,0.08);border-radius:4px;border:1px solid rgba(198,40,40,0.3);">
            &#9632; B &nbsp;·&nbsp; <b>{{ lc_file_label_b }}</b> &nbsp;·&nbsp; {{ slice_axis }} = {{ "%.3f"|format(lc_val_actual_b) }}
        </div>
    </div>
    <div id="lc-plotly-compare" style="width:100%;height:440px;"></div>
    <script>
    (function(){
      var distA = {{ lc_plotly_dist_a | safe }};
      var profA = {{ lc_plotly_prof_a | safe }};
      var distB = {{ lc_plotly_dist_b | safe }};
      var profB = {{ lc_plotly_prof_b | safe }};
      var traces = [
        { x: distA, y: profA, type: 'scatter', mode: 'lines',
          name: 'Temp A \u00b7 {{ lc_file_label_a }} ({{ slice_axis }}={{ "%.3f"|format(lc_val_actual) }})',
          line: { color: '#1565c0', width: 2.5 } },
        { x: distB, y: profB, type: 'scatter', mode: 'lines',
          name: 'Temp B \u00b7 {{ lc_file_label_b }} ({{ slice_axis }}={{ "%.3f"|format(lc_val_actual_b) }})',
          line: { color: '#c62828', width: 2.5 } }
      ];
      {% if lc_autoscale %}
      var yaxis = { title: 'Intensity (log)', type: 'log', autorange: true,
                    showgrid: true, gridcolor: '#ddd', zeroline: false };
      {% else %}
      var yaxis = { title: 'Intensity (log)', type: 'log',
                    range: [{{ "%.6f"|format(lc_vmin_log) }}, {{ "%.6f"|format(lc_vmax_log) }}],
                    showgrid: true, gridcolor: '#ddd', zeroline: false };
      {% endif %}
      var layout = {
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        xaxis: { title: 'Distance along cut (r.l.u.) \u2014 [{{ lc_n1 }},{{ lc_n2 }}]',
                 showgrid: true, gridcolor: '#ddd', zeroline: false, autorange: true },
        yaxis: yaxis,
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'left', x: 0 },
        margin: { t: 65, r: 30, b: 65, l: 75 },
        font: { family: 'Inter,sans-serif', size: 12, color: '#222' },
        hovermode: 'x unified'
      };
      var cfg = { responsive: true, displaylogo: false, scrollZoom: true,
                  displayModeBar: true,
                  modeBarButtonsToAdd: ['toggleSpikelines'],
                  toImageButtonOptions: { format: 'png', filename: 'linecut_compare', scale: 2 } };
      Plotly.newPlot('lc-plotly-compare', traces, layout, cfg);
    })();
    </script>
</div>
{# Two slice overlays side-by-side #}
{% if lc_overlay %}
<div class="card" style="border-color:rgba(100,100,120,0.4);">
    <div class="card-header">
        <div class="card-dot" style="background:#888;box-shadow:none;"></div>
        <span class="card-title">Slice Overlays</span>
        <span class="slice-timing" style="margin-left:auto;">Same linecut shown on each slice</span>
    </div>
    <div class="two-col" style="align-items:flex-start;">
        <div>
            <p style="font-size:11px;color:#00c8ff;font-weight:600;margin:0 0 4px;">A &nbsp;&middot;&nbsp; {{ lc_file_label_a }} &nbsp;·&nbsp; {{ slice_axis }}={{ "%.3f"|format(lc_val_actual) }}</p>
            <img src="data:image/png;base64,{{ lc_overlay }}" style="max-width:100%;">
        </div>
        <div>
            <p style="font-size:11px;color:#f0b429;font-weight:600;margin:0 0 4px;">B &nbsp;&middot;&nbsp; {{ lc_file_label_b }} &nbsp;·&nbsp; {{ slice_axis }}={{ "%.3f"|format(lc_val_actual_b) }}</p>
            <img src="data:image/png;base64,{{ lc_overlay_b }}" style="max-width:100%;">
        </div>
    </div>
    {% if lc_csv_data %}
    <hr style="margin:14px 0 10px;">
    <form method="POST" action="/slices/linecut/csv">
        <input type="hidden" name="csv_data" value="{{ lc_csv_data }}">
        <input type="hidden" name="lc_axis"  value="{{ slice_axis }}">
        <input type="hidden" name="lc_val"   value="{{ lc_val }}">
        <button type="submit">&#11015; Download Linecut CSV (A)</button>
    </form>
    {% endif %}
</div>
{% endif %}

{% elif lc_overlay == 'PROJ2D' and lc_proj_json %}
{# ═══ SINGLE MODE — 2D PROJECTION (heatmap with zoom) ═══ #}
<div class="card" style="border-color:rgba(0,200,255,0.45);">
    <div class="card-header">
        <div class="card-dot" style="background:#00c8ff;box-shadow:0 0 8px #00c8ff;"></div>
        <span class="card-title" style="color:#00c8ff;">2D Projection &mdash; {{ lc_xaxis }} &times; {{ lc_yaxis }}</span>
        <span style="margin-left:auto;font-size:11px;color:var(--text-muted);display:flex;align-items:center;gap:10px;">
            <span>scroll to zoom &nbsp;&#x1F50D; &nbsp;|&nbsp; drag to pan</span>
            <span class="slice-timing">{{ "%.2f"|format(lc_elapsed) }}s</span>
        </span>
    </div>
    {% if lc_proj_info %}<div style="font-size:11px;color:var(--text-muted);font-family:'IBM Plex Mono',monospace;padding:6px 10px;background:rgba(0,0,0,0.05);border-radius:4px;margin-bottom:10px;">{{ lc_proj_info }}</div>{% endif %}
    <!-- Colorscale + zmin/zmax controls -->
    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font-size:12px;">
        <label style="display:flex;align-items:center;gap:6px;color:var(--text-muted);">
            Colorscale:
            <select id="proj2d-cscale" onchange="update2DPlot()" style="font-size:12px;padding:2px 6px;border-radius:4px;">
                <option value="Viridis" selected>Viridis</option>
                <option value="Inferno">Inferno</option>
                <option value="Hot">Hot</option>
                <option value="RdBu">RdBu</option>
                <option value="Jet">Jet</option>
                <option value="Greys">Greys</option>
            </select>
        </label>
        <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
            <input type="checkbox" id="proj2d-autoz" checked onchange="update2DPlot()"> Auto-scale z
        </label>
        <span id="proj2d-zrange-grp" style="display:none;gap:8px;align-items:center;">
            <label style="color:var(--text-muted);">zmin <input type="number" id="proj2d-zmin" step="any" value="0" style="width:90px;font-size:12px;" onchange="update2DPlot()"></label>
            <label style="color:var(--text-muted);">zmax <input type="number" id="proj2d-zmax" step="any" value="1" style="width:90px;font-size:12px;" onchange="update2DPlot()"></label>
        </span>
        <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
            <input type="checkbox" id="proj2d-smooth" checked onchange="update2DPlot()"> Smooth
        </label>
        <button onclick="Plotly.relayout('lc-plotly-proj2d',{'xaxis.autorange':true,'yaxis.autorange':true})"
                style="font-size:11px;padding:3px 10px;cursor:pointer;background:rgba(0,200,255,0.1);border:1px solid rgba(0,200,255,0.4);border-radius:4px;color:#00c8ff;">
            &#8635; Reset Zoom
        </button>
    </div>
    <div id="lc-plotly-proj2d" style="width:100%;height:520px;"></div>
    <script>
    (function(){
        var _proj2d_data = {{ lc_proj_json | safe }};
        function update2DPlot() {
            var cs   = document.getElementById('proj2d-cscale').value;
            var auto = document.getElementById('proj2d-autoz').checked;
            var sm   = document.getElementById('proj2d-smooth').checked;
            var zgrp = document.getElementById('proj2d-zrange-grp');
            zgrp.style.display = auto ? 'none' : 'flex';
            var trace = {
                z: _proj2d_data.z, x: _proj2d_data.x, y: _proj2d_data.y,
                type: 'heatmap', colorscale: cs,
                colorbar: { title: { text: 'Intensity', side: 'right' }, thickness: 18, len: 0.9 },
                hovertemplate: _proj2d_data.xaxis+'=%{x:.4f}<br>'+_proj2d_data.yaxis+'=%{y:.4f}<br>I=%{z:.4e}<extra></extra>',
                zsmooth: sm ? 'best' : false
            };
            if (!auto) {
                trace.zmin = parseFloat(document.getElementById('proj2d-zmin').value) || 0;
                trace.zmax = parseFloat(document.getElementById('proj2d-zmax').value) || 1;
            }
            var layout = {
                paper_bgcolor: 'white', plot_bgcolor: 'white',
                xaxis: { title: _proj2d_data.xaxis+' (r.l.u.)', showgrid:true, gridcolor:'#eee', zeroline:false },
                yaxis: { title: _proj2d_data.yaxis+' (r.l.u.)', showgrid:true, gridcolor:'#eee', zeroline:false },
                dragmode: 'zoom',
                margin: { t:20, r:110, b:65, l:80 },
                font: { family:'Inter,sans-serif', size:12, color:'#222' }
            };
            var cfg = {
                responsive: true, displaylogo: false, scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToAdd: ['toggleSpikelines', 'resetScale2d'],
                toImageButtonOptions: { format:'png', filename:'projection_2d_{{ lc_xaxis }}_{{ lc_yaxis }}', scale:2 }
            };
            Plotly.react('lc-plotly-proj2d', [trace], layout, cfg);
        }
        // Initial render
        update2DPlot();
        // Populate zmin/zmax from data on first load
        var _flat2d = _proj2d_data.z.flat ? _proj2d_data.z.flat() : [].concat.apply([], _proj2d_data.z);
        var _pos2d = _flat2d.filter(function(v){ return v > 0 && isFinite(v); });
        if (_pos2d.length > 0) {
            _pos2d.sort(function(a,b){return a-b;});
            var p2 = _pos2d[Math.floor(_pos2d.length * 0.02)];
            var p98 = _pos2d[Math.floor(_pos2d.length * 0.98)];
            document.getElementById('proj2d-zmin').value = p2.toExponential(3);
            document.getElementById('proj2d-zmax').value = p98.toExponential(3);
        }
    })();
    </script>
</div>

{% elif lc_overlay == 'PROJ1D' %}
{# ═══ SINGLE / COMPARE MODE — 1D PROJECTION ═══ #}
<div class="card" style="border-color:rgba({% if lc_compare %}240,180,41{% else %}0,200,255{% endif %},0.45);">
    <div class="card-header">
        <div class="card-dot" style="background:{% if lc_compare %}#f0b429;box-shadow:0 0 10px #f0b429{% else %}#00c8ff;box-shadow:0 0 8px #00c8ff{% endif %};"></div>
        <span class="card-title" style="color:{% if lc_compare %}#f0b429{% else %}#00c8ff{% endif %};">
            {% if lc_compare %}&#128293; Compare Two — {% endif %}1D Projection: {{ lc_xaxis }}
        </span>
        <span class="slice-timing" style="margin-left:auto;">{{ "%.2f"|format(lc_elapsed) }}s</span>
    </div>
    <!-- File legend chips -->
    <div style="display:flex;gap:10px;margin-bottom:10px;flex-wrap:wrap;">
        <div style="font-size:12px;color:#1565c0;padding:4px 10px;background:rgba(21,101,192,0.08);border-radius:4px;border:1px solid rgba(21,101,192,0.3);">
            &#9632; {% if lc_compare %}A &nbsp;·&nbsp; {% endif %}<b>{{ lc_file_label_a }}</b>
        </div>
        {% if lc_compare %}
        <div style="font-size:12px;color:#c62828;padding:4px 10px;background:rgba(198,40,40,0.08);border-radius:4px;border:1px solid rgba(198,40,40,0.3);">
            &#9632; B &nbsp;·&nbsp; <b>{{ lc_file_label_b }}</b>
        </div>
        {% endif %}
    </div>
    {% if lc_proj_info %}<div style="font-size:11px;color:var(--text-muted);font-family:'IBM Plex Mono',monospace;padding:6px 10px;background:rgba(0,0,0,0.05);border-radius:4px;margin-bottom:10px;">{{ lc_proj_info }}</div>{% endif %}
    <div id="lc-plotly-single" style="width:100%;height:440px;"></div>
    <script>
    (function(){
      var distA = {{ lc_plotly_dist_a | safe }};
      var profA = {{ lc_plotly_prof_a | safe }};
      var mode1d = '{{ "lines" if lc_plot_lines else "markers" }}';
      var traces = [{ x: distA, y: profA, type: 'scatter', mode: mode1d,
        name: '{{ lc_file_label_a }}',
        line: { color: '#1565c0', width: 2 },
        marker: { color: '#1565c0', size: 4 } }];
      {% if lc_compare %}
      var distB = {{ lc_plotly_dist_b | safe }};
      var profB = {{ lc_plotly_prof_b | safe }};
      if (distB && distB.length > 0) {
        traces.push({ x: distB, y: profB, type: 'scatter', mode: mode1d,
          name: '{{ lc_file_label_b }}',
          line: { color: '#c62828', width: 2 },
          marker: { color: '#c62828', size: 4 } });
      }
      {% endif %}
      {% if lc_autoscale %}
      var yaxis = { title: 'Intensity', showgrid: true, gridcolor: '#ddd', zeroline: false, autorange: true };
      {% else %}
      var yaxis = { title: 'Intensity',
                    range: [{{ "%.6f"|format(lc_vmin_log) }}, {{ "%.6f"|format(lc_vmax_log) }}],
                    showgrid: true, gridcolor: '#ddd', zeroline: false };
      {% endif %}
      var layout = {
        paper_bgcolor: 'white', plot_bgcolor: 'white',
        xaxis: { title: '{{ lc_xaxis }} (r.l.u.)',
                 showgrid: true, gridcolor: '#ddd', zeroline: false, autorange: true },
        yaxis: yaxis,
        legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'left', x: 0 },
        margin: { t: {% if lc_compare %}60{% else %}30{% endif %}, r: 20, b: 60, l: 70 },
        font: { family: 'Inter,sans-serif', size: 12, color: '#222' },
        hovermode: 'x unified',
        showlegend: {{ 'true' if lc_compare else 'false' }}
      };
      var cfg = { responsive: true, displaylogo: false, scrollZoom: true,
                  displayModeBar: true,
                  modeBarButtonsToAdd: ['toggleSpikelines'],
                  toImageButtonOptions: {
                    format: 'png',
                    filename: '{% if lc_compare %}compare_{% endif %}projection_{{ lc_xaxis }}',
                    scale: 2 } };
      Plotly.newPlot('lc-plotly-single', traces, layout, cfg);
    })();
    </script>
    {% if lc_csv_data %}
    <hr style="margin:14px 0 10px;">
    <form method="POST" action="/slices/linecut/csv">
        <input type="hidden" name="csv_data" value="{{ lc_csv_data }}">
        <input type="hidden" name="lc_axis"  value="{{ lc_xaxis }}">
        <input type="hidden" name="lc_val"   value="0">
        <button type="submit">&#11015; Download Projection CSV{% if lc_compare %} (A){% endif %}</button>
    </form>
    {% endif %}
</div>
{% endif %}
{% else %}
{% if rows %}
<div class="meta-chips">
    <div class="meta-chip">⏱ <b>{{ "%.2f"|format(total_time) }}s</b></div>
    <div class="meta-chip">Mode: <b>{% if active_tab=="compare" %}Compare Two{% elif active_tab=="multi" %}Compare Multiple{% else %}Single{% endif %}</b></div>
    <div class="meta-chip">vmin/vmax: <b>{{ "%.4g"|format(used_vmin) }} / {{ "%.4g"|format(used_vmax) }}</b></div>
    <div class="meta-chip">{{ "auto-scale" if autoscale else "manual" }}</div>
    <div class="meta-chip">{{ slice_axis }}-slice</div>
</div>

{% if rows %}
<div class="slice-grid" style="margin-top:4px;">
{% for row in rows %}
    {% if row.error %}
    <div class="slice-box"><div class="alert-error" style="font-size:12px;">⚠ {{ slice_axis }}={{ row.L }}: {{ row.error }}</div></div>
    {% elif active_tab == "single" %}
        {% if row.a %}
        <div class="slice-box" style="flex:1 1 100%;max-width:100%;width:100%;">
            {# ── Header row: title + view-toggle buttons ── #}
            <div class="slice-filename" style="margin-bottom:8px;flex-wrap:wrap;gap:6px;">
                <div class="slice-dot"></div>
                <span>{{ slice_axis }} = {{ "%.3f"|format(row.a.L_actual) }}</span>
                <span style="font-size:11px;color:var(--text-muted);">{{ row.a.filename }}</span>
                <span class="slice-timing">{{ row.a.source }} · {{ "%.2f"|format(row.a.time) }}s</span>
                <div style="margin-left:auto;display:flex;gap:4px;">
                    <button id="sv-btn-png-{{ loop.index }}"
                            onclick="svShowView('{{ loop.index }}','png')"
                            style="font-size:11px;padding:3px 10px;cursor:pointer;border-radius:4px;
                                   background:rgba(0,200,255,0.2);border:1px solid rgba(0,200,255,0.5);color:#00c8ff;font-weight:600;">
                        &#128247; PNG View
                    </button>
                    <button id="sv-btn-zoom-{{ loop.index }}"
                            onclick="svShowView('{{ loop.index }}','zoom')"
                            style="font-size:11px;padding:3px 10px;cursor:pointer;border-radius:4px;
                                   background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);color:var(--text-muted);">
                        &#128269; Interactive Zoom
                    </button>
                </div>
            </div>

            {# ── PNG VIEW panel ── #}
            <div id="sv-pnlpng-{{ loop.index }}">
                <img class="slice-img" src="data:image/png;base64,{{ row.a.png }}"
                     style="width:100%;max-width:520px;display:block;border:1px solid var(--border-subtle);border-radius:5px;">
                <a class="dl-btn" href="data:image/png;base64,{{ row.a.png }}"
                   download="{{ row.a.filename.split('/')[-1] }}_{{ slice_axis }}{{ '%.3f'|format(row.a.L_actual) }}.png"
                   style="margin-top:6px;display:inline-block;">&#11015; Download PNG</a>
            </div>

            {# ── INTERACTIVE ZOOM panel (hidden by default) ── #}
            <div id="sv-pnlzoom-{{ loop.index }}" style="display:none;">
                {# Skew info badge — skew_angle comes from the route context #}
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px;">
                    <span style="background:rgba(0,200,255,0.12);border:1px solid rgba(0,200,255,0.35);
                                 border-radius:4px;padding:2px 8px;color:#00c8ff;font-family:'IBM Plex Mono',monospace;">
                        Skew: {{ skew_angle }}°
                        &nbsp;&mdash;&nbsp;{% if skew_angle != 90 %}Cartesian grid (matches PNG){% else %}rectangular{% endif %}
                    </span>
                    <span style="color:var(--text-muted);">scroll to zoom &nbsp;&#128269;&nbsp;|&nbsp; drag to pan</span>
                </div>
                {# Controls #}
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:12px;">
                    <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
                        Colorscale:
                        <select id="sv-cscale-{{ loop.index }}" onchange="svUpdate{{ loop.index }}()"
                                style="font-size:12px;padding:2px 5px;border-radius:4px;">
                            <option value="Viridis" selected>Viridis</option>
                            <option value="Inferno">Inferno</option>
                            <option value="Hot">Hot</option>
                            <option value="RdBu">RdBu</option>
                            <option value="Jet">Jet</option>
                            <option value="Greys">Greys</option>
                        </select>
                    </label>
                    <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
                        <input type="checkbox" id="sv-autoz-{{ loop.index }}" checked onchange="svUpdate{{ loop.index }}()"> Auto z
                    </label>
                    <span id="sv-zrange-{{ loop.index }}" style="display:none;gap:6px;align-items:center;">
                        <label style="color:var(--text-muted);">zmin
                            <input type="number" id="sv-zmin-{{ loop.index }}" step="any" style="width:80px;font-size:12px;"
                                   onchange="svUpdate{{ loop.index }}()">
                        </label>
                        <label style="color:var(--text-muted);">zmax
                            <input type="number" id="sv-zmax-{{ loop.index }}" step="any" style="width:80px;font-size:12px;"
                                   onchange="svUpdate{{ loop.index }}()">
                        </label>
                    </span>
                    <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
                        <input type="checkbox" id="sv-smooth-{{ loop.index }}" onchange="svUpdate{{ loop.index }}()"> Smooth
                        <span style="font-size:10px;opacity:0.7;">(slow)</span>
                    </label>
                    <button onclick="Plotly.relayout('sv-plot-{{ loop.index }}',{'xaxis.autorange':true,'yaxis.autorange':true})"
                            style="font-size:11px;padding:3px 9px;cursor:pointer;background:rgba(0,200,255,0.1);
                                   border:1px solid rgba(0,200,255,0.4);border-radius:4px;color:#00c8ff;">
                        &#8635; Reset Zoom
                    </button>
                    <a href="data:image/png;base64,{{ row.a.png }}"
                       download="{{ row.a.filename.split('/')[-1] }}_{{ slice_axis }}{{ '%.3f'|format(row.a.L_actual) }}.png"
                       style="font-size:11px;padding:3px 9px;text-decoration:none;
                              background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.15);
                              border-radius:4px;color:var(--text-muted);">
                        &#11015; PNG
                    </a>
                </div>
                <div id="sv-plot-{{ loop.index }}" style="width:100%;height:520px;"></div>
            </div>

            <script>
            (function(){
                var _d{{ loop.index }} = {{ row.a.plotly_json | safe }};
                var _plotInit{{ loop.index }} = false;

                window['svShowView{{ loop.index }}'] = function(view) {
                    var pngPnl  = document.getElementById('sv-pnlpng-{{ loop.index }}');
                    var zoomPnl = document.getElementById('sv-pnlzoom-{{ loop.index }}');
                    var btnPng  = document.getElementById('sv-btn-png-{{ loop.index }}');
                    var btnZoom = document.getElementById('sv-btn-zoom-{{ loop.index }}');
                    if (view === 'png') {
                        pngPnl.style.display  = '';
                        zoomPnl.style.display = 'none';
                        btnPng.style.background  = 'rgba(0,200,255,0.2)';
                        btnPng.style.border      = '1px solid rgba(0,200,255,0.5)';
                        btnPng.style.color       = '#00c8ff';
                        btnPng.style.fontWeight  = '600';
                        btnZoom.style.background = 'rgba(255,255,255,0.04)';
                        btnZoom.style.border     = '1px solid rgba(255,255,255,0.15)';
                        btnZoom.style.color      = 'var(--text-muted)';
                        btnZoom.style.fontWeight = 'normal';
                    } else {
                        pngPnl.style.display  = 'none';
                        zoomPnl.style.display = '';
                        btnZoom.style.background = 'rgba(0,200,255,0.2)';
                        btnZoom.style.border     = '1px solid rgba(0,200,255,0.5)';
                        btnZoom.style.color      = '#00c8ff';
                        btnZoom.style.fontWeight = '600';
                        btnPng.style.background  = 'rgba(255,255,255,0.04)';
                        btnPng.style.border      = '1px solid rgba(255,255,255,0.15)';
                        btnPng.style.color       = 'var(--text-muted)';
                        btnPng.style.fontWeight  = 'normal';
                        // lazy-init Plotly on first open
                        if (!_plotInit{{ loop.index }}) {
                            _plotInit{{ loop.index }} = true;
                            window['svUpdate{{ loop.index }}']();
                            document.getElementById('sv-zmin-{{ loop.index }}').value = _d{{ loop.index }}.zmin.toFixed(3);
                            document.getElementById('sv-zmax-{{ loop.index }}').value = _d{{ loop.index }}.zmax.toFixed(3);
                        } else {
                            Plotly.Plots.resize('sv-plot-{{ loop.index }}');
                        }
                    }
                };
                // global alias so inline onclick works
                window['svShowView'] = window['svShowView'] || function(idx, view) {
                    window['svShowView' + idx](view);
                };

                window['svUpdate{{ loop.index }}'] = function() {
                    var cs   = document.getElementById('sv-cscale-{{ loop.index }}').value;
                    var auto = document.getElementById('sv-autoz-{{ loop.index }}').checked;
                    var sm   = document.getElementById('sv-smooth-{{ loop.index }}').checked;
                    var zgrp = document.getElementById('sv-zrange-{{ loop.index }}');
                    zgrp.style.display = auto ? 'none' : 'flex';
                    var trace = {
                        z: _d{{ loop.index }}.z,
                        x: _d{{ loop.index }}.x,
                        y: _d{{ loop.index }}.y,
                        type: 'heatmap',
                        colorscale: cs,
                        colorbar: { title: { text: 'log\u2081\u2080(I)', side:'right' }, thickness:18, len:0.9 },
                        hovertemplate: _d{{ loop.index }}.xaxis+'=%{x:.4f}<br>'+_d{{ loop.index }}.yaxis+'=%{y:.4f}<br>log\u2081\u2080(I)=%{z:.3f}<extra></extra>',
                        zsmooth: sm ? 'fast' : false
                    };
                    trace.zmin = auto ? _d{{ loop.index }}.zmin
                                      : (parseFloat(document.getElementById('sv-zmin-{{ loop.index }}').value) || _d{{ loop.index }}.zmin);
                    trace.zmax = auto ? _d{{ loop.index }}.zmax
                                      : (parseFloat(document.getElementById('sv-zmax-{{ loop.index }}').value) || _d{{ loop.index }}.zmax);
                    var layout = {
                        paper_bgcolor:'white', plot_bgcolor:'white',
                        xaxis:{ title: _d{{ loop.index }}.xaxis+' (r.l.u.)', showgrid:true, gridcolor:'#eee', zeroline:false },
                        yaxis:{ title: _d{{ loop.index }}.yaxis+' (r.l.u.)', showgrid:true, gridcolor:'#eee', zeroline:false },
                        dragmode:'zoom',
                        margin:{ t:40, r:110, b:65, l:80 },
                        font:{ family:'Inter,sans-serif', size:12, color:'#222' },
                        title:{ text: _d{{ loop.index }}.title, font:{ size:12 }, x:0.5 },
                        uirevision: 'fixed'
                    };
                    var cfg = {
                        responsive:true, displaylogo:false, scrollZoom:true, displayModeBar:true,
                        modeBarButtonsToAdd:['toggleSpikelines','resetScale2d'],
                        toImageButtonOptions:{ format:'png', filename:'slice_{{ slice_axis }}_{{ "%.3f"|format(row.a.L_actual) }}', scale:2 }
                    };
                    Plotly.react('sv-plot-{{ loop.index }}', [trace], layout, cfg);
                };
            })();
            </script>
        </div>
        {% endif %}
    {% elif active_tab == "compare" %}
    {% if row.a or row.b %}
    <div class="slice-box" style="flex:1 1 100%;max-width:100%;width:100%;">
        {# ── Header: slice value + view-toggle buttons ── #}
        <div class="slice-filename" style="margin-bottom:8px;flex-wrap:wrap;gap:6px;">
            <div class="slice-dot"></div>
            <span>{{ slice_axis }} = {{ "%.3f"|format((row.a or row.b).L_actual) }}</span>
            <span class="slice-timing">
                {% if row.a %}A: {{ "%.2f"|format(row.a.time) }}s{% endif %}
                {% if row.a and row.b %} &nbsp;|&nbsp; {% endif %}
                {% if row.b %}B: {{ "%.2f"|format(row.b.time) }}s{% endif %}
            </span>
            <div style="margin-left:auto;display:flex;gap:4px;">
                <button id="cmp-btn-png-{{ loop.index }}"
                        onclick="cmpShowView{{ loop.index }}('png')"
                        style="font-size:11px;padding:3px 10px;cursor:pointer;border-radius:4px;
                               background:rgba(0,200,255,0.2);border:1px solid rgba(0,200,255,0.5);
                               color:#00c8ff;font-weight:600;">
                    &#128247; PNG View
                </button>
                <button id="cmp-btn-zoom-{{ loop.index }}"
                        onclick="cmpShowView{{ loop.index }}('zoom')"
                        style="font-size:11px;padding:3px 10px;cursor:pointer;border-radius:4px;
                               background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);
                               color:var(--text-muted);">
                    &#128269; Interactive Zoom
                </button>
            </div>
        </div>

        {# ── PNG VIEW panel ── #}
        <div id="cmp-pnlpng-{{ loop.index }}" style="display:flex;gap:12px;flex-wrap:wrap;">
            {% if row.a %}
            <div style="flex:1;min-width:200px;max-width:420px;">
                <div style="font-size:11px;color:#00c8ff;font-weight:600;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;">
                    A &mdash; {{ row.a.filename }}
                </div>
                <img class="slice-img" src="data:image/png;base64,{{ row.a.png }}"
                     style="width:100%;border:1px solid var(--border-subtle);border-radius:5px;display:block;">
                <a class="dl-btn" href="data:image/png;base64,{{ row.a.png }}"
                   download="A_{{ row.a.filename.split('/')[-1] }}_{{ slice_axis }}{{ '%.3f'|format(row.a.L_actual) }}.png"
                   style="margin-top:4px;display:inline-block;">&#11015; Download A</a>
            </div>
            {% endif %}
            {% if row.b %}
            <div style="flex:1;min-width:200px;max-width:420px;">
                <div style="font-size:11px;color:#f0b429;font-weight:600;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;">
                    B &mdash; {{ row.b.filename }}
                </div>
                <img class="slice-img" src="data:image/png;base64,{{ row.b.png }}"
                     style="width:100%;border:1px solid var(--border-subtle);border-radius:5px;display:block;">
                <a class="dl-btn" href="data:image/png;base64,{{ row.b.png }}"
                   download="B_{{ row.b.filename.split('/')[-1] }}_{{ slice_axis }}{{ '%.3f'|format(row.b.L_actual) }}.png"
                   style="margin-top:4px;display:inline-block;">&#11015; Download B</a>
            </div>
            {% endif %}
        </div>

        {# ── INTERACTIVE ZOOM panel — side by side (hidden by default) ── #}
        <div id="cmp-pnlzoom-{{ loop.index }}" style="display:none;">
            {# Skew + info badge #}
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:11px;">
                <span style="background:rgba(0,200,255,0.12);border:1px solid rgba(0,200,255,0.35);
                             border-radius:4px;padding:2px 8px;color:#00c8ff;font-family:'IBM Plex Mono',monospace;">
                    Skew: {{ skew_angle }}° &mdash; {% if skew_angle != 90 %}Cartesian (matches PNG){% else %}rectangular{% endif %}
                </span>
                <span style="color:var(--text-muted);">scroll to zoom &nbsp;&#128269;&nbsp;|&nbsp; drag to pan</span>
            </div>
            {# Shared controls row #}
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:8px;font-size:12px;">
                <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
                    Colorscale:
                    <select id="cmp-cscale-{{ loop.index }}" onchange="cmpUpdate{{ loop.index }}()"
                            style="font-size:12px;padding:2px 5px;border-radius:4px;">
                        <option value="Viridis" selected>Viridis</option>
                        <option value="Inferno">Inferno</option>
                        <option value="Hot">Hot</option>
                        <option value="RdBu">RdBu</option>
                        <option value="Jet">Jet</option>
                        <option value="Greys">Greys</option>
                    </select>
                </label>
                <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
                    <input type="checkbox" id="cmp-autoz-{{ loop.index }}" checked onchange="cmpUpdate{{ loop.index }}()"> Auto z
                </label>
                <span id="cmp-zrange-{{ loop.index }}" style="display:none;gap:6px;align-items:center;">
                    <label style="color:var(--text-muted);">zmin
                        <input type="number" id="cmp-zmin-{{ loop.index }}" step="any" style="width:80px;font-size:12px;"
                               onchange="cmpUpdate{{ loop.index }}()">
                    </label>
                    <label style="color:var(--text-muted);">zmax
                        <input type="number" id="cmp-zmax-{{ loop.index }}" step="any" style="width:80px;font-size:12px;"
                               onchange="cmpUpdate{{ loop.index }}()">
                    </label>
                </span>
                <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);">
                    <input type="checkbox" id="cmp-smooth-{{ loop.index }}" onchange="cmpUpdate{{ loop.index }}()"> Smooth
                    <span style="font-size:10px;color:var(--text-muted);opacity:0.7;">(slow)</span>
                </label>
                <label style="display:flex;align-items:center;gap:5px;color:var(--text-muted);
                              padding:3px 8px;border-radius:4px;border:1px solid rgba(240,180,41,0.4);
                              background:rgba(240,180,41,0.06);">
                    <input type="checkbox" id="cmp-sync-{{ loop.index }}" checked> &#128279; Sync Zoom
                </label>
                <button onclick="cmpResetZoom{{ loop.index }}()"
                        style="font-size:11px;padding:3px 9px;cursor:pointer;background:rgba(0,200,255,0.1);
                               border:1px solid rgba(0,200,255,0.4);border-radius:4px;color:#00c8ff;">
                    &#8635; Reset Zoom
                </button>
            </div>
            {# Side-by-side plot grid #}
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                {% if row.a %}
                <div>
                    <div style="font-size:11px;color:#00c8ff;font-weight:600;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;">
                        A &mdash; {{ row.a.filename }}
                    </div>
                    <div id="cmp-plot-{{ loop.index }}-a" style="width:100%;height:480px;"></div>
                </div>
                {% endif %}
                {% if row.b %}
                <div>
                    <div style="font-size:11px;color:#f0b429;font-weight:600;margin-bottom:4px;font-family:'IBM Plex Mono',monospace;">
                        B &mdash; {{ row.b.filename }}
                    </div>
                    <div id="cmp-plot-{{ loop.index }}-b" style="width:100%;height:480px;"></div>
                </div>
                {% endif %}
            </div>
        </div>

        <script>
        (function(){
            {% if row.a %}var _da{{ loop.index }} = {{ row.a.plotly_json | safe }};{% else %}var _da{{ loop.index }} = null;{% endif %}
            {% if row.b %}var _db{{ loop.index }} = {{ row.b.plotly_json | safe }};{% else %}var _db{{ loop.index }} = null;{% endif %}
            var _cmpInit{{ loop.index }} = false;
            var _cmpSyncing{{ loop.index }} = false;
            var _cmpSyncTA{{ loop.index }} = null;   // debounce timer A→B
            var _cmpSyncTB{{ loop.index }} = null;   // debounce timer B→A

            function _cmpMakeTrace(d, cs, auto, sm, zminVal, zmaxVal) {
                if (!d) return null;
                var tr = {
                    z: d.z, x: d.x, y: d.y, type: 'heatmap', colorscale: cs,
                    colorbar:{ title:{ text:'log\u2081\u2080(I)', side:'right' }, thickness:16, len:0.88 },
                    hovertemplate: d.xaxis+'=%{x:.4f}<br>'+d.yaxis+'=%{y:.4f}<br>log\u2081\u2080(I)=%{z:.3f}<extra></extra>',
                    // 'fast' uses bilinear (GPU-assisted); 'best' is bicubic (CPU, slow on zoom)
                    zsmooth: sm ? 'fast' : false
                };
                tr.zmin = auto ? d.zmin : (zminVal || d.zmin);
                tr.zmax = auto ? d.zmax : (zmaxVal || d.zmax);
                return tr;
            }

            function _cmpLayout(d, uirev) {
                if (!d) return {};
                return {
                    paper_bgcolor:'white', plot_bgcolor:'white',
                    xaxis:{ title: d.xaxis, showgrid:true, gridcolor:'#eee', zeroline:false },
                    yaxis:{ title: d.yaxis, showgrid:true, gridcolor:'#eee', zeroline:false },
                    dragmode:'zoom', margin:{ t:20, r:90, b:60, l:75 },
                    font:{ family:'Inter,sans-serif', size:11, color:'#222' },
                    // uirevision keeps zoom/pan state intact across Plotly.react() calls
                    uirevision: uirev || 'fixed'
                };
            }

            var _cmpCfg{{ loop.index }} = {
                responsive:true, displaylogo:false, scrollZoom:true,
                displayModeBar:'hover',
                modeBarButtonsToAdd:['toggleSpikelines','resetScale2d'],
                toImageButtonOptions:{ format:'png', scale:2 }
            };

            window['cmpUpdate{{ loop.index }}'] = function() {
                var cs   = document.getElementById('cmp-cscale-{{ loop.index }}').value;
                var auto = document.getElementById('cmp-autoz-{{ loop.index }}').checked;
                var sm   = document.getElementById('cmp-smooth-{{ loop.index }}').checked;
                var zgrp = document.getElementById('cmp-zrange-{{ loop.index }}');
                zgrp.style.display = auto ? 'none' : 'flex';
                var zmin = parseFloat(document.getElementById('cmp-zmin-{{ loop.index }}').value);
                var zmax = parseFloat(document.getElementById('cmp-zmax-{{ loop.index }}').value);
                {% if row.a %}
                var trA = _cmpMakeTrace(_da{{ loop.index }}, cs, auto, sm, zmin, zmax);
                if (trA) Plotly.react('cmp-plot-{{ loop.index }}-a', [trA], _cmpLayout(_da{{ loop.index }}), _cmpCfg{{ loop.index }});
                {% endif %}
                {% if row.b %}
                var trB = _cmpMakeTrace(_db{{ loop.index }}, cs, auto, sm, zmin, zmax);
                if (trB) Plotly.react('cmp-plot-{{ loop.index }}-b', [trB], _cmpLayout(_db{{ loop.index }}), _cmpCfg{{ loop.index }});
                {% endif %}
            };

            window['cmpResetZoom{{ loop.index }}'] = function() {
                var rel = {'xaxis.autorange':true,'yaxis.autorange':true};
                {% if row.a %}Plotly.relayout('cmp-plot-{{ loop.index }}-a', rel);{% endif %}
                {% if row.b %}Plotly.relayout('cmp-plot-{{ loop.index }}-b', rel);{% endif %}
            };

            function _cmpAttachSync{{ loop.index }}() {
                // ── Sync zoom A → B ──
                {% if row.a and row.b %}
                document.getElementById('cmp-plot-{{ loop.index }}-a').on('plotly_relayout', function(ev) {
                    if (_cmpSyncing{{ loop.index }}) return;
                    if (!document.getElementById('cmp-sync-{{ loop.index }}').checked) return;
                    if (ev['xaxis.range[0]'] === undefined && ev['xaxis.autorange'] === undefined) return;
                    clearTimeout(_cmpSyncTA{{ loop.index }});
                    _cmpSyncTA{{ loop.index }} = setTimeout(function() {
                        _cmpSyncing{{ loop.index }} = true;
                        Plotly.relayout('cmp-plot-{{ loop.index }}-b', ev).then(function() {
                            setTimeout(function() { _cmpSyncing{{ loop.index }} = false; }, 60);
                        });
                    }, 30);
                });
                // ── Sync zoom B → A ──
                document.getElementById('cmp-plot-{{ loop.index }}-b').on('plotly_relayout', function(ev) {
                    if (_cmpSyncing{{ loop.index }}) return;
                    if (!document.getElementById('cmp-sync-{{ loop.index }}').checked) return;
                    if (ev['xaxis.range[0]'] === undefined && ev['xaxis.autorange'] === undefined) return;
                    clearTimeout(_cmpSyncTB{{ loop.index }});
                    _cmpSyncTB{{ loop.index }} = setTimeout(function() {
                        _cmpSyncing{{ loop.index }} = true;
                        Plotly.relayout('cmp-plot-{{ loop.index }}-a', ev).then(function() {
                            setTimeout(function() { _cmpSyncing{{ loop.index }} = false; }, 60);
                        });
                    }, 30);
                });
                {% endif %}
            }

            window['cmpShowView{{ loop.index }}'] = function(view) {
                var pngPnl  = document.getElementById('cmp-pnlpng-{{ loop.index }}');
                var zoomPnl = document.getElementById('cmp-pnlzoom-{{ loop.index }}');
                var btnPng  = document.getElementById('cmp-btn-png-{{ loop.index }}');
                var btnZoom = document.getElementById('cmp-btn-zoom-{{ loop.index }}');
                if (view === 'png') {
                    pngPnl.style.display  = 'flex';
                    zoomPnl.style.display = 'none';
                    btnPng.style.cssText  += ';background:rgba(0,200,255,0.2);border:1px solid rgba(0,200,255,0.5);color:#00c8ff;font-weight:600;';
                    btnZoom.style.cssText += ';background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);color:var(--text-muted);font-weight:normal;';
                } else {
                    pngPnl.style.display  = 'none';
                    zoomPnl.style.display = '';
                    btnZoom.style.cssText += ';background:rgba(0,200,255,0.2);border:1px solid rgba(0,200,255,0.5);color:#00c8ff;font-weight:600;';
                    btnPng.style.cssText  += ';background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.15);color:var(--text-muted);font-weight:normal;';
                    if (!_cmpInit{{ loop.index }}) {
                        _cmpInit{{ loop.index }} = true;
                        window['cmpUpdate{{ loop.index }}']();
                        var _ref = _da{{ loop.index }} || _db{{ loop.index }};
                        if (_ref) {
                            document.getElementById('cmp-zmin-{{ loop.index }}').value = _ref.zmin.toFixed(3);
                            document.getElementById('cmp-zmax-{{ loop.index }}').value = _ref.zmax.toFixed(3);
                        }
                        _cmpAttachSync{{ loop.index }}();
                    } else {
                        {% if row.a %}Plotly.Plots.resize('cmp-plot-{{ loop.index }}-a');{% endif %}
                        {% if row.b %}Plotly.Plots.resize('cmp-plot-{{ loop.index }}-b');{% endif %}
                    }
                }
            };
        })();
        </script>
    </div>
    {% endif %}
    {% elif active_tab == "multi" %}
        {% for slot in row.slots %}
        <div class="slice-box">
            {% if slot %}
            <div class="slice-filename">
                <div class="slice-dot" style="background:hsl({{ loop.index0*47 }},80%,60%);box-shadow:0 0 6px hsl({{ loop.index0*47 }},80%,60%);"></div>
                {{ slot.filename.split('/')[-1] }} · {{ slice_axis }} = {{ "%.3f"|format(slot.L_actual) }}
                <span class="slice-timing">{{ "%.2f"|format(slot.time) }}s</span>
            </div>
            <img class="slice-img" src="data:image/png;base64,{{ slot.png }}">
            <a class="dl-btn" href="data:image/png;base64,{{ slot.png }}" download="{{ slot.filename.split('/')[-1] }}_{{ slice_axis }}{{ '%.3f'|format(slot.L_actual) }}.png">&#11015; PNG</a>
            {% else %}<div class="alert-error" style="font-size:12px;">Failed.</div>{% endif %}
        </div>
        {% endfor %}
    {% endif %}
{% endfor %}
</div>
{% endif %}

{% if logs %}
<div class="card">
    <div class="card-header"><div class="card-dot" style="background:var(--text-muted);box-shadow:none;"></div><span class="card-title">Processing Log</span></div>
    <pre>{{ logs }}</pre>
</div>
{% endif %}
{% endif %}
{% endif %}
"""

# ── LINECUT TEMPLATE ──────────────────────────────────────────────────────────
LINECUT_CONTENT = """
<div class="card">
    <h2 class="section-title">1D Linecut Extraction <span class="badge badge-gold">{{ lc_axis }}-slice</span></h2>
    <p class="subheading">Specify start and end points in r.l.u. to extract an intensity profile through the reciprocal-space slice.</p>
</div>
{% if dirs or current %}
<div class="card">
    <div class="card-header"><div class="card-dot"></div><span class="card-title">Navigate Folder</span></div>
    <p style="font-size:12px;margin:0 0 8px;">Current: <span class="path-label">{{ current }}</span></p>
    {% if dirs %}
    <ul>{% for d in dirs %}<li><a href="/slices/linecut?path={{ d }}&lc_axis={{ lc_axis }}&lc_val={{ lc_val }}&cmap={{ cmap }}&vmin={{ vmin }}&vmax={{ vmax }}&xmin={{ xmin }}&xmax={{ xmax }}&ymin={{ ymin }}&ymax={{ ymax }}">📁 {{ d.split('/')[-1] }}</a></li>{% endfor %}</ul>
    {% else %}<p style="font-size:12px;color:var(--text-muted);">No subfolders.</p>{% endif %}
    {% if not files %}<p style="font-size:12px;color:var(--text-muted);margin-top:8px;">No .nxs files in this folder — navigate into a subfolder above.</p>{% endif %}
</div>
{% endif %}
{% if files %}
<div class="card">
    <div class="card-header"><div class="card-dot"></div><span class="card-title">Linecut Configuration</span></div>
    <form method="POST" action="/slices/linecut" onsubmit="showStatus()">
        <input type="hidden" name="path" value="{{ path }}">
        <input type="hidden" name="xmin" value="{{ xmin }}">
        <input type="hidden" name="xmax" value="{{ xmax }}">
        <input type="hidden" name="ymin" value="{{ ymin }}">
        <input type="hidden" name="ymax" value="{{ ymax }}">
        <div class="inline-fields">
            <div class="field-group"><label class="field-label">NXS File</label>
                <select name="lc_file">{% for f in files %}<option value="{{ f }}" {% if f==lc_file %}selected{% endif %}>{{ f.split('/')[-1] }}</option>{% endfor %}</select>
            </div>
            <div class="field-group"><label class="field-label">Slice Axis</label>
                <select name="lc_axis">{% for ax in ["L","K","H"] %}<option value="{{ ax }}" {% if ax==lc_axis %}selected{% endif %}>{{ ax }}</option>{% endfor %}</select>
            </div>
            <div class="field-group"><label class="field-label">{{ lc_axis }} value</label>
                <input type="number" step="0.001" name="lc_val" value="{{ lc_val }}" style="width:100px;">
            </div>
        </div>
        <hr>
        <p style="font-size:12px;color:var(--text-muted);margin:0 0 10px;">
            Coordinates in r.l.u. — X axis = <b>{{ n1 or "n1" }}</b>, Y axis = <b>{{ n2 or "n2" }}</b>
        </p>
        <div class="inline-fields">
            <div class="field-group"><label class="field-label">Start X&#8321; ({{ n1 or "n1" }})</label><input type="number" step="0.01" name="lc_x1" value="{{ lc_x1 }}" style="width:110px;"></div>
            <div class="field-group"><label class="field-label">Start Y&#8321; ({{ n2 or "n2" }})</label><input type="number" step="0.01" name="lc_y1" value="{{ lc_y1 }}" style="width:110px;"></div>
            <div class="field-group"><label class="field-label">End X&#8322; ({{ n1 or "n1" }})</label><input type="number" step="0.01" name="lc_x2" value="{{ lc_x2 }}" style="width:110px;"></div>
            <div class="field-group"><label class="field-label">End Y&#8322; ({{ n2 or "n2" }})</label><input type="number" step="0.01" name="lc_y2" value="{{ lc_y2 }}" style="width:110px;"></div>
        </div>
        <div class="inline-fields" style="margin-top:10px;">
            <div class="field-group"><label class="field-label">Sample pts</label><input type="number" name="lc_npts" value="{{ lc_npts }}" min="50" max="2000" style="width:100px;"></div>
            <div class="field-group"><label class="field-label">Colormap</label>
                <select name="cmap">{% for cm in ["inferno","viridis","plasma","magma","turbo"] %}<option value="{{ cm }}" {% if cm==cmap %}selected{% endif %}>{{ cm }}</option>{% endfor %}</select>
            </div>
            <div class="field-group"><label class="field-label">vmin</label><input type="number" step="0.0001" name="vmin" value="{{ vmin }}" min="0.0001" style="width:110px;"></div>
            <div class="field-group"><label class="field-label">vmax</label><input type="number" step="0.0001" name="vmax" value="{{ vmax }}" min="0.0001" style="width:110px;"></div>
        </div>
        <br>
        <button type="submit" class="btn-primary">Extract Linecut</button>
        <div id="status"></div>
        <div id="progress-container" class="progress-container"><div class="progress-bar"></div></div>
    </form>
</div>
{% if error %}<div class="card"><div class="alert-error">&#9888; {{ error }}</div></div>{% endif %}
{% if overlay %}
<div class="card">
    <div class="card-header">
        <div class="card-dot" style="background:#f0b429;box-shadow:0 0 8px #f0b429;"></div>
        <span class="card-title">Linecut Results</span>
        <span class="slice-timing" style="margin-left:auto;">{{ "%.2f"|format(elapsed) }}s &middot; {{ lc_npts }} pts &middot; actual {{ lc_axis }}={{ "%.3f"|format(val_actual) }}</span>
    </div>
    <div class="two-col" style="align-items:flex-start;">
        <div>
            <p style="font-size:11px;color:var(--text-muted);margin:0 0 6px;">Slice with linecut overlay</p>
            <img src="data:image/png;base64,{{ overlay }}" style="max-width:100%;">
        </div>
        <div>
            <p style="font-size:11px;color:var(--text-muted);margin:0 0 6px;">1D intensity profile</p>
            <img src="data:image/png;base64,{{ profile_img }}" style="max-width:100%;">
        </div>
    </div>
    {% if csv_data %}
    <hr>
    <form method="POST" action="/slices/linecut/csv">
        <input type="hidden" name="csv_data" value="{{ csv_data }}">
        <input type="hidden" name="lc_axis"  value="{{ lc_axis }}">
        <input type="hidden" name="lc_val"   value="{{ lc_val }}">
        <button type="submit">&#11015; Download Linecut CSV</button>
    </form>
    {% endif %}
</div>
{% endif %}
{% endif %}
<div class="card" style="padding:12px 18px;"><a href="/slices?path={{ path }}">&larr; Back to Slice Viewer</a></div>
"""

# ── POWDER DATA HELPERS ───────────────────────────────────────────────────────
def load_avg_radial_sum(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            q  = f["f1/radial_sum/Q"][()]
            f1 = f["f1/radial_sum/radial_sum"][()]
            f2 = f["f2/radial_sum/radial_sum"][()]
            f3 = f["f3/radial_sum/radial_sum"][()]
            if f1.shape != q.shape: return None, None
            return q, (f1 + f2 + f3) / 3.0
    except Exception: return None, None

def find_temperature_files(folder):
    out = []
    if not os.path.isdir(folder): return out
    for fname in os.listdir(folder):
        m = re.search(r"_(\d+)\.nxs$", fname)
        if not m: continue
        fpath = os.path.join(folder, fname)
        q, avg = load_avg_radial_sum(fpath)
        if q is not None: out.append((int(m.group(1)), fpath))
    return sorted(out)

def scan_folder(base):
    rows = []
    if not os.path.isdir(base): return rows
    for fname in sorted(os.listdir(base)):
        if not fname.endswith(".nxs"): continue
        path = os.path.join(base, fname)
        try: data = nxload(path)
        except Exception: rows.append({"file": fname, "entry": "ERROR", "processes": []}); continue
        for entry_name, entry_group in data.entries.items():
            processes = [n for n, g in entry_group.items() if getattr(g, "nxclass", None) == "NXprocess"]
            rows.append({"file": fname, "entry": entry_name, "processes": processes})
    return rows

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ── pyFAI HELPERS ─────────────────────────────────────────────────────────────
def pyfai_load_mask(mask_path):
    if not mask_path: return None
    mask_path = mask_path.strip().replace("\u00A0","").replace("\t","").replace("\n","").replace("\r","")
    if not mask_path: return None
    ext = os.path.splitext(mask_path)[1].lower()
    if ext in [".tif",".tiff",".cbf",".edf"]: return fabio.open(mask_path).data.astype(bool)
    elif ext == ".npy": return np.load(mask_path).astype(bool)
    raise ValueError(f"Unsupported mask format: {ext}")

def pyfai_integrate1d(img_path, poni_path, output_base, mask_array=None, thbin=10000):
    im = fabio.open(img_path).data.astype("uint32"); ai = pyFAI.load(poni_path)
    sat_mask = (im > 65530) | (im <= 0)
    full_mask = (sat_mask | mask_array) if mask_array is not None else sat_mask
    if mask_array is not None and mask_array.shape != im.shape:
        raise ValueError(f"Mask shape {mask_array.shape} != image shape {im.shape}")
    q, I = ai.integrate1d(im, thbin, mask=full_mask, unit="q_A^-1")
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    pd.DataFrame({"Q": q, "I": I}).to_csv(output_base + "_Q.csv", index=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(q, I, lw=1, color="steelblue"); ax.set_xlabel("Q (Å⁻¹)"); ax.set_ylabel("Intensity")
    ax.set_title(os.path.basename(img_path)); ax.set_xlim(2,10); ax.set_ylim(bottom=0); fig.tight_layout()
    return q, I, fig_to_base64(fig)

def pyfai_convert_q_to_tth(q, I, poni_path, output_base):
    ai = pyFAI.load(poni_path)
    wavelength_A = ai.wavelength * 1e10          # metres → Ångströms (q is in Å⁻¹)
    arg = np.clip(q * wavelength_A / (4 * np.pi), -1.0, 1.0)
    tth_deg = np.degrees(2 * np.arcsin(arg))
    pd.DataFrame({"2theta_deg": tth_deg, "I": I}).to_csv(output_base + "_2theta.csv", index=False)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(tth_deg, I, lw=1, color="darkorange"); ax.set_xlabel("2θ (deg)"); ax.set_ylabel("Intensity")
    ax.set_title(os.path.basename(output_base) + " (2θ)"); ax.set_xlim(tth_deg.min(), tth_deg.max()); ax.set_ylim(bottom=0); fig.tight_layout()
    return tth_deg, I, fig_to_base64(fig)

def pyfai_cake(img_path, poni_path, output_base, mask_array=None, nrad=2000, nazim=360):
    im = fabio.open(img_path).data.astype("uint32"); ai = pyFAI.load(poni_path)
    sat_mask = (im > 65530) | (im <= 0)
    full_mask = (sat_mask | mask_array) if mask_array is not None else sat_mask
    cake, tth, chi = ai.integrate2d(im, npt_rad=nrad, npt_azim=nazim, unit="2th_deg", mask=full_mask)
    np.save(output_base + "_cake.npy", cake)
    fig, ax = plt.subplots(figsize=(6,5))
    im_plot = ax.imshow(cake, extent=[chi.min(),chi.max(),tth.max(),tth.min()], aspect="auto", cmap="viridis",
                        vmin=0, vmax=np.percentile(cake[cake>0],99) if np.any(cake>0) else 1)
    ax.set_xlabel("Chi (deg)"); ax.set_ylabel("2θ (deg)"); ax.set_title("Caked: "+os.path.basename(img_path))
    ax.set_xlim(-180,180); ax.set_ylim(0,20); fig.colorbar(im_plot, ax=ax, label="Intensity"); fig.tight_layout()
    return fig_to_base64(fig), cake.shape

def pyfai_process_one(img_path, poni_path, output_base, mask_array, do_1d, do_tth, do_cake, thbin, nazim):
    import json as _json
    result = {"filename": os.path.basename(img_path), "ok": False, "error": None,
              "plot_1d": None, "plot_tth": None, "plot_cake": None, "q_points": "—", "cake_shape": "—",
              "q_json": None, "I_json": None, "tth_json": None, "I_tth_json": None}
    try:
        q, I = None, None
        if do_1d:
            q, I, result["plot_1d"] = pyfai_integrate1d(img_path, poni_path, output_base, mask_array, thbin)
            result["q_points"] = len(q)
            _m = np.isfinite(q) & np.isfinite(I)
            result["q_json"] = _json.dumps(np.round(q[_m], 8).tolist())
            result["I_json"] = _json.dumps(np.round(I[_m], 8).tolist())
        if do_tth:
            if q is None: q, I, _ = pyfai_integrate1d(img_path, poni_path, output_base, mask_array, thbin); result["q_points"] = len(q)
            tth_deg, I_tth, result["plot_tth"] = pyfai_convert_q_to_tth(q, I, poni_path, output_base)
            _mt = np.isfinite(tth_deg) & np.isfinite(I_tth)
            result["tth_json"] = _json.dumps(np.round(tth_deg[_mt], 8).tolist())
            result["I_tth_json"] = _json.dumps(np.round(I_tth[_mt], 8).tolist())
        if do_cake:
            result["plot_cake"], shape = pyfai_cake(img_path, poni_path, output_base, mask_array, nazim=nazim)
            result["cake_shape"] = str(shape)
        result["ok"] = True
    except Exception as exc: result["error"] = str(exc)
    return result

# ── HEIGHT BATCH HELPERS ──────────────────────────────────────────────────────
def _hb_get_ai(poni_path):
    """Return a cached AzimuthalIntegrator for poni_path.
    pyFAI builds its geometry LUT on the first integrate1d call — this takes
    60–90 s.  Reusing the same integrator across requests means that cost is
    paid only once per server lifetime."""
    if poni_path not in _hb_ai_cache:
        _hb_ai_cache[poni_path] = pyFAI.load(poni_path)
    return _hb_ai_cache[poni_path]

def _hb_raw_to_b64(im_uint, mask_array=None, max_px=600):
    """Fast PIL-based raw detector image → base64 PNG. No matplotlib figure overhead."""
    from PIL import Image as _PILImg
    import matplotlib.cm as _cm
    im_log = np.log10(np.maximum(im_uint.astype(np.float32), 1.0))
    vmin = float(np.percentile(im_log, 2))
    vmax = float(np.percentile(im_log, 99))
    norm = np.clip((im_log - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)
    # Apply viridis as a pure array operation — no figure/axes created
    rgb = (_cm.viridis(norm)[:, :, :3] * 255).astype(np.uint8)
    # Red tint on masked pixels
    if mask_array is not None and mask_array.shape == im_uint.shape:
        m = mask_array
        rgb[m, 0] = np.minimum(rgb[m, 0].astype(np.int16) + 110, 255).astype(np.uint8)
        rgb[m, 1] = (rgb[m, 1] * 0.35).astype(np.uint8)
        rgb[m, 2] = (rgb[m, 2] * 0.35).astype(np.uint8)
    img = _PILImg.fromarray(rgb)
    h, w = img.size[1], img.size[0]
    if max(h, w) > max_px:
        scale = max_px / max(h, w)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImg.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def pyfai_height_one(img_path, poni_path, mask_array, thbin=2000, show_raw=False):
    """Process one .cbf for height-batch preview — no file I/O.
    show_raw=False (default): pyFAI 1D integration only — fastest path, skips PIL.
    show_raw=True: PIL raw image only — skips pyFAI entirely (used for on-demand image button).
    Returns plain Python lists suitable for direct JSON serialisation."""
    import time as _time
    result = {"filename": os.path.basename(img_path), "ok": False, "error": None,
              "q": None, "I": None, "raw_b64": None, "q_points": "—",
              "img_shape": None, "t_load_s": None, "t_integrate_s": None,
              "t_render_s": None, "t_total_s": None}
    t0 = _time.perf_counter()
    try:
        t_a = _time.perf_counter()
        im_uint = fabio.open(img_path).data.astype("uint32")   # single load
        result["t_load_s"]  = round(_time.perf_counter() - t_a, 2)
        result["img_shape"] = f"{im_uint.shape[1]}\u00d7{im_uint.shape[0]}"
        if show_raw:
            # Image-only path: skip pyFAI entirely
            t_b = _time.perf_counter()
            result["raw_b64"]    = _hb_raw_to_b64(im_uint, mask_array)
            result["t_render_s"] = round(_time.perf_counter() - t_b, 2)
            result["ok"] = True
        else:
            # Integration-only path: skip PIL entirely
            ai = _hb_get_ai(poni_path)   # cached — LUT built only on first call
            if mask_array is not None and mask_array.shape != im_uint.shape:
                raise ValueError(f"Mask shape {mask_array.shape} != image {im_uint.shape}")
            sat_mask = (im_uint > 65530) | (im_uint <= 0)
            full_mask = (sat_mask | mask_array) if mask_array is not None else sat_mask
            t_b = _time.perf_counter()
            q, I = ai.integrate1d(im_uint, thbin, mask=full_mask, unit="q_A^-1")
            result["t_integrate_s"] = round(_time.perf_counter() - t_b, 2)
            _m = np.isfinite(q) & np.isfinite(I)
            result["q_points"] = int(_m.sum())
            result["q"] = np.round(q[_m], 4).tolist()   # 4 dp is plenty for preview
            result["I"] = np.round(I[_m], 4).tolist()
            result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    result["t_total_s"] = round(_time.perf_counter() - t0, 2)
    return result

# ── SLICE VIEWER HELPERS ──────────────────────────────────────────────────────
def _sv_cache_key(path, suffix):
    h = hashlib.md5(f"{path}:{suffix}".encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{h}.npy")

def _sv_ensure_loaded(path):
    with _cache_lock:
        if path not in _sv_file_cache: _sv_file_cache[path] = nxload(path)
        if path not in _sv_meta_cache:
            g = _sv_file_cache[path]
            _sv_meta_cache[path] = (
                np.array(g.entry.transform.Qh.nxdata),
                np.array(g.entry.transform.Qk.nxdata),
                np.array(g.entry.transform.Ql.nxdata),
            )

def sv_load_slice(path, value, axis="L"):
    _sv_ensure_loaded(path)
    with _cache_lock: Qh, Qk, Ql = _sv_meta_cache[path]
    if axis == "L":   ax_arr, ax1, ax2, n1, n2 = Ql, Qh, Qk, "Qh", "Qk"
    elif axis == "K": ax_arr, ax1, ax2, n1, n2 = Qk, Qh, Ql, "Qh", "Ql"
    else:             ax_arr, ax1, ax2, n1, n2 = Qh, Qk, Ql, "Qk", "Ql"
    idx = int(np.argmin(np.abs(ax_arr - value)))
    mem_key = (path, axis, idx); disk_key = _sv_cache_key(path, f"{axis}_{idx}")
    with _cache_lock:
        if mem_key in _sv_slice_cache:
            return _sv_slice_cache[mem_key], ax1, ax2, float(ax_arr[idx]), n1, n2, "memory"
    if os.path.exists(disk_key):
        s = np.load(disk_key)
        with _cache_lock: _sv_slice_cache[mem_key] = s
        return s, ax1, ax2, float(ax_arr[idx]), n1, n2, "disk-cache"
    with _cache_lock:
        g = _sv_file_cache[path]
        if axis == "L":   s = np.array(g.entry.transform.data[idx,:,:]).T
        elif axis == "K": s = np.array(g.entry.transform.data[:,idx,:]).T
        else:             s = np.array(g.entry.transform.data[:,:,idx])
        np.save(disk_key, s); _sv_slice_cache[mem_key] = s
    return s, ax1, ax2, float(ax_arr[idx]), n1, n2, "hdf5"

def _sv_get_grid():
    global _GRID_LINES
    if _GRID_LINES: return _GRID_LINES
    hl = list(np.arange(-12,12)); slope = np.sqrt(3)*1.15; dp, dn = [], []
    for x0 in np.arange(-12,12):
        x = np.linspace(x0-5, x0+5, 100)
        dp.append((x, slope*(x-x0))); dn.append((x, -slope*(x-x0)))
    _GRID_LINES = (hl, dp, dn); return _GRID_LINES

def sv_draw_grid(ax, **kw):
    hl, dp, dn = _sv_get_grid()
    for y in hl: ax.axhline(y, **kw)
    for x,y in dp: ax.plot(x, y, **kw)
    for x,y in dn: ax.plot(x, y, **kw)

def sv_safe_vminmax(lo, hi, logs, label):
    F = 1e-6
    if not np.isfinite(lo) or lo <= 0: lo = F; logs.append(f"  [{label}] vmin→{lo:.4g}")
    if not np.isfinite(hi) or hi <= 0: hi = lo*10; logs.append(f"  [{label}] vmax→{hi:.4g}")
    lo = max(lo, F); hi = max(hi, F)
    if hi <= lo: hi = lo*10; logs.append(f"  [{label}] vmax=vmin*10={hi:.4g}")
    return float(lo), float(hi)

def sv_compute_shared_vminmax(paths, Ls, axis, logs):
    tasks = [(p, L) for p in paths for L in Ls]; slices = {}
    def _load(args):
        p, L = args
        try: s,*_ = sv_load_slice(p, L, axis); return (p,L), s
        except Exception as e: logs.append(f"  [preload] {os.path.basename(p)} {axis}={L}: {e}"); return (p,L), None
    with ThreadPoolExecutor(max_workers=MAX_IO_WORKERS) as pool:
        for k, s in pool.map(_load, tasks):
            if s is not None: slices[k] = s
    all_pos = [s[s>0] for s in slices.values() if np.any(s>0)]
    if not all_pos: return 1e-3, 1.0
    combined = np.concatenate(all_pos)
    lo, hi = float(np.percentile(combined,1)), float(np.percentile(combined,99))
    logs.append(f"  [shared vmin/vmax] {lo:.4g} / {hi:.4g} ({len(all_pos)} slices)")
    return lo, hi

_SV_AXIS_CFG = {
    "L": {"xlim":(-1,5),"ylim":(-4,4),"skew":60,"grid":True},
    "K": {"xlim":(-1,5),"ylim":(-6,6),"skew":90,"grid":False},
    "H": {"xlim":(-4,4),"ylim":(-6,6),"skew":90,"grid":False},
}

def _sv_skew_to_cartesian(s, ax1, ax2, skew_angle):
    """
    Resample 2D slice s[i,j] at (ax1[i], ax2[j]) from skewed reciprocal-lattice
    coordinates onto a regular Cartesian grid.

    Skew transform:  x_cart = ax1 + ax2*cos(skew_angle)
                     y_cart = ax2*sin(skew_angle)

    Returns (x_cart_1d, y_cart_1d, s_cart) where s_cart[i,j] is at
    (x_cart_1d[i], y_cart_1d[j]), matching the existing z-array convention.
    Returns (ax1, ax2, s) unchanged when skew_angle ≈ 90° (rectangular grid).
    """
    angle = float(skew_angle)
    if abs(angle - 90.0) < 0.5:
        return ax1, ax2, s          # rectangular — nothing to do

    from scipy.interpolate import RegularGridInterpolator
    angle_rad = np.radians(angle)
    sin_a = np.sin(angle_rad)
    cos_a = np.cos(angle_rad)

    # Cartesian corners of the ax1/ax2 domain
    corners_x = [ax1[0]  + ax2[0]  * cos_a,
                 ax1[0]  + ax2[-1] * cos_a,
                 ax1[-1] + ax2[0]  * cos_a,
                 ax1[-1] + ax2[-1] * cos_a]
    x_min, x_max = float(min(corners_x)), float(max(corners_x))
    y_min = float(ax2[0]  * sin_a)
    y_max = float(ax2[-1] * sin_a)
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    # Keep roughly the same pixel count as the source grid
    nx_c = len(ax1)
    ny_c = max(int(len(ax2) * abs(sin_a) + 0.5), 4)

    x_cart = np.linspace(x_min, x_max, nx_c)
    y_cart = np.linspace(y_min, y_max, ny_c)

    # Inverse transform: Cartesian → skewed reciprocal lattice
    # y_cart = a2 * sin_a  =>  a2 = y_cart / sin_a
    # x_cart = a1 + a2 * cos_a  =>  a1 = x_cart - a2 * cos_a
    xx, yy = np.meshgrid(x_cart, y_cart, indexing='ij')   # (nx_c, ny_c)
    a2_orig = yy / sin_a
    a1_orig = xx - a2_orig * cos_a

    interp = RegularGridInterpolator(
        (ax1, ax2), s, method='linear', bounds_error=False, fill_value=np.nan
    )
    pts = np.column_stack([a1_orig.ravel(), a2_orig.ravel()])
    s_cart = interp(pts).reshape(nx_c, ny_c)

    return x_cart, y_cart, s_cart

def sv_render_one(path, val, vmin, vmax, cmap, axis="L", xlim=None, ylim=None, skew=None, show_grid=None):
    import json as _json_sv
    t0 = time.time()
    s, ax1, ax2, val_actual, n1, n2, source = sv_load_slice(path, val, axis)
    nx = NXdata(NXfield(s, name="counts"), (NXfield(ax1, name=n1), NXfield(ax2, name=n2)))
    cfg = _SV_AXIS_CFG.get(axis, _SV_AXIS_CFG["L"])
    xlim = xlim if xlim else cfg["xlim"]; ylim = ylim if ylim else cfg["ylim"]
    skew_angle = skew if skew is not None else cfg["skew"]
    draw_grid   = show_grid if show_grid is not None else cfg["grid"]
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=130)
    plot_slice(nx, vmin=vmin, vmax=vmax, skew_angle=skew_angle, xlim=xlim, ylim=ylim,
               logscale=True, ax=ax, cbar=True, cmap=cmap)
    ax.set_title(f"{os.path.basename(path)}\n{axis} = {val} (actual {val_actual:.3f})")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f"{n1} (r.l.u.)"); ax.set_ylabel(f"{n2} (r.l.u.)")
    if draw_grid: sv_draw_grid(ax, c="w", lw=0.5, alpha=0.5)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig); buf.seek(0)
    # ── Build Plotly JSON for interactive zoom (with skew + downsampling) ──
    _PLOTLY_MAX = 250   # cap each axis at 250 pts — keeps JSON small, rendering fast
    _safe_vmin = max(float(vmin), 1e-12)
    _safe_vmax = max(float(vmax), _safe_vmin * 10)
    _s_raw = np.where(np.isfinite(s) & (s > 0), np.maximum(s, _safe_vmin), _safe_vmin)
    # Apply same skew transformation as the PNG render
    _px, _py, _ps = _sv_skew_to_cartesian(_s_raw, ax1, ax2, skew_angle)
    _ps_safe = np.where(np.isfinite(_ps) & (_ps > 0), np.maximum(_ps, _safe_vmin), _safe_vmin)
    # Downsample to _PLOTLY_MAX × _PLOTLY_MAX if needed (PNG keeps full res)
    _nx_p, _ny_p = _ps_safe.shape
    if _nx_p > _PLOTLY_MAX or _ny_p > _PLOTLY_MAX:
        from scipy.ndimage import zoom as _ndz
        _fx = min(1.0, _PLOTLY_MAX / _nx_p)
        _fy = min(1.0, _PLOTLY_MAX / _ny_p)
        _ps_safe = _ndz(_ps_safe, (_fx, _fy), order=1)
        _px = np.linspace(float(_px[0]), float(_px[-1]), _ps_safe.shape[0])
        _py = np.linspace(float(_py[0]), float(_py[-1]), _ps_safe.shape[1])
    # Round to 4 d.p. — adequate for log-scale display, cuts JSON size ~40 %
    _z_log = np.round(np.log10(_ps_safe), 4).tolist()
    _px_r  = np.round(_px, 5).tolist()
    _py_r  = np.round(_py, 5).tolist()
    _is_skewed = abs(skew_angle - 90.0) >= 0.5
    _pjson = _json_sv.dumps({
        "z":          _z_log,
        "x":          _px_r,
        "y":          _py_r,
        "xaxis":      f"{n1} {'(Cartesian)' if _is_skewed else '(r.l.u.)'}",
        "yaxis":      f"{n2} {'(Cartesian)' if _is_skewed else '(r.l.u.)'}",
        "zmin":       float(np.log10(_safe_vmin)),
        "zmax":       float(np.log10(_safe_vmax)),
        "title":      f"{os.path.basename(path)} · {axis}={val_actual:.3f}",
        "skew_angle": skew_angle,
        "is_skewed":  _is_skewed
    })
    return {"filename": os.path.basename(path), "L_actual": val_actual, "source": source,
            "vmin": vmin, "vmax": vmax, "time": time.time()-t0,
            "png": base64.b64encode(buf.read()).decode("utf-8"),
            "plotly_json": _pjson}

def sv_generate_rows(path_a, path_b, Ls, vmin_m, vmax_m, autoscale, cmap, axis, xlim, ylim, skew=None, show_grid=None):
    logs=[]; rows=[]; t0=time.time(); compare=path_b is not None
    paths=[path_a]+([path_b] if compare else [])
    logs.append(f"File A: {path_a}"); compare and logs.append(f"File B: {path_b}")
    logs.append(f"Axis:{axis}  Ls:{Ls}  autoscale:{autoscale}  skew:{skew}  grid:{show_grid}")
    lo,hi = sv_compute_shared_vminmax(paths,Ls,axis,logs) if autoscale else (float(vmin_m), float(vmax_m))
    vmin,vmax = sv_safe_vminmax(lo,hi,logs,"global")
    for val in Ls:
        row={"L":val,"error":None,"a":None,"b":None}
        try: row["a"] = sv_render_one(path_a,val,vmin,vmax,cmap,axis,xlim,ylim,skew=skew,show_grid=show_grid)
        except Exception as e: row["error"]=f"File A {axis}={val}: {e}"; logs.append(f"  ERR A: {e}")
        if compare:
            try: row["b"] = sv_render_one(path_b,val,vmin,vmax,cmap,axis,xlim,ylim,skew=skew,show_grid=show_grid)
            except Exception as e: row["error"]=(row["error"]+" | " if row["error"] else "")+f"File B {axis}={val}: {e}"
        rows.append(row)
    total=time.time()-t0; logs.append(f"Total: {total:.2f}s")
    return rows, total, vmin, vmax, "\n".join(logs)

def sv_generate_rows_multi(paths, Ls, vmin_m, vmax_m, autoscale, cmap, axis, xlim, ylim, skew=None, show_grid=None):
    logs=[]; rows=[]; t0=time.time()
    logs.append(f"Files: {[os.path.basename(p) for p in paths]}")
    lo,hi = sv_compute_shared_vminmax(paths,Ls,axis,logs) if autoscale else (float(vmin_m), float(vmax_m))
    vmin,vmax = sv_safe_vminmax(lo,hi,logs,"global")
    for val in Ls:
        row={"L":val,"error":None,"slots":[]}
        for path in paths:
            try: slot=sv_render_one(path,val,vmin,vmax,cmap,axis,xlim,ylim,skew=skew,show_grid=show_grid)
            except Exception as e: logs.append(f"  ERR {os.path.basename(path)}: {e}"); slot=None
            row["slots"].append(slot)
        rows.append(row)
    total=time.time()-t0; logs.append(f"Total: {total:.2f}s")
    return rows, total, vmin, vmax, "\n".join(logs)

# ── LINECUT HELPER ────────────────────────────────────────────────────────────
def sv_compute_linecut(path, slice_val, axis, x1, y1, x2, y2,
                       n_points=300, cmap="inferno",
                       vmin=1e-4, vmax=2.49, xlim=None, ylim=None, show_grid=None):
    """
    Extract a 1-D intensity profile along a straight line through a 2-D
    NxRefine slice.  Uses pure matplotlib (no plot_slice dependency) so that
    the linecut overlay is rendered in the same r.l.u. coordinate frame that
    the line endpoints are specified in.

    s has shape (len(ax1_arr), len(ax2_arr)).
    X axis of the plot = n1 = ax1_arr,  Y axis = n2 = ax2_arr.
    So (x1,y1) and (x2,y2) are (n1_value, n2_value) in r.l.u.
    """
    from scipy.ndimage import map_coordinates
    import matplotlib.colors as mcolors

    t0 = time.time(); logs = []

    # ── load 2-D slice ────────────────────────────────────────────────────────
    s, ax1_arr, ax2_arr, val_actual, n1, n2, source = sv_load_slice(
        path, slice_val, axis)
    ax1_arr = np.asarray(ax1_arr, dtype=float)
    ax2_arr = np.asarray(ax2_arr, dtype=float)
    logs.append(
        f"Slice {axis}={val_actual:.3f}  shape={s.shape}  source={source}\n"
        f"  {n1}: [{ax1_arr[0]:.3f} .. {ax1_arr[-1]:.3f}]  n={len(ax1_arr)}\n"
        f"  {n2}: [{ax2_arr[0]:.3f} .. {ax2_arr[-1]:.3f}]  n={len(ax2_arr)}")

    # ── r.l.u. → fractional array index ──────────────────────────────────────
    def _to_idx(v, arr):
        """Map a physical coordinate to a (possibly fractional) array index."""
        if arr[-1] < arr[0]:          # descending axis — reverse for interp
            return float(np.interp(v, arr[::-1], np.arange(len(arr))[::-1]))
        return float(np.interp(v, arr, np.arange(len(arr))))

    i1 = _to_idx(x1, ax1_arr);  j1 = _to_idx(y1, ax2_arr)
    i2 = _to_idx(x2, ax1_arr);  j2 = _to_idx(y2, ax2_arr)
    logs.append(f"Pixel coords: ({i1:.2f},{j1:.2f}) → ({i2:.2f},{j2:.2f})")

    ii = np.linspace(i1, i2, n_points)
    jj = np.linspace(j1, j2, n_points)

    # ── sample 1-D profile via bilinear interpolation ─────────────────────────
    # Replace non-positive / NaN with 0 before interpolation, then re-mask
    s_safe  = np.where(np.isfinite(s) & (s > 0), s, 0.0)
    profile = map_coordinates(s_safe, [ii, jj], order=1,
                              mode="constant", cval=0.0)
    profile = np.where(profile > 0, profile, np.nan)

    x_arr = np.linspace(x1, x2, n_points)
    y_arr = np.linspace(y1, y2, n_points)
    dist  = np.sqrt((x_arr - x1)**2 + (y_arr - y1)**2)
    logs.append(f"Profile: {np.sum(np.isfinite(profile))}/{n_points} valid pts")

    cfg    = _SV_AXIS_CFG.get(axis, _SV_AXIS_CFG["L"])
    _show_grid = show_grid
    xlim   = xlim or cfg["xlim"]
    ylim   = ylim or cfg["ylim"]
    vmin_s, vmax_s = sv_safe_vminmax(vmin, vmax, logs, "linecut")

    # ── Overlay: pure pcolormesh — no plot_slice / no skew transform ──────────
    # pcolormesh(X, Y, C): X shape (M,), Y shape (N,), C shape (N, M)
    # s shape = (M, N) = (len(ax1_arr), len(ax2_arr))  → transpose to (N, M)
    fig_ov, ax_ov = plt.subplots(figsize=(6, 5), dpi=90)
    fig_ov.patch.set_facecolor("#0d1520")
    ax_ov.set_facecolor("#0d1520")

    s_plot = np.where(s > 0, s, np.nan)
    norm   = mcolors.LogNorm(vmin=vmin_s, vmax=vmax_s)
    mesh   = ax_ov.pcolormesh(ax1_arr, ax2_arr, s_plot.T,
                               norm=norm, cmap=cmap, shading="auto")
    plt.colorbar(mesh, ax=ax_ov, label="Intensity", pad=0.02)

    # draw the linecut
    ax_ov.plot([x1, x2], [y1, y2],
               color="white", lw=2, alpha=0.9, zorder=5, solid_capstyle="round")
    ax_ov.plot(x1, y1, "o", color="lime",  ms=7, zorder=6, label=f"start ({x1:.2f},{y1:.2f})")
    ax_ov.plot(x2, y2, "^", color="yellow",ms=7, zorder=6, label=f"end ({x2:.2f},{y2:.2f})")
    ax_ov.legend(fontsize=8, loc="upper right", framealpha=0.5,
                 facecolor="#0d1520", labelcolor="white")

    ax_ov.set_xlim(xlim); ax_ov.set_ylim(ylim)
    ax_ov.set_xlabel(f"{n1} (r.l.u.)", color="white")
    ax_ov.set_ylabel(f"{n2} (r.l.u.)", color="white")
    ax_ov.tick_params(colors="white")
    for sp in ax_ov.spines.values(): sp.set_edgecolor("#444")
    ax_ov.set_title(
        f"{os.path.basename(path)}\n"
        f"{axis} = {slice_val:.3f}  (actual {val_actual:.3f}) — linecut",
        color="white", fontsize=9)
    if (_show_grid if _show_grid is not None else cfg.get("grid")):
        sv_draw_grid(ax_ov, c="w", lw=0.4, alpha=0.35)
    fig_ov.tight_layout()
    overlay_b64 = fig_to_base64(fig_ov)

    # ── 1-D profile plot ──────────────────────────────────────────────────────
    fig_p, ax_p = plt.subplots(figsize=(7, 3.8), dpi=90)
    fig_p.patch.set_facecolor("white")
    ax_p.set_facecolor("white")
    valid = np.isfinite(profile) & (profile > 0)
    if np.any(valid):
        ax_p.semilogy(dist[valid], profile[valid],
                      lw=1.8, color="#1565c0", solid_capstyle="round")
        ax_p.set_ylim(bottom=vmin_s)
    else:
        ax_p.text(0.5, 0.5, "No valid data along cut",
                  ha="center", va="center",
                  transform=ax_p.transAxes, color="red", fontsize=12)
    ax_p.set_xlabel(
        f"Distance (r.l.u.)  [{n1},{n2}]: "
        f"({x1:.3f},{y1:.3f}) \u2192 ({x2:.3f},{y2:.3f})",
        color="#222")
    ax_p.set_ylabel("Intensity (log scale)", color="#222")
    ax_p.set_title(f"1D Linecut  |  {axis} = {val_actual:.3f}",
                   color="#111", fontweight="bold")
    ax_p.tick_params(colors="#333")
    ax_p.grid(True, which="both", alpha=0.35, color="#ccc")
    ax_p.grid(True, which="major", alpha=0.6, color="#bbb")
    for sp in ax_p.spines.values(): sp.set_edgecolor("#aaa")
    fig_p.tight_layout()
    profile_b64 = fig_to_base64(fig_p)

    elapsed = time.time() - t0
    logs.append(f"Done: {elapsed:.2f}s")
    return overlay_b64, profile_b64, dist, profile, n1, n2, val_actual, elapsed, logs


def _sv_combine_profiles_plot(dist_a, profile_a, label_a,
                              dist_b, profile_b, label_b,
                              axis, val_a, val_b,
                              n1, n2, x1, y1, x2, y2,
                              vmin=1e-4, vmax=2.49):
    """
    Render two 1-D intensity profiles overlaid on a single semilogy chart.
    Returns base64-encoded PNG or raises on hard error.
    """
    vmin_s, vmax_s = sv_safe_vminmax(vmin, vmax, [], "combine")

    # make sure inputs are numpy arrays
    dist_a    = np.asarray(dist_a,    dtype=float)
    profile_a = np.asarray(profile_a, dtype=float)
    dist_b    = np.asarray(dist_b,    dtype=float)
    profile_b = np.asarray(profile_b, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=95)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    valid_a = np.isfinite(profile_a) & (profile_a > 0)
    valid_b = np.isfinite(profile_b) & (profile_b > 0)

    plotted = False
    if np.any(valid_a):
        ax.semilogy(dist_a[valid_a], profile_a[valid_a],
                    lw=2.2, color="#1565c0",         # blue
                    label=f"Temp A · {label_a}  ({axis}={val_a:.3f})",
                    solid_capstyle="round")
        plotted = True
    if np.any(valid_b):
        ax.semilogy(dist_b[valid_b], profile_b[valid_b],
                    lw=2.2, color="#c62828",         # red
                    label=f"Temp B · {label_b}  ({axis}={val_b:.3f})",
                    solid_capstyle="round")
        plotted = True

    if plotted:
        ax.set_ylim(bottom=vmin_s)
    else:
        ax.text(0.5, 0.5, "No valid data found along cut",
                ha="center", va="center",
                transform=ax.transAxes, color="red", fontsize=13)

    ax.set_xlabel(
        f"Distance along cut (r.l.u.)   [{n1},{n2}]:  "
        f"({x1:.3f},{y1:.3f}) \u2192 ({x2:.3f},{y2:.3f})",
        color="#222", fontsize=10)
    ax.set_ylabel("Intensity (log scale)", color="#222", fontsize=10)
    ax.set_title(
        f"1D Linecut — Temperature Comparison   |   {axis}-slice",
        color="#111", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#333")
    ax.grid(True, which="both", alpha=0.35, color="#ccc")
    ax.grid(True, which="major", alpha=0.6, color="#bbb")
    for sp in ax.spines.values(): sp.set_edgecolor("#aaa")
    ax.legend(fontsize=10, framealpha=0.9,
              facecolor="white", edgecolor="#aaa",
              labelcolor="#111", loc="best")
    fig.tight_layout(pad=1.2)
    return fig_to_base64(fig)


# ── PYFAI ROUTES ──────────────────────────────────────────────────────────────
@app.route("/pyfai")
def pyfai_page():
    kw = {k: request.args.get(k,"") for k in ["img_path","folder_path","poni_path","mask_path","output_path"]}
    kw["mode"] = request.args.get("mode","single")
    return render_base(render_template_string(PYFAI_CONTENT, **kw), "pyfai")

@app.route("/pyfai/browse")
def pyfai_browse():
    from urllib.parse import urlencode
    field   = request.args.get("field","img_path")
    pick    = request.args.get("pick","file")
    ext_raw = request.args.get("ext","")
    state   = {k: request.args.get(k,"") for k in ["img_path","folder_path","poni_path","mask_path","output_path","mode"]}
    default = PYFAI_IMG_ROOT if field in ("img_path","folder_path") else ROOT
    path    = os.path.abspath(request.args.get("path", default))
    exts    = [e.strip().lower() for e in ext_raw.split(",") if e.strip()]
    parts   = path.split("/")
    crumbs  = [("/" if not parts[1:i] else parts[i-1], "/" + "/".join(parts[1:i]) if parts[1:i] else "/") for i in range(1, len(parts)+1)]
    subdirs, files = [], []
    if os.path.isdir(path):
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            if os.path.isdir(full): subdirs.append(full)
            elif pick=="file" and (not exts or any(item.lower().endswith(e) for e in exts)): files.append(full)
    def browse_qs(nav_path):
        p={"field":field,"pick":pick,"ext":ext_raw,"path":nav_path}; p.update(state); return urlencode(p)
    content = render_template_string(PYFAI_BROWSE_CONTENT, field=field, pick=pick, ext=ext_raw,
        current=path, subdirs=subdirs, files=files, breadcrumbs=crumbs,
        state=state, browse_qs=browse_qs, back_url="/pyfai?"+urlencode(state))
    return render_base(content, "pyfai")

@app.route("/pyfai/pick", methods=["POST"])
def pyfai_pick():
    from flask import redirect
    from urllib.parse import urlencode
    field = request.form["field"]; value = request.form["value"]
    fields = {k: request.form.get(k,"") for k in ["img_path","folder_path","poni_path","mask_path","output_path","mode"]}
    fields[field] = value
    return redirect("/pyfai?" + urlencode(fields))

@app.route("/pyfai/run", methods=["POST"])
def pyfai_run():
    mode      = request.form.get("mode","single")
    poni_path = request.form.get("poni_path","").strip()
    mask_path = request.form.get("mask_path","").strip()
    # thbin: height_batch uses its own field (thbin_hb), standard modes use thbin
    try:
        thbin = int(request.form.get("thbin_hb" if mode=="height_batch" else "thbin", 10000))
    except (ValueError, TypeError):
        thbin = 10000
    errors=[]; results=[]; mask_array=None
    if not poni_path or not os.path.isfile(poni_path): errors.append(f"PONI not found: {poni_path!r}")
    if mask_path:
        try: mask_array = pyfai_load_mask(mask_path)
        except Exception as e: errors.append(f"Mask error: {e}")

    # ── Height Batch: return viewer page immediately — images fetched via AJAX ──
    if mode == "height_batch":
        import json as _json
        def _hb_err(msg):
            if msg: errors.append(msg)
            return render_base(f'<div class="card"><p class="missing">{"<br>".join(errors)}</p>'
                               f'<a href="/pyfai">&#8592; Back</a></div>', "pyfai")
        if errors: return _hb_err("")
        folder_path = request.form.get("folder_path","").strip()
        if not folder_path or not os.path.isdir(folder_path):
            return _hb_err(f"Folder not found: {folder_path!r}")
        img_paths = sorted(glob.glob(os.path.join(folder_path,"*.cbf")))
        if not img_paths:
            return _hb_err(f"No .cbf files found in {folder_path!r}")
        # Return the viewer immediately — no image processing here
        return render_base(render_template_string(PYFAI_HEIGHT_VIEWER_CONTENT,
            img_paths_json=_json.dumps(img_paths),
            poni_path=poni_path, mask_path=mask_path or "",
            thbin=thbin, n_files=len(img_paths)), "pyfai")

    # ── Standard single / batch modes ─────────────────────────────────────────
    output_path = request.form.get("output_path","").strip()
    do_1d       = "do_1d"   in request.form
    do_tth      = "do_tth"  in request.form
    do_cake     = "do_cake" in request.form
    nazim       = int(request.form.get("nazim",360))
    if not output_path: errors.append("Output folder required.")
    if not (do_1d or do_tth or do_cake): errors.append("Enable at least one integration option.")
    if errors:
        return render_base(render_template_string(PYFAI_RESULT_CONTENT, results=[], errors=errors,
            output_path=output_path or "(not set)", log_rows=[]), "pyfai")
    os.makedirs(output_path, exist_ok=True)
    if mode == "single":
        img_path = request.form.get("img_path","").strip()
        if not img_path or not os.path.isfile(img_path):
            errors.append(f"Image not found: {img_path!r}")
            return render_base(render_template_string(PYFAI_RESULT_CONTENT, results=[], errors=errors,
                output_path=output_path, log_rows=[]), "pyfai")
        img_paths = [img_path]
    else:
        folder_path = request.form.get("folder_path","").strip()
        if not folder_path or not os.path.isdir(folder_path):
            errors.append(f"Folder not found: {folder_path!r}")
            return render_base(render_template_string(PYFAI_RESULT_CONTENT, results=[], errors=errors,
                output_path=output_path, log_rows=[]), "pyfai")
        img_paths = sorted(glob.glob(os.path.join(folder_path,"*.cbf")))
        if not img_paths:
            errors.append(f"No .cbf in {folder_path!r}")
            return render_base(render_template_string(PYFAI_RESULT_CONTENT, results=[], errors=errors,
                output_path=output_path, log_rows=[]), "pyfai")
    for img_path in img_paths:
        label = os.path.splitext(os.path.basename(img_path))[0]
        results.append(pyfai_process_one(img_path, poni_path, os.path.join(output_path,label),
            mask_array, do_1d, do_tth, do_cake, thbin, nazim))
    log_rows=[]
    if len(results)>1:
        log_rows=[{"filename":r["filename"],"q_points":r["q_points"],"cake_shape":r["cake_shape"],
                   "ok":r["ok"],"error":r.get("error","")} for r in results]
        try: pd.DataFrame(log_rows).to_csv(os.path.join(output_path,"batch_log.csv"),index=False)
        except: pass
    return render_base(render_template_string(PYFAI_RESULT_CONTENT, results=results, errors=errors,
        output_path=output_path, log_rows=log_rows), "pyfai")


@app.route("/pyfai/height_one_ajax", methods=["POST"])
def pyfai_height_one_ajax():
    """AJAX endpoint: process a single .cbf and return JSON — called by height batch viewer.
    show_raw=0 (default): integration only (fast).
    show_raw=1: raw PIL image only, no pyFAI (on-demand image button)."""
    from flask import jsonify
    img_path  = request.form.get("img_path","").strip()
    poni_path = request.form.get("poni_path","").strip()
    mask_path = request.form.get("mask_path","").strip()
    show_raw  = request.form.get("show_raw","0") == "1"
    try:    thbin = int(request.form.get("thbin", 2000))
    except: thbin = 2000
    fname = os.path.basename(img_path) if img_path else "?"
    if not img_path or not os.path.isfile(img_path):
        return jsonify({"ok": False, "error": f"File not found: {img_path!r}", "filename": fname})
    if not show_raw and (not poni_path or not os.path.isfile(poni_path)):
        return jsonify({"ok": False, "error": f"PONI not found: {poni_path!r}", "filename": fname})
    mask_array = None
    if mask_path:
        try: mask_array = pyfai_load_mask(mask_path)
        except Exception as e:
            return jsonify({"ok": False, "error": f"Mask error: {e}", "filename": fname})
    result = pyfai_height_one(img_path, poni_path, mask_array, thbin, show_raw=show_raw)
    return jsonify(result)

# ── POWDER ROUTES ─────────────────────────────────────────────────────────────
@app.context_processor
def inject_year():
    import datetime
    return {"year": datetime.datetime.now().year}

@app.route("/")
def home():
    return browse()

@app.route("/browse")
def browse():
    path = os.path.abspath(request.args.get("path", ROOT))
    dirs, files = [], []
    if os.path.isdir(path):
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            if os.path.isdir(full): dirs.append(full)
            elif item.endswith(".nxs"): files.append(full)
    return render_base(render_template_string(BROWSER_CONTENT, current=path, dirs=dirs, files=files), "browse")

@app.route("/help")
def help_page():
    return render_base(render_template_string(HELP_CONTENT), "help")


@app.route("/analyze_file")
def analyze_file():
    file = request.args.get("file")
    return render_base(render_template_string(ANALYSIS_CONTENT, file=file, parent=os.path.dirname(file), xmin=0, xmax=10, traces_json=None), "browse")

@app.route("/plot", methods=["POST"])
def plot():
    import json as _json
    file = request.form["file"]
    xmin = float(request.form["xmin"]); xmax = float(request.form["xmax"])
    q, avg = load_avg_radial_sum(file)
    if q is None: return "Error loading data."
    mask = np.isfinite(q) & np.isfinite(avg)
    traces_json = _json.dumps([{"x": np.round(q[mask], 8).tolist(),
                                 "y": np.round(avg[mask], 8).tolist(),
                                 "name": os.path.basename(file),
                                 "color": "#1e88e5"}])
    return render_base(render_template_string(ANALYSIS_CONTENT, file=file,
        parent=os.path.dirname(file), xmin=xmin, xmax=xmax,
        traces_json=traces_json), "browse")

@app.route("/export_csv", methods=["POST"])
def export_csv():
    file = request.form["file"]
    q, avg = load_avg_radial_sum(file)
    if q is None: return "Error loading data."
    out = io.StringIO(); out.write("Q,AverageRadialSum\n")
    for qi, ai in zip(q, avg): out.write(f"{qi},{ai}\n")
    csv_bytes = out.getvalue().encode("utf-8"); out.close()
    return csv_bytes, 200, {"Content-Type":"text/csv","Content-Disposition":f"attachment; filename={os.path.basename(file).replace('.nxs','_avg.csv')}"}

@app.route("/select_temps")
def select_temps():
    path = os.path.abspath(request.args.get("path", ROOT))
    return render_base(render_template_string(SELECT_TEMPS_CONTENT, path=path,
        temp_files=find_temperature_files(path),
        xmin=0, xmax=10, traces_json=None, selected_files=None,
        selected_files_serialized="",
        roi_enabled=False, roi_qmin=2.0, roi_qmax=3.0,
        roi_data=[], roi_temps_json="[]", roi_intens_json="[]",
        roi_colors_json="[]", roi_csv_serialized=""), "browse")

# colour ramp: vivid blue (cold) → vivid red (hot) across N temperatures
def _temp_color(i, n):
    if n <= 1: return "#1e88e5"
    t = i / (n - 1)          # 0 = cold, 1 = hot
    r = int(30  + t * (229 - 30))
    g = int(136 - t * (136 - 57))
    b = int(229 - t * (229 - 53))
    return f"#{r:02x}{g:02x}{b:02x}"

@app.route("/plot_temp", methods=["POST"])
def plot_temp():
    import json as _json
    selected = request.form.getlist("temps")
    xmin = float(request.form.get("xmin", 0)); xmax = float(request.form.get("xmax", 10))
    roi_enabled = "roi_enabled" in request.form
    try:    roi_qmin = float(request.form.get("roi_qmin", 2.0))
    except: roi_qmin = 2.0
    try:    roi_qmax = float(request.form.get("roi_qmax", 3.0))
    except: roi_qmax = 3.0

    traces = []; all_q_avg = []
    for i, fpath in enumerate(selected):
        q, avg = load_avg_radial_sum(fpath)
        if q is None: continue
        m = re.search(r"_(\d+)\.nxs$", fpath)
        label = f"{m.group(1) if m else '?'} K"
        color = _temp_color(i, len(selected))
        mask = np.isfinite(q) & np.isfinite(avg)
        traces.append({"x": np.round(q[mask], 8).tolist(),
                        "y": np.round(avg[mask], 8).tolist(),
                        "name": label, "color": color})
        all_q_avg.append((q, avg, m, color))

    # ── ROI integration ───────────────────────────────────────────────────────
    roi_data = []
    if roi_enabled:
        for i, (q, avg, m, color) in enumerate(all_q_avg):
            T = int(m.group(1)) if m else i
            mask_roi = np.isfinite(q) & np.isfinite(avg) & (q >= roi_qmin) & (q <= roi_qmax)
            integral = float(np.trapz(avg[mask_roi], q[mask_roi])) if mask_roi.sum() > 1 else 0.0
            roi_data.append({"T": T, "integral": integral, "color": color})
        roi_data.sort(key=lambda d: d["T"])
    roi_temps_json  = _json.dumps([d["T"] for d in roi_data])
    roi_intens_json = _json.dumps([d["integral"] for d in roi_data])
    roi_colors_json = _json.dumps([d["color"] for d in roi_data])
    roi_csv_rows    = "temperature_K,integral_I_dQ\n" + "\n".join(
                          f"{d['T']},{d['integral']:.8g}" for d in roi_data)

    path = request.form["path"]
    return render_base(render_template_string(SELECT_TEMPS_CONTENT, path=path,
        temp_files=find_temperature_files(path),
        xmin=xmin, xmax=xmax, traces_json=_json.dumps(traces),
        selected_files=selected, selected_files_serialized=";".join(selected),
        roi_enabled=roi_enabled, roi_qmin=roi_qmin, roi_qmax=roi_qmax,
        roi_data=roi_data, roi_temps_json=roi_temps_json,
        roi_intens_json=roi_intens_json, roi_colors_json=roi_colors_json,
        roi_csv_serialized=roi_csv_rows), "browse")

@app.route("/export_temp_csv", methods=["POST"])
def export_temp_csv():
    files = [f for f in request.form["files"].split(";") if f]
    out = io.StringIO(); out.write("Temperature,File,Q,Intensity\n")
    for fpath in files:
        q, avg = load_avg_radial_sum(fpath)
        if q is None: continue
        m = re.search(r"_(\d+)\.nxs$", fpath); T = m.group(1) if m else "?"
        for qi, ai in zip(q, avg): out.write(f"{T},{os.path.basename(fpath)},{qi},{ai}\n")
    csv_bytes = out.getvalue().encode("utf-8"); out.close()
    return csv_bytes, 200, {"Content-Type":"text/csv","Content-Disposition":"attachment; filename=temperature_overlays.csv"}

@app.route("/export_roi_csv", methods=["POST"])
def export_roi_csv():
    csv_data = request.form.get("roi_csv_data", "temperature_K,integral_I_dQ\n")
    try:    qmin = float(request.form.get("roi_qmin", 0))
    except: qmin = 0.0
    try:    qmax = float(request.form.get("roi_qmax", 0))
    except: qmax = 0.0
    fname = f"roi_integral_Q{qmin:.2f}-{qmax:.2f}.csv".replace(" ", "_")
    return (csv_data.encode("utf-8"), 200,
            {"Content-Type": "text/csv",
             "Content-Disposition": f"attachment; filename={fname}"})

@app.route("/nxprocess")
def nxprocess():
    path = os.path.abspath(request.args.get("path", ROOT))
    return render_base(render_template_string(NXPROCESS_CONTENT, rows=scan_folder(path), path=path), "browse")

@app.route("/choose_folder_A")
def choose_folder_A():
    path = os.path.abspath(request.args.get("path", ROOT))
    subdirs = [os.path.join(path,d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path,d))]
    return render_base(render_template_string(CHOOSE_FOLDER_A_CONTENT, path=path, subdirs=subdirs), "browse")

@app.route("/choose_folder_B", methods=["POST"])
def choose_folder_B():
    folderA = request.form["folderA"]; path = os.path.abspath(request.form["path"])
    subdirs = [os.path.join(path,d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path,d))]
    return render_base(render_template_string(CHOOSE_FOLDER_B_CONTENT, folderA=folderA, path=path, subdirs=subdirs), "browse")

@app.route("/choose_temps_compare", methods=["POST"])
def choose_temps_compare():
    folderA=request.form["folderA"]; folderB=request.form["folderB"]; path=os.path.abspath(request.form["path"])
    return render_base(render_template_string(CHOOSE_TEMPS_COMPARE_CONTENT, folderA=folderA, folderB=folderB,
        tempsA=find_temperature_files(folderA), tempsB=find_temperature_files(folderB),
        xmin=0, xmax=10, traces_json=None, selectedA=None, selectedB=None,
        selectedA_serialized="", selectedB_serialized="", parent=path), "browse")

@app.route("/plot_temp_compare", methods=["POST"])
def plot_temp_compare():
    import json as _json
    folderA=request.form["folderA"]; folderB=request.form["folderB"]; parent=request.form.get("parent",ROOT)
    tempsA=request.form.getlist("tempsA"); tempsB=request.form.getlist("tempsB")
    xmin=float(request.form["xmin"]); xmax=float(request.form["xmax"])
    nameA=os.path.basename(folderA); nameB=os.path.basename(folderB)
    traces=[]
    for i, fpath in enumerate(tempsA):
        q,avg=load_avg_radial_sum(fpath)
        if q is None: continue
        m=re.search(r"_(\d+)\.nxs$",fpath); T=m.group(1) if m else "?"
        mask=np.isfinite(q)&np.isfinite(avg)
        traces.append({"x":np.round(q[mask],8).tolist(),"y":np.round(avg[mask],8).tolist(),
                        "name":f"{nameA} — {T} K",
                        "color":_temp_color(i, len(tempsA)), "dash":"solid"})
    for i, fpath in enumerate(tempsB):
        q,avg=load_avg_radial_sum(fpath)
        if q is None: continue
        m=re.search(r"_(\d+)\.nxs$",fpath); T=m.group(1) if m else "?"
        mask=np.isfinite(q)&np.isfinite(avg)
        traces.append({"x":np.round(q[mask],8).tolist(),"y":np.round(avg[mask],8).tolist(),
                        "name":f"{nameB} — {T} K",
                        "color":_temp_color(i, len(tempsB)), "dash":"dash"})
    selA=";".join(tempsA); selB=";".join(tempsB)
    return render_base(render_template_string(CHOOSE_TEMPS_COMPARE_CONTENT,
        folderA=folderA, folderB=folderB,
        tempsA=find_temperature_files(folderA), tempsB=find_temperature_files(folderB),
        xmin=xmin, xmax=xmax, traces_json=_json.dumps(traces),
        selectedA=tempsA, selectedB=tempsB,
        selectedA_serialized=selA, selectedB_serialized=selB, parent=parent), "browse")

@app.route("/export_compare_csv", methods=["POST"])
def export_compare_csv():
    folderA=request.form["folderA"]; folderB=request.form["folderB"]
    filesA=[f for f in request.form["selA"].split(";") if f]
    filesB=[f for f in request.form["selB"].split(";") if f]
    out=io.StringIO(); out.write("Folder,Temperature,File,Q,Intensity\n")
    for label, files in [(os.path.basename(folderA),filesA),(os.path.basename(folderB),filesB)]:
        for fpath in files:
            q,avg=load_avg_radial_sum(fpath)
            if q is None: continue
            m=re.search(r"_(\d+)\.nxs$",fpath); T=m.group(1) if m else "?"
            for qi,ai in zip(q,avg): out.write(f"{label},{T},{os.path.basename(fpath)},{qi},{ai}\n")
    csv_bytes=out.getvalue().encode("utf-8"); out.close()
    return csv_bytes, 200, {"Content-Type":"text/csv","Content-Disposition":"attachment; filename=compare_two_subfolders.csv"}


# ── SLICE VIEWER ROUTE ────────────────────────────────────────────────────────
@app.route("/slices", methods=["GET","POST"])
def slice_viewer():
    path = request.values.get("path", ROOT)
    path = os.path.realpath(path)
    if not path.startswith(os.path.realpath(ROOT)):
        path = os.path.realpath(ROOT)
    dirs, files = [], []
    try:
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            if os.path.isdir(full): dirs.append(full)
            elif item.endswith(".nxs"): files.append(full)
    except PermissionError:
        pass

    active_tab     = request.form.get("active_tab",  "single")
    file_a         = request.form.get("file_a",       files[0] if files else "")
    file_a_cmp     = request.form.get("file_a_cmp",   files[0] if files else "")
    file_b_cmp     = request.form.get("file_b_cmp",   files[-1] if len(files)>1 else (files[0] if files else ""))
    multi_selected = request.form.getlist("files_multi")
    L_values_str   = request.form.get("Ls",           "0, 0.5, 1, 1.5, 2, 2.5")
    cmap           = request.form.get("cmap",          "inferno")
    autoscale      = request.form.get("autoscale") == "1"
    slice_axis     = request.form.get("slice_axis",    "L")
    if slice_axis not in ("L","K","H"): slice_axis = "L"
    try:    vmin = float(request.form.get("vmin","0.0001"))
    except: vmin = 0.0001
    try:    vmax = float(request.form.get("vmax","2.4901"))
    except: vmax = 2.4901
    _defs = {"L":(-1,5,-4,4),"K":(-1,5,-6,6),"H":(-4,4,-6,6)}
    dx1,dx2,dy1,dy2 = _defs.get(slice_axis,(-1,5,-4,4))
    _skew_defs = {"L":60,"K":90,"H":90}
    try:    skew_angle = int(float(request.form.get("skew_angle", str(_skew_defs.get(slice_axis,60)))))
    except: skew_angle = _skew_defs.get(slice_axis, 60)
    # show_grid: checkbox — default True for L (which has grid=True in cfg), False for K/H
    _grid_defs = {"L":True,"K":False,"H":False}
    # Checkbox fix: unchecked boxes are NOT sent in POST data at all.
    # On GET (first load) use the axis default; on POST always trust the form
    # (present = checked/True, absent = unchecked/False).
    if request.method == "POST":
        show_grid = request.form.get("show_grid") == "1"
    else:
        show_grid = _grid_defs.get(slice_axis, False)
    try:    xmin = float(request.form.get("xmin",str(dx1)))
    except: xmin = float(dx1)
    try:    xmax = float(request.form.get("xmax",str(dx2)))
    except: xmax = float(dx2)
    try:    ymin = float(request.form.get("ymin",str(dy1)))
    except: ymin = float(dy1)
    try:    ymax = float(request.form.get("ymax",str(dy2)))
    except: ymax = float(dy2)

    vmin_warning = None
    if not autoscale:
        if vmin<=0 or vmax<=0: vmin_warning="vmin/vmax must be > 0 for log scale. Auto-corrected."
        elif vmax<=vmin: vmin_warning=f"vmax ({vmax}) must be > vmin ({vmin}). Auto-corrected."

    rows, total_time, used_vmin, used_vmax, logs = [], 0.0, vmin, vmax, None

    # ── linecut tab state ────────────────────────────────────────────────────
    _n1_map = {"L":"Qh","K":"Qh","H":"Qk"}
    _n2_map = {"L":"Qk","K":"Ql","H":"Ql"}
    lc_file   = request.form.get("lc_file",   file_a or (files[0] if files else ""))
    lc_mode   = request.form.get("lc_mode", "single")   # "single" | "compare" | "multi"
    lc_compare = (lc_mode == "compare")                  # backward-compat for template
    # default True on GET (no form data); False only when form submitted without checkbox
    lc_autoscale = (request.method == "GET") or (request.form.get("lc_autoscale") == "1")
    lc_file_b  = request.form.get("lc_file_b", files[-1] if len(files) > 1 else (files[0] if files else ""))
    lc_files_multi = request.form.getlist("lc_files_multi")
    try:    lc_val_multi = float(request.form.get("lc_val_multi", "0"))
    except: lc_val_multi = 0.0
    # Analysis option flags (multi-temp mode)
    lc_show_overlays  = "lc_show_overlays"  in request.form
    lc_show_heatmap   = "lc_show_heatmap"   in request.form
    lc_show_peakfit   = "lc_show_peakfit"   in request.form
    lc_show_peakplot  = "lc_show_peakplot"  in request.form
    hm_autoscale = (request.method == "GET") or ("hm_autoscale" in request.form)
    try:    hm_vmin = float(request.form.get("hm_vmin", 0.0))
    except: hm_vmin = 0.0
    try:    hm_vmax = float(request.form.get("hm_vmax", 1.0))
    except: hm_vmax = 1.0
    try:    lc_fit_qmin   = float(request.form.get("lc_fit_qmin",   0.0))
    except: lc_fit_qmin   = 0.0
    try:    lc_fit_qmax   = float(request.form.get("lc_fit_qmax",   1.0))
    except: lc_fit_qmax   = 1.0
    try:    lc_fit_center = float(request.form.get("lc_fit_center", 0.5))
    except: lc_fit_center = 0.5
    try:    lc_val  = float(request.form.get("lc_val",  "0"))
    except: lc_val  = 0.0
    try:    lc_val_b = float(request.form.get("lc_val_b", "0"))
    except: lc_val_b = 0.0
    try:    lc_x1   = float(request.form.get("lc_x1",  "-1"))
    except: lc_x1   = -1.0
    try:    lc_y1   = float(request.form.get("lc_y1",  "0"))
    except: lc_y1   = 0.0
    try:    lc_x2   = float(request.form.get("lc_x2",  "4"))
    except: lc_x2   = 4.0
    try:    lc_y2   = float(request.form.get("lc_y2",  "0"))
    except: lc_y2   = 0.0
    try:    lc_npts = int(request.form.get("lc_npts", "300"))
    except: lc_npts = 300
    # ── Projection Panel fields (shared + per-file overrides) ────────────
    lc_transform_path = request.form.get("lc_transform_path", "entry/transform").strip() or "entry/transform"
    # Per-file overrides — blank means use shared lc_transform_path
    _lc_a_override = request.form.get("lc_file_a_transform_path", "").strip()
    lc_file_a_transform_path = _lc_a_override if _lc_a_override else lc_transform_path
    _lc_b_override = request.form.get("lc_file_b_transform_path", "").strip()
    lc_file_b_transform_path = _lc_b_override if _lc_b_override else lc_transform_path
    lc_xaxis = request.form.get("lc_xaxis", "Ql")
    lc_yaxis = request.form.get("lc_yaxis", "None")
    lc_plot_lines = "lc_plot_lines" in request.form or request.method == "GET"
    def _lc_pf(name, default):
        try:    return float(request.form.get(name, default))
        except: return float(default)
    lc_ql_min = _lc_pf("lc_ql_min", -1.0); lc_ql_max = _lc_pf("lc_ql_max", 1.0)
    lc_qk_min = _lc_pf("lc_qk_min", -5.0); lc_qk_max = _lc_pf("lc_qk_max", 5.0)
    lc_qh_min = _lc_pf("lc_qh_min", -5.0); lc_qh_max = _lc_pf("lc_qh_max", 5.0)
    lc_ql_lock = "lc_ql_lock" in request.form
    lc_qk_lock = "lc_qk_lock" in request.form
    lc_qh_lock = "lc_qh_lock" in request.form
    # Projection result
    lc_proj_json = None
    lc_proj_info = None

    lc_n1            = _n1_map.get(slice_axis, "n1")
    lc_n2            = _n2_map.get(slice_axis, "n2")
    lc_overlay       = lc_profile_img = lc_csv_data = None
    lc_overlay_b     = lc_compare_img = None
    lc_val_actual    = lc_val
    lc_val_actual_b  = lc_val_b
    lc_elapsed       = 0.0
    lc_error         = None
    lc_file_label_a  = os.path.basename(lc_file)  if lc_file  else ""
    lc_file_label_b  = os.path.basename(lc_file_b) if lc_file_b else ""
    # Plotly JSON arrays (populated after compute)
    lc_plotly_dist_a = "[]"; lc_plotly_prof_a = "[]"
    lc_plotly_dist_b = "[]"; lc_plotly_prof_b = "[]"
    # Multi-temperature linecut results
    lc_multi_traces_json = None
    lc_multi_overlays    = []
    lc_multi_items       = []
    lc_multi_csv         = None
    # Heatmap + peak-fit result variables
    lc_heatmap_json      = None
    lc_heatmap_ntemps    = 0
    lc_fit_json          = None
    lc_fit_table         = []
    lc_fit_csv           = None
    lc_peakplot_json     = None   # Peak Parameters vs T (height, center, fwhm)

    if request.method == "POST" and files:
        if active_tab == "linecut" and lc_mode == "multi":
            # ── Multi-Temperature linecut (NeXpy native NXdata API) ────────
            import json as _json_mt
            if not lc_files_multi:
                lc_error = "No files selected. Hold Ctrl / ⌘ to pick multiple files."
            elif lc_xaxis == "None":
                lc_error = "Select a Slice Axis (H/K/L) before computing."
            else:
                try:
                    _cut_map_mt = {"Qh": "H", "Qk": "K", "Ql": "L"}
                    _cut_axis_mt = _cut_map_mt.get(lc_xaxis, "L")
                    _roi_mt = dict(qh_min=lc_qh_min, qh_max=lc_qh_max,
                                   qk_min=lc_qk_min, qk_max=lc_qk_max,
                                   ql_min=lc_ql_min, ql_max=lc_ql_max)

                    # ── Pre-sort files by temperature ──────────────────────
                    def _extract_temp(fpath, idx):
                        _lbl = os.path.basename(fpath)
                        _m = (re.search(r'(\d+)\s*[Kk]', _lbl) or
                              re.search(r'_(\d+)\.nxs$', _lbl) or
                              re.search(r'(\d+)', _lbl))
                        return int(_m.group(1)) if _m else idx
                    _file_temp_pairs = [(fp, _extract_temp(fp, i))
                                        for i, fp in enumerate(lc_files_multi)]
                    _file_temp_pairs.sort(key=lambda x: x[1])

                    _n_colors = len(_file_temp_pairs)
                    _mt_traces = []; _mt_overlays = []; _mt_items = []
                    _mt_csv_lines = [f"file,temperature_K,{lc_xaxis}_rlu,intensity"]
                    _mt_trace_data = {}
                    _t0_mt = time.time()

                    for _i, (_fpath, _T_val) in enumerate(_file_temp_pairs):
                        _label = os.path.basename(_fpath)
                        _color = _temp_color(_i, _n_colors)
                        try:
                            _nxd_mt = _op_load_nxdata(_fpath, lc_transform_path)
                            _q_vals_mt, _prof_mt = _op_slice_and_project_1d(
                                _nxd_mt, _roi_mt, _cut_axis_mt)
                            _valid_mt = np.isfinite(_q_vals_mt) & np.isfinite(_prof_mt)
                            _xlist = np.round(_q_vals_mt[_valid_mt], 8).tolist()
                            _ylist = np.round(_prof_mt[_valid_mt], 8).tolist()

                            _mt_traces.append({
                                "x": _xlist, "y": _ylist,
                                "type": "scatter",
                                "mode": "lines" if lc_plot_lines else "markers",
                                "name": f"{_T_val} K \u2014 {_label}",
                                "line": {"color": _color, "width": 2}})
                            _mt_items.append({"label": f"{_T_val} K \u2014 {_label}", "color": _color})
                            _mt_trace_data[_fpath] = {
                                "x": _xlist, "y": _ylist,
                                "color": _color, "T": _T_val, "label": _label}
                            for _q, _p in zip(_xlist, _ylist):
                                _mt_csv_lines.append(f"{_label},{_T_val},{_q:.6f},{_p:.6g}")

                            # ── 2D overlay projection for Slice Overlays ──
                            if lc_show_overlays:
                                try:
                                    # Sum over the cut axis → 2D heatmap of remaining two axes
                                    _img2d_mt, _ax0_mt, _ax1_mt, _ax0n, _ax1n = \
                                        _op_slice_and_project_2d(_nxd_mt, _roi_mt, lc_xaxis)
                                    _mt_overlays.append({
                                        "label": f"{_T_val} K \u2014 {_label}",
                                        "color": _color,
                                        "proj_json": _json_mt.dumps({
                                            "z": np.round(_img2d_mt, 6).tolist(),
                                            "x": np.round(_ax1_mt, 6).tolist(),
                                            "y": np.round(_ax0_mt, 6).tolist(),
                                            "xaxis": _ax1n, "yaxis": _ax0n})
                                    })
                                except Exception as _oe:
                                    _mt_overlays.append({
                                        "label": f"{_T_val} K \u2014 {_label} [overlay error: {_oe}]",
                                        "color": _color, "proj_json": None})
                        except Exception as _e:
                            _mt_items.append({"label": f"{_label} [ERROR: {_e}]", "color": "#f97373"})
                            _mt_traces.append({
                                "x": [], "y": [],
                                "type": "scatter", "mode": "lines",
                                "name": f"{_T_val} K \u2014 {_label} [ERROR]",
                                "line": {"color": "#999", "width": 1, "dash": "dot"}})
                    lc_multi_traces_json = _json_mt.dumps(_mt_traces)
                    lc_multi_overlays    = _mt_overlays
                    lc_multi_items       = _mt_items
                    lc_multi_csv         = "\n".join(_mt_csv_lines)
                    lc_elapsed           = time.time() - _t0_mt
                    lc_n1                = lc_xaxis

                    # ── T vs Q HEATMAP ─────────────────────────────────────────
                    if lc_show_heatmap and _mt_trace_data:
                        try:
                            from scipy.interpolate import interp1d as _interp1d
                            _hm_items = sorted(_mt_trace_data.values(), key=lambda r: r["T"])
                            _all_x = [np.array(r["x"]) for r in _hm_items if len(r["x"]) > 1]
                            if _all_x:
                                _d_min = min(a.min() for a in _all_x)
                                _d_max = max(a.max() for a in _all_x)
                                _npts_hm = max(len(a) for a in _all_x)
                                _d_common = np.linspace(_d_min, _d_max, _npts_hm)
                                _hm_z = []
                                _hm_temps_sorted = []
                                for _hr in _hm_items:
                                    _hm_temps_sorted.append(_hr["T"])
                                    _hx = np.array(_hr["x"]); _hy = np.array(_hr["y"])
                                    if len(_hx) > 1:
                                        _fi = _interp1d(_hx, _hy, bounds_error=False, fill_value=0.0)
                                        _hm_z.append(np.round(_fi(_d_common), 8).tolist())
                                    else:
                                        _hm_z.append([0.0] * _npts_hm)
                                lc_heatmap_json = _json_mt.dumps({
                                    "z": _hm_z,
                                    "x": np.round(_d_common, 6).tolist(),
                                    "y": _hm_temps_sorted})
                                lc_heatmap_ntemps = len(_hm_temps_sorted)
                                # Compute actual z-range for the form defaults
                                _flat = [v for row in _hm_z for v in row if v > 0]
                                if _flat and hm_autoscale:
                                    hm_vmin = float(np.percentile(_flat, 2))
                                    hm_vmax = float(np.percentile(_flat, 98))
                        except Exception as _he:
                            lc_error = (lc_error or "") + f" | Heatmap error: {_he}"

                    # ── PEAK FIT: Gaussian + Linear → height vs T ──────────────
                    if lc_show_peakfit and _mt_trace_data:
                        try:
                            from lmfit.models import GaussianModel as _GModel, LinearModel as _LModel
                            _fit_model = _GModel() + _LModel()
                            _fit_temps = []; _fit_heights = []; _fit_colors = []
                            _fit_rows = []
                            _fit_csv_lines = [
                                "temperature_K,peak_height,center_rlu,fwhm,amplitude,sigma,"
                                "bg_slope,bg_intercept,redchi,status"]
                            _sorted_td = sorted(_mt_trace_data.values(), key=lambda r: r["T"])
                            for _fr in _sorted_td:
                                _T2 = _fr["T"]
                                _dx = np.array(_fr["x"]); _dy = np.array(_fr["y"])
                                _roi = (_dx >= lc_fit_qmin) & (_dx <= lc_fit_qmax)
                                if _roi.sum() < 5:
                                    _fit_rows.append({"T": _T2, "height": None, "center": None,
                                                      "fwhm": None, "amplitude": None, "slope": None,
                                                      "redchi": None, "success": False,
                                                      "status": "< 5 pts in ROI"})
                                    continue
                                _dxr = _dx[_roi]; _dyr = _dy[_roi]
                                _c_init = (lc_fit_center
                                           if lc_fit_qmin < lc_fit_center < lc_fit_qmax
                                           else float(_dxr[np.argmax(_dyr)]))
                                _amp_init = float(np.max(_dyr) * abs(lc_fit_qmax - lc_fit_qmin))
                                _pars = _fit_model.make_params(
                                    amplitude=dict(value=_amp_init, min=0),
                                    center=dict(value=_c_init,
                                                min=lc_fit_qmin, max=lc_fit_qmax),
                                    sigma=dict(value=(lc_fit_qmax - lc_fit_qmin) * 0.15,
                                               min=1e-6, max=lc_fit_qmax - lc_fit_qmin),
                                    slope=dict(value=0.0),
                                    intercept=dict(value=float(np.min(_dyr)), min=0))
                                try:
                                    _res = _fit_model.fit(_dyr, _pars, x=_dxr, method="leastsq")
                                    _sig = abs(_res.params["sigma"].value)
                                    _height = _res.params["amplitude"].value / (_sig * np.sqrt(2 * np.pi))
                                    _fwhm = 2.3548 * _sig
                                    _center = _res.params["center"].value
                                    _slope = _res.params["slope"].value
                                    _intercept = _res.params["intercept"].value
                                    _redchi = _res.redchi if _res.redchi is not None else 0.0
                                    _ok = bool(_res.success)
                                    _status = "OK" if _ok else str(_res.message)[:35]
                                    _fit_temps.append(_T2)
                                    _fit_heights.append(_height)
                                    _fit_colors.append(_fr["color"])
                                    _fit_rows.append({"T": _T2, "height": _height, "center": _center,
                                                      "fwhm": _fwhm, "amplitude": _res.params["amplitude"].value,
                                                      "slope": _slope, "redchi": _redchi,
                                                      "success": _ok, "status": _status})
                                    _fit_csv_lines.append(
                                        f"{_T2},{_height:.6e},{_center:.6f},{_fwhm:.6f},"
                                        f"{_res.params['amplitude'].value:.6e},{_sig:.6f},"
                                        f"{_slope:.6e},{_intercept:.6e},{_redchi:.6f},{_status}")
                                except Exception as _fe2:
                                    _fit_rows.append({"T": _T2, "height": None, "center": None,
                                                      "fwhm": None, "amplitude": None, "slope": None,
                                                      "redchi": None, "success": False,
                                                      "status": str(_fe2)[:40]})
                            if _fit_temps:
                                lc_fit_json = _json_mt.dumps({
                                    "temps": _fit_temps,
                                    "heights": _fit_heights,
                                    "colors": _fit_colors})
                            lc_fit_table = _fit_rows
                            lc_fit_csv = "\n".join(_fit_csv_lines)
                            # ── Peak Parameters vs T (4th analysis option) ──
                            if lc_show_peakplot and _fit_rows:
                                _pp_T=[]; _pp_h=[]; _pp_c=[]; _pp_fw=[]; _pp_col=[]
                                for _pr in _fit_rows:
                                    if _pr["success"] and _pr["height"] is not None:
                                        _pp_T.append(_pr["T"])
                                        _pp_h.append(round(_pr["height"], 8))
                                        _pp_c.append(round(_pr["center"], 8))
                                        _pp_fw.append(round(_pr["fwhm"],  8))
                                        # reuse colour from _mt_trace_data if available
                                        _pp_col.append(
                                            _mt_trace_data.get(
                                                next((f for f,d in _mt_trace_data.items()
                                                      if d["T"]==_pr["T"]), None), {}
                                            ).get("color", "#38bdf8"))
                                if _pp_T:
                                    lc_peakplot_json = _json_mt.dumps({
                                        "T":      _pp_T,
                                        "height": _pp_h,
                                        "center": _pp_c,
                                        "fwhm":   _pp_fw,
                                        "colors": _pp_col,
                                        "fit_qmin":  lc_fit_qmin,
                                        "fit_qmax":  lc_fit_qmax,
                                        "slice_axis": lc_xaxis,
                                        "roi": {
                                            "ql_min": lc_ql_min, "ql_max": lc_ql_max,
                                            "qk_min": lc_qk_min, "qk_max": lc_qk_max,
                                            "qh_min": lc_qh_min, "qh_max": lc_qh_max
                                        }
                                    })
                        except ImportError:
                            lc_error = (lc_error or "") + \
                                " | lmfit not installed — run: pip install lmfit"
                        except Exception as _pe:
                            lc_error = (lc_error or "") + f" | Peak fit error: {_pe}"

                except Exception as exc:
                    import traceback
                    lc_error = f"{exc}\n{traceback.format_exc()}"
        elif active_tab == "linecut":
            if not lc_file or not os.path.isfile(lc_file):
                lc_error = f"File not found: {lc_file!r}"
            else:
                try:
                    import json as _json
                    _t0_lc = time.time()
                    lc_file_label_a = os.path.basename(lc_file)

                    # ── 3D ROI extraction (NeXpy native NXdata API) ──
                    # Uses float-based slicing and .sum() — same as NXRefine/NeXpy
                    _roi_lc = dict(qh_min=lc_qh_min, qh_max=lc_qh_max,
                                   qk_min=lc_qk_min, qk_max=lc_qk_max,
                                   ql_min=lc_ql_min, ql_max=lc_ql_max)

                    _nxdata = _op_load_nxdata(lc_file, lc_file_a_transform_path)

                    if lc_xaxis != "None" and lc_yaxis == "None":
                        # ── 1D projection: sum over two axes, profile along X-Axis ──
                        _cut_map = {"Qh": "H", "Qk": "K", "Ql": "L"}
                        _cut_axis = _cut_map.get(lc_xaxis, "L")
                        _q_vals, _profile_a = _op_slice_and_project_1d(
                            _nxdata, _roi_lc, _cut_axis)
                        _valid = np.isfinite(_q_vals) & np.isfinite(_profile_a)
                        _xlist = np.round(_q_vals[_valid], 8).tolist()
                        _ylist = np.round(_profile_a[_valid], 8).tolist()
                        lc_plotly_dist_a = _json.dumps(_xlist)
                        lc_plotly_prof_a = _json.dumps(_ylist)
                        lc_n1 = lc_xaxis
                        lc_n2 = ""
                        lc_overlay = "PROJ1D"  # sentinel to trigger results display
                        lc_proj_info = (
                            f"1D Projection: X={lc_xaxis} | "
                            f"Ql[{lc_ql_min},{lc_ql_max}] "
                            f"Qk[{lc_qk_min},{lc_qk_max}] "
                            f"Qh[{lc_qh_min},{lc_qh_max}]")
                        rows_csv = "\n".join(
                            f"{q:.6f},{p:.6g}" for q, p in zip(_xlist, _ylist))
                        lc_csv_data = (
                            f"# file={lc_file_label_a}  projection={lc_xaxis}\n"
                            f"# Ql[{lc_ql_min},{lc_ql_max}] "
                            f"Qk[{lc_qk_min},{lc_qk_max}] "
                            f"Qh[{lc_qh_min},{lc_qh_max}]\n"
                            f"{lc_xaxis.lower()}_rlu,intensity\n{rows_csv}")

                        # ── Compare mode: also load and project File B ──────────
                        if lc_mode == "compare" and lc_file_b and os.path.isfile(lc_file_b):
                            try:
                                _nxdata_b = _op_load_nxdata(lc_file_b, lc_file_b_transform_path)
                                _q_vals_b, _profile_b = _op_slice_and_project_1d(
                                    _nxdata_b, _roi_lc, _cut_axis)
                                _valid_b = np.isfinite(_q_vals_b) & np.isfinite(_profile_b)
                                lc_plotly_dist_b = _json.dumps(
                                    np.round(_q_vals_b[_valid_b], 8).tolist())
                                lc_plotly_prof_b = _json.dumps(
                                    np.round(_profile_b[_valid_b], 8).tolist())
                                lc_file_label_b = os.path.basename(lc_file_b)
                                if lc_proj_info:
                                    lc_proj_info += f" | B: {lc_file_label_b}"
                            except Exception as _eb:
                                import traceback as _tbb
                                lc_error = f"File B error: {_eb}\n{_tbb.format_exc()}"

                    elif lc_xaxis != "None" and lc_yaxis != "None" and lc_xaxis != lc_yaxis:
                        # ── 2D projection: sum over one axis, heatmap of X vs Y ──
                        _axes_set = {"Qh", "Qk", "Ql"}
                        _sum_axis_name = (_axes_set - {lc_xaxis, lc_yaxis}).pop()
                        _img2d, _ax0, _ax1, _ax0_name, _ax1_name = \
                            _op_slice_and_project_2d(_nxdata, _roi_lc, _sum_axis_name)
                        # Map NXdata axes to requested X/Y orientation
                        # _ax0 = rows (first remaining dim), _ax1 = cols (second)
                        # We need: x along columns, y along rows for Plotly heatmap
                        if _ax1_name.lower().startswith(lc_xaxis[1:].lower()):
                            # _ax1 is X, _ax0 is Y — default orientation
                            _x_arr = _ax1; _y_arr = _ax0
                        elif _ax0_name.lower().startswith(lc_xaxis[1:].lower()):
                            # _ax0 is X, _ax1 is Y — need to transpose
                            _img2d = _img2d.T
                            _x_arr = _ax0; _y_arr = _ax1
                        else:
                            _x_arr = _ax1; _y_arr = _ax0
                        lc_proj_json = _json.dumps({
                            "z": np.round(_img2d, 6).tolist(),
                            "x": np.round(_x_arr, 6).tolist(),
                            "y": np.round(_y_arr, 6).tolist(),
                            "xaxis": lc_xaxis,
                            "yaxis": lc_yaxis})
                        lc_overlay = "PROJ2D"
                        lc_proj_info = (
                            f"2D Projection: X={lc_xaxis}, Y={lc_yaxis} (sum over {_sum_axis_name}) | "
                            f"Ql[{lc_ql_min},{lc_ql_max}] "
                            f"Qk[{lc_qk_min},{lc_qk_max}] "
                            f"Qh[{lc_qh_min},{lc_qh_max}]")
                    else:
                        lc_error = "Select a valid X-Axis. For 1D, set Y-Axis to None. For 2D, pick different X and Y axes."

                    lc_elapsed = time.time() - _t0_lc
                except Exception as exc:
                    import traceback
                    lc_error = f"{exc}\n{traceback.format_exc()}"
        else:
            Ls = [float(x) for x in re.split(r"[\s,]+", L_values_str.strip()) if x]
            try:
                if active_tab == "multi":
                    if not multi_selected: logs="No files selected."; rows=[]
                    else:
                        rows,total_time,used_vmin,used_vmax,logs = sv_generate_rows_multi(
                            multi_selected,Ls,vmin,vmax,autoscale,cmap,slice_axis,(xmin,xmax),(ymin,ymax),skew=skew_angle,show_grid=show_grid)
                else:
                    pa = file_a_cmp if active_tab=="compare" else file_a
                    pb = file_b_cmp if active_tab=="compare" else None
                    rows,total_time,used_vmin,used_vmax,logs = sv_generate_rows(
                        pa,pb,Ls,vmin,vmax,autoscale,cmap,slice_axis,(xmin,xmax),(ymin,ymax),skew=skew_angle,show_grid=show_grid)
            except Exception as exc:
                import traceback; logs=traceback.format_exc()
                rows=[{"L":L,"error":str(exc),"a":None,"b":None,"slots":[]} for L in Ls]

    content = render_template_string(SLICE_VIEWER_CONTENT,
        current=path, dirs=dirs, files=files,
        active_tab=active_tab, file_a=file_a, file_a_cmp=file_a_cmp, file_b_cmp=file_b_cmp,
        multi_selected=multi_selected, L_values=L_values_str, cmap=cmap,
        slice_axis=slice_axis, autoscale=autoscale, vmin=vmin, vmax=vmax,
        skew_angle=skew_angle, show_grid=show_grid,
        vmin_warning=vmin_warning, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        rows=rows, total_time=total_time, used_vmin=used_vmin, used_vmax=used_vmax,
        logs=logs or "",
        lc_file=lc_file, lc_val=lc_val, lc_npts=lc_npts,
        lc_mode=lc_mode, lc_compare=lc_compare, lc_autoscale=lc_autoscale,
        lc_file_b=lc_file_b, lc_val_b=lc_val_b,
        lc_files_multi=lc_files_multi, lc_val_multi=lc_val_multi,
        lc_x1=lc_x1, lc_y1=lc_y1, lc_x2=lc_x2, lc_y2=lc_y2,
        lc_n1=lc_n1, lc_n2=lc_n2,
        lc_overlay=lc_overlay, lc_profile_img=lc_profile_img,
        lc_overlay_b=lc_overlay_b, lc_compare_img=lc_compare_img,
        lc_val_actual=lc_val_actual, lc_val_actual_b=lc_val_actual_b,
        lc_file_label_a=lc_file_label_a, lc_file_label_b=lc_file_label_b,
        lc_plotly_dist_a=lc_plotly_dist_a, lc_plotly_prof_a=lc_plotly_prof_a,
        lc_plotly_dist_b=lc_plotly_dist_b, lc_plotly_prof_b=lc_plotly_prof_b,
        lc_vmin_log=__import__('math').log10(max(vmin, 1e-12)),
        lc_vmax_log=__import__('math').log10(max(vmax, 1e-12)),
        lc_elapsed=lc_elapsed,
        lc_error=lc_error, lc_csv_data=lc_csv_data,
        # Projection panel fields
        lc_transform_path=lc_transform_path,
        lc_file_a_transform_path=lc_file_a_transform_path,
        lc_file_b_transform_path=lc_file_b_transform_path,
        lc_xaxis=lc_xaxis, lc_yaxis=lc_yaxis, lc_plot_lines=lc_plot_lines,
        lc_ql_min=lc_ql_min, lc_ql_max=lc_ql_max, lc_ql_lock=lc_ql_lock,
        lc_qk_min=lc_qk_min, lc_qk_max=lc_qk_max, lc_qk_lock=lc_qk_lock,
        lc_qh_min=lc_qh_min, lc_qh_max=lc_qh_max, lc_qh_lock=lc_qh_lock,
        lc_proj_json=lc_proj_json, lc_proj_info=lc_proj_info,
        lc_multi_traces_json=lc_multi_traces_json,
        lc_multi_overlays=lc_multi_overlays,
        lc_multi_items=lc_multi_items,
        lc_multi_csv=lc_multi_csv,
        lc_show_overlays=lc_show_overlays,
        lc_show_heatmap=lc_show_heatmap, lc_heatmap_json=lc_heatmap_json,
        lc_heatmap_ntemps=lc_heatmap_ntemps,
        hm_autoscale=hm_autoscale, hm_vmin=hm_vmin, hm_vmax=hm_vmax,
        lc_show_peakfit=lc_show_peakfit, lc_fit_qmin=lc_fit_qmin,
        lc_fit_qmax=lc_fit_qmax, lc_fit_center=lc_fit_center,
        lc_fit_json=lc_fit_json, lc_fit_table=lc_fit_table, lc_fit_csv=lc_fit_csv,
        lc_show_peakplot=lc_show_peakplot, lc_peakplot_json=lc_peakplot_json,
        tf_file=files[0] if files else "", tf_transform_path="entry/transform",
        tf_signal="data", tf_qh="Qh", tf_qk="Qk", tf_ql="Ql",
        tf_slice_axis="Ql", tf_qls="0", tf_rotation=-45, tf_cmap="viridis",
        tf_mode="log", tf_autoscale=True,
        tf_vmin_pct=2.0, tf_vmax_pct=98.0, tf_vmin=0.0, tf_vmax=1.0,
        tf_maxpx=600, tf_axauto=True,
        tf_xmin=-5.0, tf_xmax=5.0, tf_ymin=-5.0, tf_ymax=5.0,
        tf_rows=None, tf_info=None, tf_error=None, tf_elapsed=None,
        **_OP_DEFAULTS)
    return render_base(content, "slices")


# ── THIN-FILM HK ROUTE ───────────────────────────────────────────────────────
@app.route("/slices/thinfilm", methods=["POST"])
def slices_thinfilm():
    import traceback as _tb
    from scipy.ndimage import rotate as _ndrot

    # ── folder context (for nav + file quick-pick) ──
    path = request.form.get("path", ROOT)
    path = os.path.realpath(path)
    if not path.startswith(os.path.realpath(ROOT)):
        path = os.path.realpath(ROOT)
    dirs, files = [], []
    try:
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            if os.path.isdir(full): dirs.append(full)
            elif item.endswith(".nxs"): files.append(full)
    except PermissionError:
        pass

    # ── parse thin-film params ──
    tf_file           = request.form.get("tf_file",           "").strip()
    tf_transform_path = request.form.get("tf_transform_path", "entry/transform").strip()
    tf_signal         = request.form.get("tf_signal",         "data").strip()
    tf_qh             = request.form.get("tf_qh",             "Qh").strip()
    tf_qk             = request.form.get("tf_qk",             "Qk").strip()
    tf_ql             = request.form.get("tf_ql",             "Ql").strip()
    tf_slice_axis     = request.form.get("tf_slice_axis",     "Ql").strip()
    tf_qls_str        = request.form.get("tf_qls",            "0").strip()
    try:    tf_rotation  = float(request.form.get("tf_rotation",  -45))
    except: tf_rotation  = -45.0
    tf_cmap       = request.form.get("tf_cmap", "viridis")
    tf_mode       = request.form.get("tf_mode", "log")          # linear / log / sqrt
    tf_autoscale  = bool(request.form.get("tf_autoscale"))       # percentile-based when True
    try:    tf_vmin_pct = float(request.form.get("tf_vmin_pct", 2))
    except: tf_vmin_pct = 2.0
    try:    tf_vmax_pct = float(request.form.get("tf_vmax_pct", 98))
    except: tf_vmax_pct = 98.0
    try:    tf_vmin     = float(request.form.get("tf_vmin", 0))
    except: tf_vmin     = None
    try:    tf_vmax     = float(request.form.get("tf_vmax", 1))
    except: tf_vmax     = None
    try:    tf_maxpx    = int(request.form.get("tf_maxpx", 600))
    except: tf_maxpx    = 600
    tf_axauto  = bool(request.form.get("tf_axauto"))        # True = matplotlib auto range
    try:    tf_xmin = float(request.form.get("tf_xmin", -5))
    except: tf_xmin = -5.0
    try:    tf_xmax = float(request.form.get("tf_xmax",  5))
    except: tf_xmax =  5.0
    try:    tf_ymin = float(request.form.get("tf_ymin", -5))
    except: tf_ymin = -5.0
    try:    tf_ymax = float(request.form.get("tf_ymax",  5))
    except: tf_ymax =  5.0

    # parse comma- or space-separated slice-axis values (like "L values" in other tabs)
    tf_qls_list = []
    for s in re.split(r"[\s,]+", tf_qls_str.strip()):
        try: tf_qls_list.append(float(s))
        except: pass
    if not tf_qls_list:
        tf_qls_list = [0.0]

    tf_rows = []
    tf_info = tf_error = None
    tf_elapsed = None

    if tf_file:
        _t0 = time.time()
        try:
            from nexusformat.nexus import nxload as _nxload, nxsetmemory as _nxsetmem
            from scipy.ndimage import zoom as _zoom
            _nxsetmem(1_000_000)

            nx   = _nxload(tf_file, "r")
            node = nx
            for p in tf_transform_path.strip("/").split("/"):
                if p: node = node[p]

            # 1-D axis arrays (small, safe to load fully)
            Qh_arr = np.asarray(node[tf_qh].nxvalue, dtype=float)
            Qk_arr = np.asarray(node[tf_qk].nxvalue, dtype=float)
            Ql_arr = np.asarray(node[tf_ql].nxvalue, dtype=float)
            data_node = node[tf_signal]

            # ── axis-selection table ──────────────────────────────────────────
            # Data assumed shape: [nQl, nQk, nQh]  (standard NxRefine order)
            # fix_axis_arr : 1-D array of the axis being fixed (sliced through)
            # fix_dim      : which data dimension to index into
            # view_ax1/2   : the two axes visible in the resulting 2-D image
            #   image shape after slab read: (rows=ax1, cols=ax2) for imshow
            _axis_cfg = {
                "Ql": dict(fix_arr=Ql_arr, fix_dim=0,
                           view_ax1_name=tf_qk, view_ax1_arr=Qk_arr,
                           view_ax2_name=tf_qh, view_ax2_arr=Qh_arr),
                "Qk": dict(fix_arr=Qk_arr, fix_dim=1,
                           view_ax1_name=tf_ql, view_ax1_arr=Ql_arr,
                           view_ax2_name=tf_qh, view_ax2_arr=Qh_arr),
                "Qh": dict(fix_arr=Qh_arr, fix_dim=2,
                           view_ax1_name=tf_ql, view_ax1_arr=Ql_arr,
                           view_ax2_name=tf_qk, view_ax2_arr=Qk_arr),
            }
            _cfg = _axis_cfg.get(tf_slice_axis, _axis_cfg["Ql"])
            fix_arr      = _cfg["fix_arr"]
            fix_dim      = _cfg["fix_dim"]
            ax1_name     = _cfg["view_ax1_name"]   # y-axis of image (rows)
            ax1_arr      = _cfg["view_ax1_arr"]
            ax2_name     = _cfg["view_ax2_name"]   # x-axis of image (cols)
            ax2_arr      = _cfg["view_ax2_arr"]

            tf_info = (
                f"File:       {tf_file}\n"
                f"Transform:  {tf_transform_path}  /  signal: {tf_signal}\n"
                f"Data shape: {list(data_node.shape)}\n"
                f"Qh: {Qh_arr[0]:.4f} \u2192 {Qh_arr[-1]:.4f}  ({len(Qh_arr)} pts)\n"
                f"Qk: {Qk_arr[0]:.4f} \u2192 {Qk_arr[-1]:.4f}  ({len(Qk_arr)} pts)\n"
                f"Ql: {Ql_arr[0]:.4f} \u2192 {Ql_arr[-1]:.4f}  ({len(Ql_arr)} pts)\n"
                f"Slice axis: {tf_slice_axis}  |  view: {ax2_name} vs {ax1_name}\n"
                f"Max px/side: {tf_maxpx}  rotation: {tf_rotation}\u00b0"
            )

            # Q-space extent for the two view axes (after rotation)
            ax1_span = float(ax1_arr[-1] - ax1_arr[0])
            ax2_span = float(ax2_arr[-1] - ax2_arr[0])
            theta_r  = np.radians(tf_rotation)
            ex = abs(ax2_span * np.cos(theta_r)) + abs(ax1_span * np.sin(theta_r))
            ey = abs(ax2_span * np.sin(theta_r)) + abs(ax1_span * np.cos(theta_r))
            extent = [-ex/2, ex/2, -ey/2, ey/2]

            for val_target in tf_qls_list:
                try:
                    fix_idx    = int(np.argmin(np.abs(fix_arr - val_target)))
                    actual_val = float(fix_arr[fix_idx])

                    # Slab read – only this 2-D slice (axis-dependent)
                    if fix_dim == 0:
                        slab = data_node[fix_idx, :, :].nxvalue
                    elif fix_dim == 1:
                        slab = data_node[:, fix_idx, :].nxvalue
                    else:  # fix_dim == 2
                        slab = data_node[:, :, fix_idx].nxvalue
                    HK = np.asarray(slab, dtype=float)
                    HK = np.nan_to_num(HK, nan=0.0)   # NaN → 0 before rotate

                    # Downsample for speed (bilinear, order=1)
                    if HK.shape[0] > tf_maxpx or HK.shape[1] > tf_maxpx:
                        scale = min(tf_maxpx / HK.shape[0], tf_maxpx / HK.shape[1])
                        HK = _zoom(HK, scale, order=1)

                    # Rotate: order=1 (fast bilinear), cval=0 (clean zero bg)
                    HK_rot = _ndrot(HK, tf_rotation, reshape=True,
                                    order=1, mode="constant", cval=0.0)

                    # Apply display mode scaling
                    if tf_mode == "log":
                        plot_data = np.log10(np.clip(HK_rot, 0, None) + 1)
                        _cb_label = "log₁₀(I+1)"
                    elif tf_mode == "sqrt":
                        plot_data = np.sqrt(np.clip(HK_rot, 0, None))
                        _cb_label = "√I"
                    else:  # linear
                        plot_data = HK_rot
                        _cb_label = "Intensity"
                    # vmin / vmax: percentile-based or explicit
                    if tf_autoscale or tf_vmin is None or tf_vmax is None:
                        vm = np.nanpercentile(plot_data, tf_vmin_pct)
                        vx = np.nanpercentile(plot_data, tf_vmax_pct)
                    else:
                        vm = tf_vmin
                        vx = tf_vmax

                    fig, ax = plt.subplots(figsize=(5, 4.5))
                    im = ax.imshow(plot_data, origin="lower", aspect="auto",
                                   cmap=tf_cmap, vmin=vm, vmax=vx, extent=extent)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=_cb_label)
                    ax.set_title(
                        f"{tf_slice_axis} = {actual_val:.4f} \u212b\u207b\u00b9"
                        f"  [{tf_mode}]  rot {tf_rotation}\u00b0",
                        fontsize=9.5)
                    ax.set_xlabel(f"{ax2_name}\u2019 (\u212b\u207b\u00b9)")
                    ax.set_ylabel(f"{ax1_name}\u2019 (\u212b\u207b\u00b9)")
                    if not tf_axauto:
                        ax.set_xlim(tf_xmin, tf_xmax)
                        ax.set_ylim(tf_ymin, tf_ymax)
                    fig.tight_layout()

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
                    buf.seek(0)
                    img_b64 = base64.b64encode(buf.read()).decode()
                    plt.close(fig)
                    tf_rows.append({"axis_name": tf_slice_axis, "axis_val": actual_val,
                                    "img": img_b64, "error": None})

                except Exception as inner:
                    tf_rows.append({"axis_name": tf_slice_axis, "axis_val": val_target,
                                    "img": None, "error": str(inner)})

        except Exception:
            tf_error = _tb.format_exc()
        tf_elapsed = time.time() - _t0

    # ── render slice_viewer template with thinfilm tab active ──
    _f0 = files[0] if files else ""
    content = render_template_string(SLICE_VIEWER_CONTENT,
        current=path, dirs=dirs, files=files,
        active_tab="thinfilm",
        file_a=_f0, file_a_cmp=_f0,
        file_b_cmp=files[-1] if len(files) > 1 else _f0,
        multi_selected=[], L_values="0", cmap="inferno",
        slice_axis="L", autoscale=True, vmin=0.0001, vmax=2.4901,
        skew_angle=60, show_grid=True,
        vmin_warning=None, xmin=-1, xmax=5, ymin=-4, ymax=4,
        rows=[], total_time=0.0, used_vmin=0.0001, used_vmax=2.4901, logs="",
        lc_file=_f0, lc_val=0.0, lc_npts=300,
        lc_mode="single", lc_compare=False, lc_autoscale=True,
        lc_file_b=_f0, lc_val_b=0.0,
        lc_files_multi=[], lc_val_multi=0.0,
        lc_x1=-1.0, lc_y1=0.0, lc_x2=4.0, lc_y2=0.0,
        lc_n1="Qh", lc_n2="Qk",
        lc_overlay=None, lc_profile_img=None,
        lc_overlay_b=None, lc_compare_img=None,
        lc_val_actual=0.0, lc_val_actual_b=0.0,
        lc_file_label_a="", lc_file_label_b="",
        lc_plotly_dist_a=None, lc_plotly_prof_a=None,
        lc_plotly_dist_b=None, lc_plotly_prof_b=None,
        lc_vmin_log=-4, lc_vmax_log=0,
        lc_elapsed=0.0, lc_error=None, lc_csv_data=None,
        lc_transform_path="entry/transform",
        lc_file_a_transform_path="entry/transform",
        lc_file_b_transform_path="entry/transform",
        lc_xaxis="Ql", lc_yaxis="None", lc_plot_lines=True,
        lc_ql_min=-1.0, lc_ql_max=1.0, lc_ql_lock=False,
        lc_qk_min=-5.0, lc_qk_max=5.0, lc_qk_lock=False,
        lc_qh_min=-5.0, lc_qh_max=5.0, lc_qh_lock=False,
        lc_proj_json=None, lc_proj_info=None,
        lc_multi_traces_json=None, lc_multi_overlays=[], lc_multi_items=[], lc_multi_csv=None,
        lc_show_overlays=False,
        lc_show_heatmap=False, lc_heatmap_json=None, lc_heatmap_ntemps=0,
        hm_autoscale=True, hm_vmin=0.0, hm_vmax=1.0,
        lc_show_peakfit=False, lc_fit_qmin=0.0, lc_fit_qmax=1.0, lc_fit_center=0.5,
        lc_fit_json=None, lc_fit_table=[], lc_fit_csv=None,
        lc_show_peakplot=False, lc_peakplot_json=None,
        tf_file=tf_file, tf_transform_path=tf_transform_path,
        tf_signal=tf_signal, tf_qh=tf_qh, tf_qk=tf_qk, tf_ql=tf_ql,
        tf_slice_axis=tf_slice_axis, tf_qls=tf_qls_str, tf_rotation=tf_rotation,
        tf_cmap=tf_cmap, tf_mode=tf_mode, tf_autoscale=tf_autoscale,
        tf_vmin_pct=tf_vmin_pct, tf_vmax_pct=tf_vmax_pct,
        tf_vmin=tf_vmin if tf_vmin is not None else 0.0,
        tf_vmax=tf_vmax if tf_vmax is not None else 1.0,
        tf_maxpx=tf_maxpx, tf_axauto=tf_axauto,
        tf_xmin=tf_xmin, tf_xmax=tf_xmax, tf_ymin=tf_ymin, tf_ymax=tf_ymax,
        tf_rows=tf_rows, tf_info=tf_info,
        tf_error=tf_error, tf_elapsed=tf_elapsed,
        **_OP_DEFAULTS)
    return render_base(content, "slices")


# ── ORDER PARAMETER: 3D extraction helpers (NeXpy native NXdata API) ─────────
#
# Uses NeXpy's native float-based slicing and .sum() — the same approach as
# NXRefine / NeXpy Projection Panel.
#
# NXRefine standard: data shape = [nQl, nQk, nQh]
#   dim 0 → Ql,  dim 1 → Qk,  dim 2 → Qh
#
# nxdata[ql_lo:ql_hi, qk_lo:qk_hi, qh_lo:qh_hi] — float-based slicing
# nxdata.sum(axis)  — reduces rank by summing along axis; returns NXdata
# ──────────────────────────────────────────────────────────────────────────────

def _op_load_nxdata(nxs_path, transform_path="entry/transform"):
    """
    Open the NXS file and return the NXdata group at *transform_path*.
    The returned object supports NeXpy native float-based slicing and .sum().
    """
    from nexusformat.nexus import nxload as _nxload_op, nxsetmemory as _nxsetmem_op
    _nxsetmem_op(1_000_000)          # raise threshold so large datasets don't error
    root = _nxload_op(nxs_path, "r")
    node = root
    for part in transform_path.strip("/").split("/"):
        if part:
            node = node[part]
    return node                       # NXdata group — supports slicing & .sum()


def _nxdata_to_array(nxfield):
    """Safely extract a numpy float64 array from an NXfield / NXdata signal."""
    if hasattr(nxfield, 'nxdata'):
        arr = np.asarray(nxfield.nxdata, dtype=np.float64)
    elif hasattr(nxfield, 'nxvalue'):
        arr = np.asarray(nxfield.nxvalue, dtype=np.float64)
    else:
        arr = np.asarray(nxfield, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, 0.0)


def _op_slice_roi(nxdata, roi):
    """
    Float-based slice of the NXdata within the ROI box.
    Returns the sliced NXdata (still an NXdata with proper axes metadata).
    Dimension order: [Ql, Qk, Qh].
    """
    return nxdata[roi['ql_min']:roi['ql_max'],
                  roi['qk_min']:roi['qk_max'],
                  roi['qh_min']:roi['qh_max']]


def _op_slice_and_integrate(nxdata, roi):
    """Float-based slice then total sum.  Returns float."""
    sliced = _op_slice_roi(nxdata, roi)
    arr = _nxdata_to_array(sliced.nxsignal)
    return float(arr.sum())


def _op_slice_and_project_1d(nxdata, roi, cut_axis):
    """
    Float-based slice then .sum() over non-cut axes → 1D profile.
    Returns (q_values, intensity_1d) as numpy float64 arrays.

    cut_axis: "H", "K", or "L"

    Dimension-aware: if the float-based slice collapses one or more
    dimensions (e.g. Ql_min ≈ Ql_max → singleton), the sliced NXdata
    may be 2D or even 1D.  We inspect the remaining axis names to
    decide which axes to sum over.
    """
    sliced = _op_slice_roi(nxdata, roi)
    _cut_name = {"H": "Qh", "K": "Qk", "L": "Ql"}[cut_axis]

    # Discover which axes survived the float slice
    remaining = []
    for ax in sliced.nxaxes:
        name = ax.nxname if hasattr(ax, 'nxname') else str(ax)
        remaining.append(name)
    ndim = len(remaining)

    if ndim == 1:
        # Already 1D after slice — nothing to sum
        proj = sliced
    elif ndim == 2:
        # One dimension was collapsed by slicing
        if _cut_name in remaining:
            # Sum over the other surviving axis
            sum_idx = [i for i, n in enumerate(remaining) if n != _cut_name]
            proj = sliced.sum(sum_idx[0])
        else:
            # The cut axis itself was collapsed — sum remaining to get
            # a scalar-ish result; return whatever axis is left
            proj = sliced.sum(0)
    else:
        # Normal 3D case — sum over the two axes that are NOT the cut axis
        sum_axes = tuple(i for i, n in enumerate(remaining) if n != _cut_name)
        # Sum one at a time (highest index first) for compatibility
        proj = sliced
        for ax_idx in sorted(sum_axes, reverse=True):
            proj = proj.sum(ax_idx)

    # Extract arrays from the resulting 1-D NXdata
    intensity = _nxdata_to_array(proj.nxsignal)
    ax = proj.nxaxes[0]
    q_vals = np.asarray(ax.nxdata if hasattr(ax, 'nxdata') else ax, dtype=np.float64)
    return q_vals, intensity


def _op_slice_and_project_2d(nxdata, roi, sum_axis_name):
    """
    Float-based slice then .sum() over one axis for a 2D heatmap.
    sum_axis_name: "Qh", "Qk", or "Ql"
    Returns (img2d, x_arr, y_arr, x_label, y_label).

    Dimension-aware: if the slice already collapsed the sum_axis,
    the result is already 2D and no further sum is needed.
    """
    sliced = _op_slice_roi(nxdata, roi)

    # Discover which axes survived the float slice
    remaining = []
    for ax in sliced.nxaxes:
        name = ax.nxname if hasattr(ax, 'nxname') else str(ax)
        remaining.append(name)
    ndim = len(remaining)

    if sum_axis_name in remaining and ndim > 2:
        # Normal case: sum over the requested axis
        sum_idx = remaining.index(sum_axis_name)
        proj = sliced.sum(sum_idx)
    elif ndim == 2:
        # The sum axis was already collapsed by slicing — result is 2D
        proj = sliced
    elif ndim > 2:
        # sum_axis not in remaining (shouldn't happen) — sum first axis
        proj = sliced.sum(0)
    else:
        proj = sliced

    sig = _nxdata_to_array(proj.nxsignal)
    axes = proj.nxaxes
    ax0 = np.asarray(axes[0].nxdata if hasattr(axes[0], 'nxdata') else axes[0], dtype=np.float64)
    ax1 = np.asarray(axes[1].nxdata if hasattr(axes[1], 'nxdata') else axes[1], dtype=np.float64)
    ax0_name = axes[0].nxname if hasattr(axes[0], 'nxname') else "axis0"
    ax1_name = axes[1].nxname if hasattr(axes[1], 'nxname') else "axis1"
    return sig, ax0, ax1, ax0_name, ax1_name


# ── ORDER PARAMETER DEFAULTS ──────────────────────────────────────────────────
_OP_DEFAULTS = dict(
    op_files=[],
    op_transform_path="entry/transform",
    op_qh_min=-1.0, op_qh_max=1.0,
    op_qk_min=-1.0, op_qk_max=1.0,
    op_ql_min=-1.0, op_ql_max=1.0,
    op_cut_axis="L",
    op_int_auto=True, op_int_ymin=0, op_int_ymax=1e6, op_int_tmin=0, op_int_tmax=500,
    op_lc_vmin=0, op_lc_vmax=50, op_lc_auto=True,
    op_show_heatmap=False,
    op_hm_vmin=0, op_hm_vmax=50, op_hm_auto=True,
    op_do_fit=False,
    op_fit_qmin=-0.6, op_fit_qmax=-0.4, op_fit_center=-0.5,
    op_fit_cmin=-0.6, op_fit_cmax=-0.4,
    op_error=None, op_scan_info=None, op_elapsed=None,
    op_integ_json=None, op_integ_csv=None, op_ntemps=0,
    op_traces_json=None,
    op_heatmap_json=None,
    op_fit_json=None, op_fit_table=[], op_fit_csv=None,
)


# ── ORDER PARAMETER ROUTE ────────────────────────────────────────────────────
@app.route("/slices/orderpar", methods=["POST"])
def slices_orderpar():
    """
    Order parameter tab — 3D ROI extraction from NXS transform data.
    For each selected NXS file (at different temperatures):
      1. Load 3D transform (Qh, Qk, Ql)
      2. Extract sub-volume within ROI box
      3. 1D linecut along chosen cut axis (sum over other two)
      4. Integrated intensity = total sum of ROI
      5. Optionally: heatmap, peak fitting
    """
    import json as _json_op
    import traceback as _tb_op

    # ── folder context ──
    path = request.form.get("path", ROOT)
    path = os.path.realpath(path)
    if not path.startswith(os.path.realpath(ROOT)):
        path = os.path.realpath(ROOT)
    dirs, files = [], []
    try:
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            if os.path.isdir(full): dirs.append(full)
            elif item.endswith(".nxs"): files.append(full)
    except PermissionError:
        pass

    # ── parse form inputs ──
    op_files = request.form.getlist("op_files")
    def _pf(name, default):
        try:    return float(request.form.get(name, default))
        except: return float(default)
    op_transform_path = request.form.get("op_transform_path", "entry/transform").strip()
    op_qh_min = _pf("op_qh_min", -1.0); op_qh_max = _pf("op_qh_max", 1.0)
    op_qk_min = _pf("op_qk_min", -1.0); op_qk_max = _pf("op_qk_max", 1.0)
    op_ql_min = _pf("op_ql_min", -1.0); op_ql_max = _pf("op_ql_max", 1.0)
    op_cut_axis = request.form.get("op_cut_axis", "L")
    if op_cut_axis not in ("H", "K", "L"):
        op_cut_axis = "L"
    op_int_auto = "op_int_auto" in request.form
    op_int_ymin = _pf("op_int_ymin", 0); op_int_ymax = _pf("op_int_ymax", 1e6)
    op_int_tmin = _pf("op_int_tmin", 0); op_int_tmax = _pf("op_int_tmax", 500)
    op_lc_vmin = _pf("op_lc_vmin", 0); op_lc_vmax = _pf("op_lc_vmax", 50)
    op_lc_auto = "op_lc_auto" in request.form
    op_show_heatmap = "op_show_heatmap" in request.form
    op_hm_vmin = _pf("op_hm_vmin", 0); op_hm_vmax = _pf("op_hm_vmax", 50)
    op_hm_auto = "op_hm_auto" in request.form
    op_do_fit = "op_do_fit" in request.form
    op_fit_qmin   = _pf("op_fit_qmin",   -0.6)
    op_fit_qmax   = _pf("op_fit_qmax",   -0.4)
    op_fit_center = _pf("op_fit_center", -0.5)
    op_fit_cmin   = _pf("op_fit_cmin",   -0.6)
    op_fit_cmax   = _pf("op_fit_cmax",   -0.4)

    # Build ROI dict
    roi = dict(qh_min=op_qh_min, qh_max=op_qh_max,
               qk_min=op_qk_min, qk_max=op_qk_max,
               ql_min=op_ql_min, ql_max=op_ql_max)

    # shared controls (for re-render only)
    slice_axis = request.form.get("slice_axis", "L")
    cmap = request.form.get("cmap", "inferno")
    try:    vmin = float(request.form.get("vmin", 0.0001))
    except: vmin = 0.0001
    try:    vmax = float(request.form.get("vmax", 2.4901))
    except: vmax = 2.4901
    try:    xmin = float(request.form.get("xmin", -1))
    except: xmin = -1
    try:    xmax = float(request.form.get("xmax", 5))
    except: xmax = 5
    try:    ymin = float(request.form.get("ymin", -4))
    except: ymin = -4
    try:    ymax = float(request.form.get("ymax", 4))
    except: ymax = 4
    show_grid = "show_grid" in request.form

    # ── result variables ──
    op_error = None; op_scan_info = None; op_elapsed = None
    op_integ_json = None; op_integ_csv = None; op_ntemps = 0
    op_traces_json = None
    op_heatmap_json = None
    op_fit_json = None; op_fit_table = []; op_fit_csv = None

    _t0 = time.time()
    try:
        if not op_files:
            op_error = "No files selected. Hold Ctrl / \u2318 to pick multiple NXS files."
        else:
            # ── Extract temperature from each filename, sort by T ──
            def _op_extract_temp(fpath, idx):
                lbl = os.path.basename(fpath)
                m = (re.search(r'(\d+)\s*[Kk]', lbl) or
                     re.search(r'_(\d+)\.nxs$', lbl) or
                     re.search(r'(\d+)', lbl))
                return int(m.group(1)) if m else idx
            temp_entries = [(fp, _op_extract_temp(fp, i))
                           for i, fp in enumerate(op_files)]
            temp_entries.sort(key=lambda x: x[1])
            op_ntemps = len(temp_entries)
            _n_colors = op_ntemps

            _integ_temps = []; _integ_vals = []; _integ_colors = []
            _integ_csv_lines = ["temperature_K,integrated_intensity"]
            _trace_data = {}   # T → {x, y, color, label}
            _op_traces = []

            for _i, (nxs_path, T) in enumerate(temp_entries):
                _label = os.path.basename(nxs_path)
                _color = _temp_color(_i, _n_colors)
                try:
                    _nxd = _op_load_nxdata(nxs_path, op_transform_path)

                    # Integrated intensity = sum of entire ROI
                    _integ = _op_slice_and_integrate(_nxd, roi)

                    # 1D linecut along cut axis
                    _q_vals, _profile = _op_slice_and_project_1d(
                        _nxd, roi, op_cut_axis)

                    _valid = np.isfinite(_q_vals) & np.isfinite(_profile) & (_profile > 0)
                    _xlist = np.round(_q_vals[_valid], 8).tolist()
                    _ylist = np.round(_profile[_valid], 8).tolist()

                    _integ_temps.append(T)
                    _integ_vals.append(_integ)
                    _integ_colors.append(_color)
                    _integ_csv_lines.append(f"{T},{_integ:.6e}")

                    _op_traces.append({
                        "x": _xlist, "y": _ylist,
                        "type": "scatter", "mode": "lines",
                        "name": f"{T} K \u2014 {_label}",
                        "line": {"color": _color, "width": 2}})

                    _trace_data[T] = {
                        "x": _xlist, "y": _ylist,
                        "color": _color, "label": _label}

                except Exception as _e:
                    _integ_temps.append(T)
                    _integ_vals.append(0.0)
                    _integ_colors.append("#999")
                    _integ_csv_lines.append(f"{T},0.0")
                    _op_traces.append({
                        "x": [], "y": [],
                        "type": "scatter", "mode": "lines",
                        "name": f"{T} K \u2014 {_label} [ERROR: {_e}]",
                        "line": {"color": "#999", "width": 1, "dash": "dot"}})

            # ── Build integrated intensity plot JSON ──
            if _integ_temps:
                op_integ_json = _json_op.dumps({
                    "temps": _integ_temps,
                    "integ": [round(v, 6) for v in _integ_vals],
                    "colors": _integ_colors})
                op_integ_csv = "\n".join(_integ_csv_lines)
                if op_int_auto and _integ_vals:
                    _pos = [v for v in _integ_vals if v > 0]
                    if _pos:
                        op_int_ymin = 0.0
                        op_int_ymax = float(max(_pos) * 1.1)
                    op_int_tmin = float(min(_integ_temps))
                    op_int_tmax = float(max(_integ_temps))

            if _op_traces:
                op_traces_json = _json_op.dumps(_op_traces)

            temps_str = ", ".join(f"{t[1]} K" for t in temp_entries)
            op_scan_info = (
                f"{op_ntemps} files \u2192 temperatures: {temps_str}\n"
                f"ROI: Qh[{op_qh_min},{op_qh_max}] Qk[{op_qk_min},{op_qk_max}] "
                f"Ql[{op_ql_min},{op_ql_max}] | Cut axis: Q{op_cut_axis.lower()}")

            # ── OPTIONAL: HEATMAP ──
            if op_show_heatmap and _trace_data:
                try:
                    from scipy.interpolate import interp1d as _interp1d_op
                    _sorted_temps = sorted(_trace_data.keys())
                    _all_x = [np.array(r["x"]) for r in
                              [_trace_data[t] for t in _sorted_temps] if len(r["x"]) > 1]
                    if _all_x:
                        _d_min = min(a.min() for a in _all_x)
                        _d_max = max(a.max() for a in _all_x)
                        _npts_hm = max(len(a) for a in _all_x)
                        _d_common = np.linspace(_d_min, _d_max, _npts_hm)
                        _hm_z = []; _hm_temps = []
                        for t in _sorted_temps:
                            _hm_temps.append(t)
                            _td = _trace_data[t]
                            _hx = np.array(_td["x"]); _hy = np.array(_td["y"])
                            if len(_hx) > 1:
                                _fi = _interp1d_op(_hx, _hy, bounds_error=False, fill_value=0.0)
                                _hm_z.append(np.round(_fi(_d_common), 8).tolist())
                            else:
                                _hm_z.append([0.0] * _npts_hm)
                        op_heatmap_json = _json_op.dumps({
                            "z": _hm_z,
                            "x": np.round(_d_common, 6).tolist(),
                            "y": _hm_temps})
                        if op_hm_auto:
                            _flat = [v for row in _hm_z for v in row if v > 0]
                            if _flat:
                                op_hm_vmin = float(np.percentile(_flat, 2))
                                op_hm_vmax = float(np.percentile(_flat, 98))
                except Exception as _he:
                    op_error = (op_error or "") + f" | Heatmap error: {_he}"

            # ── OPTIONAL: PEAK FIT → ORDER PARAMETER ──
            if op_do_fit and _trace_data:
                try:
                    from lmfit.models import GaussianModel as _GM, LinearModel as _LM
                    _fit_model = _GM() + _LM()
                    _sorted_temps = sorted(_trace_data.keys())
                    _fit_temps = []; _fit_heights = []; _fit_amps = []; _fit_colors = []
                    _fit_rows = []
                    _fit_csv_lines = [
                        "temperature_K,peak_height,peak_amplitude,peak_center,"
                        "peak_fwhm,peak_sigma,bg_slope,bg_intercept,redchi,status"]

                    for T in _sorted_temps:
                        _td = _trace_data[T]
                        _dx = np.array(_td["x"]); _dy = np.array(_td["y"])
                        _roi_mask = (_dx >= op_fit_qmin) & (_dx <= op_fit_qmax)
                        if _roi_mask.sum() < 5:
                            _fit_rows.append({"T": T, "height": None, "amplitude": None,
                                              "center": None, "fwhm": None, "slope": None,
                                              "redchi": None, "success": False,
                                              "status": "< 5 pts in ROI"})
                            continue
                        _qr = _dx[_roi_mask]; _Ir = _dy[_roi_mask]
                        _pars = _fit_model.make_params(
                            amplitude=dict(value=float(np.max(_Ir) * 0.1), min=0),
                            center=dict(value=op_fit_center,
                                        min=op_fit_cmin, max=op_fit_cmax),
                            sigma=dict(value=abs(op_fit_qmax - op_fit_qmin) * 0.15,
                                       min=1e-6),
                            slope=dict(value=0.0),
                            intercept=dict(value=float(np.min(_Ir)), min=0))
                        try:
                            _res = _fit_model.fit(np.array(_Ir), _pars, x=np.array(_qr),
                                                  method='leastsq')
                            _sig = abs(_res.params['sigma'].value)
                            _amp = _res.params['amplitude'].value
                            _height = _amp / (_sig * np.sqrt(2 * np.pi))
                            _fwhm = 2.3548 * _sig
                            _center = _res.params['center'].value
                            _slope = _res.params['slope'].value
                            _intercept = _res.params['intercept'].value
                            _redchi = _res.redchi if _res.redchi is not None else 0.0
                            _ok = bool(_res.success)
                            _status = "OK" if _ok else str(getattr(_res, 'message', ''))[:35]
                            _fit_temps.append(T)
                            _fit_heights.append(_height)
                            _fit_amps.append(_amp)
                            _fit_colors.append(_td["color"])
                            _fit_rows.append({"T": T, "height": _height, "amplitude": _amp,
                                              "center": _center, "fwhm": _fwhm, "slope": _slope,
                                              "redchi": _redchi, "success": _ok, "status": _status})
                            _fit_csv_lines.append(
                                f"{T},{_height:.6e},{_amp:.6e},{_center:.6f},{_fwhm:.6f},"
                                f"{_sig:.6f},{_slope:.6e},{_intercept:.6e},{_redchi:.6f},{_status}")
                        except Exception as _fe:
                            _fit_rows.append({"T": T, "height": None, "amplitude": None,
                                              "center": None, "fwhm": None, "slope": None,
                                              "redchi": None, "success": False,
                                              "status": str(_fe)[:40]})

                    if _fit_temps:
                        op_fit_json = _json_op.dumps({
                            "temps": _fit_temps,
                            "heights": _fit_heights,
                            "amplitudes": _fit_amps,
                            "colors": _fit_colors})
                    op_fit_table = _fit_rows
                    op_fit_csv = "\n".join(_fit_csv_lines)
                except ImportError:
                    op_error = (op_error or "") + " | lmfit not installed \u2014 run: pip install lmfit"
                except Exception as _pe:
                    op_error = (op_error or "") + f" | Peak fit error: {_pe}"

    except Exception as exc:
        op_error = f"{exc}\n{_tb_op.format_exc()}"

    op_elapsed = time.time() - _t0

    # ── re-render slice viewer with orderpar tab active ──
    _f0 = files[0] if files else ""
    content = render_template_string(SLICE_VIEWER_CONTENT,
        current=path, dirs=dirs, files=files,
        active_tab="orderpar",
        file_a=_f0, file_a_cmp=_f0,
        file_b_cmp=files[-1] if len(files) > 1 else _f0,
        multi_selected=[], L_values="0", cmap=cmap,
        slice_axis=slice_axis, autoscale=True, vmin=vmin, vmax=vmax,
        skew_angle=60, show_grid=show_grid,
        vmin_warning=None, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        rows=[], total_time=0.0, used_vmin=vmin, used_vmax=vmax, logs="",
        lc_file=_f0, lc_val=0.0, lc_npts=300,
        lc_mode="single", lc_compare=False, lc_autoscale=True,
        lc_file_b=_f0, lc_val_b=0.0,
        lc_files_multi=[], lc_val_multi=0.0,
        lc_x1=-1.0, lc_y1=0.0, lc_x2=4.0, lc_y2=0.0,
        lc_n1="Qh", lc_n2="Qk",
        lc_overlay=None, lc_profile_img=None,
        lc_overlay_b=None, lc_compare_img=None,
        lc_val_actual=0.0, lc_val_actual_b=0.0,
        lc_file_label_a="", lc_file_label_b="",
        lc_plotly_dist_a=None, lc_plotly_prof_a=None,
        lc_plotly_dist_b=None, lc_plotly_prof_b=None,
        lc_vmin_log=-4, lc_vmax_log=0,
        lc_elapsed=0.0, lc_error=None, lc_csv_data=None,
        lc_transform_path="entry/transform",
        lc_file_a_transform_path="entry/transform",
        lc_file_b_transform_path="entry/transform",
        lc_xaxis="Ql", lc_yaxis="None", lc_plot_lines=True,
        lc_ql_min=-1.0, lc_ql_max=1.0, lc_ql_lock=False,
        lc_qk_min=-5.0, lc_qk_max=5.0, lc_qk_lock=False,
        lc_qh_min=-5.0, lc_qh_max=5.0, lc_qh_lock=False,
        lc_proj_json=None, lc_proj_info=None,
        lc_multi_traces_json=None, lc_multi_overlays=[], lc_multi_items=[], lc_multi_csv=None,
        lc_show_overlays=False,
        lc_show_heatmap=False, lc_heatmap_json=None, lc_heatmap_ntemps=0,
        hm_autoscale=True, hm_vmin=0.0, hm_vmax=1.0,
        lc_show_peakfit=False, lc_fit_qmin=0.0, lc_fit_qmax=1.0, lc_fit_center=0.5,
        lc_fit_json=None, lc_fit_table=[], lc_fit_csv=None,
        lc_show_peakplot=False, lc_peakplot_json=None,
        tf_file=_f0, tf_transform_path="entry/transform",
        tf_signal="data", tf_qh="Qh", tf_qk="Qk", tf_ql="Ql",
        tf_slice_axis="Ql", tf_qls="0", tf_rotation=-45, tf_cmap="viridis",
        tf_mode="log", tf_autoscale=True,
        tf_vmin_pct=2.0, tf_vmax_pct=98.0, tf_vmin=0.0, tf_vmax=1.0,
        tf_maxpx=600, tf_axauto=True,
        tf_xmin=-5.0, tf_xmax=5.0, tf_ymin=-5.0, tf_ymax=5.0,
        tf_rows=None, tf_info=None, tf_error=None, tf_elapsed=None,
        # Order Parameter results
        op_files=op_files,
        op_transform_path=op_transform_path,
        op_qh_min=op_qh_min, op_qh_max=op_qh_max,
        op_qk_min=op_qk_min, op_qk_max=op_qk_max,
        op_ql_min=op_ql_min, op_ql_max=op_ql_max,
        op_cut_axis=op_cut_axis,
        op_int_auto=op_int_auto, op_int_ymin=op_int_ymin, op_int_ymax=op_int_ymax,
        op_int_tmin=op_int_tmin, op_int_tmax=op_int_tmax,
        op_lc_vmin=op_lc_vmin, op_lc_vmax=op_lc_vmax, op_lc_auto=op_lc_auto,
        op_show_heatmap=op_show_heatmap,
        op_hm_vmin=op_hm_vmin, op_hm_vmax=op_hm_vmax, op_hm_auto=op_hm_auto,
        op_do_fit=op_do_fit,
        op_fit_qmin=op_fit_qmin, op_fit_qmax=op_fit_qmax, op_fit_center=op_fit_center,
        op_fit_cmin=op_fit_cmin, op_fit_cmax=op_fit_cmax,
        op_error=op_error, op_scan_info=op_scan_info, op_elapsed=op_elapsed,
        op_integ_json=op_integ_json, op_integ_csv=op_integ_csv, op_ntemps=op_ntemps,
        op_traces_json=op_traces_json,
        op_heatmap_json=op_heatmap_json,
        op_fit_json=op_fit_json, op_fit_table=op_fit_table, op_fit_csv=op_fit_csv)
    return render_base(content, "slices")


@app.route("/slices/orderpar/csv", methods=["POST"])
def slices_orderpar_csv():
    """Download order parameter fit results as CSV."""
    csv_data = request.form.get("csv_data", "temperature_K,peak_height\n")
    return (csv_data.encode("utf-8"), 200,
            {"Content-Type": "text/csv",
             "Content-Disposition": "attachment; filename=order_parameter.csv"})


# ── LINECUT ROUTES ────────────────────────────────────────────────────────────
@app.route("/slices/linecut", methods=["GET","POST"])
def slices_linecut():
    path = request.values.get("path", ROOT)
    path = os.path.realpath(path)
    if not path.startswith(os.path.realpath(ROOT)):
        path = os.path.realpath(ROOT)

    dirs, files = [], []
    try:
        for item in sorted(os.listdir(path)):
            full = os.path.join(path, item)
            if os.path.isdir(full):
                dirs.append(full)
            elif item.endswith(".nxs") and os.path.isfile(full):
                files.append(full)
    except PermissionError:
        pass

    lc_file  = request.values.get("lc_file",  files[0] if files else "")
    lc_axis  = request.values.get("lc_axis",  "L")
    if lc_axis not in ("L","K","H"): lc_axis = "L"
    try:    lc_val  = float(request.values.get("lc_val",  "0"))
    except: lc_val  = 0.0
    try:    lc_x1   = float(request.values.get("lc_x1",  "-1"))
    except: lc_x1   = -1.0
    try:    lc_y1   = float(request.values.get("lc_y1",  "0"))
    except: lc_y1   = 0.0
    try:    lc_x2   = float(request.values.get("lc_x2",  "4"))
    except: lc_x2   = 4.0
    try:    lc_y2   = float(request.values.get("lc_y2",  "0"))
    except: lc_y2   = 0.0
    try:    lc_npts = int(request.values.get("lc_npts", "300"))
    except: lc_npts = 300
    cmap    = request.values.get("cmap", "inferno")
    try:    vmin    = float(request.values.get("vmin",   "0.0001"))
    except: vmin    = 1e-4
    try:    vmax    = float(request.values.get("vmax",   "2.4901"))
    except: vmax    = 2.4901

    cfg  = _SV_AXIS_CFG.get(lc_axis, _SV_AXIS_CFG["L"])
    dx1, dx2, dy1, dy2 = (cfg["xlim"][0], cfg["xlim"][1], cfg["ylim"][0], cfg["ylim"][1])
    try:    xmin = float(request.values.get("xmin", str(dx1)))
    except: xmin = float(dx1)
    try:    xmax = float(request.values.get("xmax", str(dx2)))
    except: xmax = float(dx2)
    try:    ymin = float(request.values.get("ymin", str(dy1)))
    except: ymin = float(dy1)
    try:    ymax = float(request.values.get("ymax", str(dy2)))
    except: ymax = float(dy2)

    _n1_map = {"L":"Qh","K":"Qh","H":"Qk"}
    _n2_map = {"L":"Qk","K":"Ql","H":"Ql"}
    n1 = _n1_map.get(lc_axis, "n1")
    n2 = _n2_map.get(lc_axis, "n2")

    overlay_b64 = profile_b64 = csv_data = None
    val_actual  = lc_val
    elapsed     = 0.0
    error       = None

    if request.method == "POST":
        if not lc_file or not os.path.isfile(lc_file):
            error = f"File not found: {lc_file!r}"
        elif not (lc_x1 != lc_x2 or lc_y1 != lc_y2):
            error = "Start and end points must be different."
        else:
            try:
                (overlay_b64, profile_b64,
                 dist, profile,
                 n1, n2, val_actual, elapsed, _logs) = sv_compute_linecut(
                    lc_file, lc_val, lc_axis,
                    lc_x1, lc_y1, lc_x2, lc_y2,
                    n_points=max(10, min(lc_npts, 2000)),
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    xlim=(xmin, xmax), ylim=(ymin, ymax))
                rows_csv = "\n".join(
                    f"{d:.6f},{p:.6g}"
                    for d, p in zip(dist, profile)
                    if np.isfinite(p))
                csv_data = (f"# file={os.path.basename(lc_file)}"
                            f"  axis={lc_axis}  val={val_actual:.4f}"
                            f"  ({n1},{n2}): ({lc_x1},{lc_y1})->({lc_x2},{lc_y2})\n"
                            f"distance_rlu,intensity\n{rows_csv}")
            except Exception as exc:
                import traceback
                error = f"{exc}\n{traceback.format_exc()}"

    content = render_template_string(
        LINECUT_CONTENT,
        path=path, current=path, dirs=dirs, files=files,
        lc_file=lc_file, lc_axis=lc_axis, lc_val=lc_val,
        lc_x1=lc_x1, lc_y1=lc_y1, lc_x2=lc_x2, lc_y2=lc_y2,
        lc_npts=lc_npts,
        cmap=cmap, vmin=vmin, vmax=vmax,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        overlay=overlay_b64, profile_img=profile_b64,
        n1=n1, n2=n2, val_actual=val_actual, elapsed=elapsed,
        error=error, csv_data=csv_data)
    return render_base(content, "linecut")


@app.route("/slices/linecut/csv", methods=["POST"])
def slices_linecut_csv():
    csv_data = request.form.get("csv_data", "")
    axis     = request.form.get("lc_axis", "L")
    val      = request.form.get("lc_val",  "0")
    fname    = f"linecut_{axis}{val}.csv".replace(" ", "_")
    return (csv_data.encode("utf-8"), 200,
            {"Content-Type": "text/csv",
             "Content-Disposition": f"attachment; filename={fname}"})


@app.route("/slices/linecut/fitcsv", methods=["POST"])
def slices_linecut_fitcsv():
    """Download Gaussian peak-fit results as CSV."""
    csv_data   = request.form.get("csv_data", "temperature_K,peak_height\n")
    axis       = request.form.get("slice_axis", "L")
    val        = request.form.get("lc_val_actual", "0")
    fname      = f"peak_fit_{axis}{val}.csv".replace(" ", "_")
    return (csv_data.encode("utf-8"), 200,
            {"Content-Type": "text/csv",
             "Content-Disposition": f"attachment; filename={fname}"})



# ── SAMPLE SEARCH TEMPLATE ────────────────────────────────────────────────────
SAMPLE_SEARCH_CONTENT = """
<style>
.ss-card{background:var(--card-bg);border:1px solid var(--border);border-radius:12px;padding:20px 24px;margin-bottom:20px;}
.ss-title{font-size:1.1rem;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.ss-root-badge{font-size:11px;background:rgba(99,179,237,.15);color:#63b3ed;border:1px solid rgba(99,179,237,.3);border-radius:6px;padding:2px 8px;font-family:monospace;}
.ss-tag{display:inline-flex;align-items:center;gap:4px;background:var(--accent);color:#fff;border-radius:20px;padding:3px 11px;font-size:13px;font-weight:700;}
.ss-tag-list{display:flex;flex-wrap:wrap;gap:6px;min-height:28px;align-items:center;}
.ss-results-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px;}
.ss-results-table th{background:var(--thead-bg,#2a3a5e);color:var(--thead-color,#c9d8ff);padding:8px 10px;text-align:left;position:sticky;top:0;z-index:1;}
.ss-results-table td{padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:middle;}
.ss-results-table tr:hover td{background:var(--row-hover,rgba(99,179,237,.07));}
.ss-badge-folder{background:#1a4a7a;color:#bee3f8;padding:2px 8px;border-radius:10px;font-size:11px;}
.ss-open-btn{font-size:12px;padding:3px 10px;border-radius:6px;border:1px solid var(--accent);color:var(--accent);background:transparent;cursor:pointer;white-space:nowrap;}
.ss-open-btn:hover{background:var(--accent);color:#fff;}
.ss-scroll{max-height:480px;overflow-y:auto;border:1px solid var(--border);border-radius:8px;}
/* ── Periodic Table ── */
.pt-outer{overflow-x:auto;padding-bottom:6px;margin-bottom:4px;}
.pt-grid{
  display:grid;
  grid-template-columns:repeat(18,38px);
  grid-template-rows:repeat(7,38px) 10px 38px 38px;
  gap:2px;
  min-width:706px;
}
.pt-cell{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  border-radius:4px;cursor:pointer;
  border:1.5px solid rgba(255,255,255,.15);
  transition:transform .1s,border-color .1s,box-shadow .1s;
  -webkit-user-select:none;user-select:none;
}
.pt-cell:hover{transform:scale(1.15);z-index:20;border-color:rgba(255,255,255,.8);box-shadow:0 2px 8px rgba(0,0,0,.6);}
.pt-cell.sel{background:#d97706 !important;color:#fff !important;border-color:#fbbf24 !important;box-shadow:0 0 0 2px #fbbf24;transform:scale(1.1);z-index:10;}
.pt-num{font-size:7px;line-height:1;opacity:.65;}
.pt-sym{font-size:14px;font-weight:800;line-height:1.1;}
.pt-ph{display:flex;align-items:center;justify-content:center;border-radius:4px;
  border:1px dashed rgba(255,255,255,.2);font-size:8px;color:rgba(255,255,255,.3);}
.pt-lbl{display:flex;align-items:center;justify-content:flex-end;padding-right:3px;
  font-size:8px;font-style:italic;color:rgba(255,255,255,.3);}
/* colours */
.el-H{background:#2b6cb0;color:#fff;}
.el-alkali{background:#c53030;color:#fff;}
.el-alkaline{background:#c05621;color:#fff;}
.el-transition{background:#2b4c8c;color:#fff;}
.el-post{background:#276749;color:#fff;}
.el-metalloid{background:#744210;color:#fff;}
.el-nonmetal{background:#276749;color:#fff;background:#22543d;}
.el-halogen{background:#285e61;color:#fff;}
.el-noble{background:#553c9a;color:#fff;}
.el-lanthanide{background:#822727;color:#fff;}
.el-actinide{background:#44337a;color:#fff;}
.el-unknown{background:#2d3748;color:#ccc;}
/* legend */
.pt-legend{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px;}
.pt-legend span{display:inline-flex;align-items:center;gap:4px;font-size:11px;color:var(--text-muted);}
.pt-legend i{display:inline-block;width:12px;height:12px;border-radius:2px;flex-shrink:0;}
/* tooltip */
.pt-tip{position:fixed;pointer-events:none;z-index:99999;
  background:#1a202c;color:#e2e8f0;border:1px solid #4a5568;border-radius:6px;
  padding:4px 10px;font-size:12px;white-space:nowrap;display:none;}
/* controls */
.ss-ctrl{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:12px;}
.ss-txt{flex:1;min-width:150px;padding:7px 11px;border:1px solid var(--border);
  border-radius:6px;background:var(--input-bg);color:var(--text);font-size:13px;}
</style>

<div id="pt-tip" class="pt-tip"></div>

<div class="ss-card">
  <div class="ss-title">⚗️ Sample Search by Element <span class="ss-root-badge">{{ root }}</span></div>

  <form method="POST" action="/sample_search">
    <input type="hidden" name="elements_raw" id="el-hidden" value="{{ elements_raw }}">
    <input type="hidden" name="search_in" value="folders">

    <!-- Periodic Table (pre-rendered HTML) -->
    <div class="pt-outer" id="pt-outer">
      <div class="pt-grid">
        {{ pt_html | safe }}
      </div>
    </div>

    <!-- Legend -->
    <div class="pt-legend">{{ pt_legend | safe }}</div>

    <!-- Selected tags -->
    <div style="margin-top:12px;">
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:5px;">
        Selected <span style="opacity:.6;">(click table above, or type below)</span>:
      </div>
      <div class="ss-tag-list" id="sel-tags">
        <span id="sel-none" style="font-size:12px;color:var(--text-muted);">None — click an element to start</span>
      </div>
    </div>

    <div class="ss-ctrl">
      <input class="ss-txt" type="text" id="el-txt"
             placeholder="or type: Fe Te Mn …"
             value="{{ elements_raw }}" autocomplete="off">
      <button type="button" id="btn-clear"
              style="padding:7px 14px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text-muted);cursor:pointer;">
        ✕ Clear
      </button>
      <button type="submit" class="btn-primary" style="padding:7px 22px;">🔍 Search</button>
    </div>
  </form>
</div>

{% if searched %}
<div class="ss-card">
  <div class="ss-title">
    Results
    <span style="font-size:13px;font-weight:400;color:var(--text-muted);">
      {% if results %}
        {{ results|length }} folder{{ 's' if results|length != 1 }} found for
        {% for el in elements %}<span class="ss-tag" style="font-size:12px;padding:2px 9px;">{{ el }}</span> {% endfor %}
      {% elif error %}
        error
      {% else %}
        no matches for
        {% for el in elements %}<span class="ss-tag" style="font-size:12px;padding:2px 9px;">{{ el }}</span> {% endfor %}
      {% endif %}
    </span>
  </div>
  {% if error %}
    <div style="color:#fc8181;background:rgba(252,129,129,.1);border-radius:6px;padding:10px 14px;font-size:13px;">{{ error }}</div>
  {% elif results %}
  <div class="ss-scroll">
    <table class="ss-results-table">
      <thead><tr><th>#</th><th>Sample Folder</th><th>Full Path</th><th>Open</th></tr></thead>
      <tbody>
      {% for r in results %}
        <tr>
          <td style="color:var(--text-muted);">{{ loop.index }}</td>
          <td style="font-weight:700;font-family:monospace;font-size:13px;">📁 {{ r.name }}</td>
          <td style="font-family:monospace;font-size:11px;word-break:break-all;color:var(--text-muted);">{{ r.path }}</td>
          <td><a href="/slices?path={{ r.path | urlencode }}"><button class="ss-open-btn">📂 Browse</button></a></td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <p style="color:var(--text-muted);font-size:14px;">No sample folders found under <code>{{ root }}</code>.</p>
  {% endif %}
</div>
{% endif %}

<script>
(function(){
  /* ─── grab DOM refs with null guards ─── */
  var elHidden = document.getElementById('el-hidden');
  var elTxt    = document.getElementById('el-txt');
  var elTags   = document.getElementById('sel-tags');
  var elOuter  = document.getElementById('pt-outer');
  var elTip    = document.getElementById('pt-tip');
  var elClear  = document.getElementById('btn-clear');

  if(!elHidden || !elTxt || !elTags || !elOuter){ return; } /* abort if DOM missing */

  /* ─── selection state ─── */
  var sel = {};

  /* pre-load any server-returned selection (after a search POST) */
  var initRaw = elHidden.value.trim();
  if(initRaw){
    initRaw.split(/[ ,;]+/).forEach(function(s){ if(s) sel[s] = true; });
  }

  /* ─── sync helpers ─── */
  function getCells(){ return elOuter.querySelectorAll('.pt-cell'); }

  function highlightCells(){
    getCells().forEach(function(c){
      var s = c.getAttribute('data-sym');
      if(sel[s]){
        c.classList.add('sel');
      } else {
        c.classList.remove('sel');
      }
    });
  }

  function refreshHidden(){
    var keys = Object.keys(sel);
    elHidden.value = keys.join(' ');
    elTxt.value    = keys.join(' ');
  }

  function refreshTags(){
    var keys = Object.keys(sel);
    elTags.innerHTML = '';
    if(keys.length === 0){
      var msg = document.createElement('span');
      msg.style.cssText = 'font-size:12px;color:#888;';
      msg.textContent = 'None — click an element above to start';
      elTags.appendChild(msg);
    } else {
      keys.forEach(function(sym){
        var sp = document.createElement('span');
        sp.className = 'ss-tag';
        sp.setAttribute('data-tag', sym);
        var rm = document.createElement('span');
        rm.style.cssText = 'cursor:pointer;opacity:.7;font-size:11px;margin-left:3px;';
        rm.textContent = '✕';
        rm.setAttribute('data-rm', sym);
        sp.textContent = sym;
        sp.appendChild(rm);
        elTags.appendChild(sp);
      });
    }
  }

  function fullRefresh(){
    highlightCells();
    refreshHidden();
    refreshTags();
  }

  /* ─── toggle a symbol ─── */
  function ptToggle(sym){
    if(!sym) return;
    if(sel[sym]){ delete sel[sym]; } else { sel[sym] = true; }
    fullRefresh();
  }

  /* ─── CLICK on periodic table (event delegation on container) ─── */
  elOuter.addEventListener('click', function(ev){
    var el = ev.target;
    /* walk up to find the .pt-cell */
    while(el && el !== elOuter){
      if(el.getAttribute('data-sym')){
        ptToggle(el.getAttribute('data-sym'));
        return;
      }
      el = el.parentElement;
    }
  });

  /* ─── CLICK on tag ✕ buttons (delegation on tag list) ─── */
  elTags.addEventListener('click', function(ev){
    var sym = ev.target.getAttribute('data-rm');
    if(sym){ delete sel[sym]; fullRefresh(); }
  });

  /* ─── TOOLTIP via delegation ─── */
  if(elTip){
    elOuter.addEventListener('mouseover', function(ev){
      var el = ev.target;
      while(el && el !== elOuter){
        var sym = el.getAttribute('data-sym');
        if(sym){
          var z    = el.getAttribute('data-z')    || '';
          var name = el.getAttribute('data-name') || sym;
          elTip.textContent = z + '  ' + name + '  (' + sym + ')';
          elTip.style.display = 'block';
          return;
        }
        el = el.parentElement;
      }
      elTip.style.display = 'none';
    });
    elOuter.addEventListener('mousemove', function(ev){
      var x = ev.clientX + 14, y = ev.clientY - 32;
      if(x + 220 > window.innerWidth) x = ev.clientX - 224;
      elTip.style.left = x + 'px';
      elTip.style.top  = y + 'px';
    });
    elOuter.addEventListener('mouseleave', function(){
      elTip.style.display = 'none';
    });
  }

  /* ─── CLEAR button ─── */
  if(elClear){
    elClear.addEventListener('click', function(){
      sel = {};
      fullRefresh();
    });
  }

  /* ─── MANUAL TEXT INPUT ─── */
  elTxt.addEventListener('input', function(){
    sel = {};
    elTxt.value.split(/[ ,;]+/).forEach(function(s){
      s = s.trim();
      if(s) sel[s] = true;
    });
    highlightCells();
    refreshHidden();
    /* rebuild tags without touching elTxt.value */
    var keys = Object.keys(sel);
    elTags.innerHTML = '';
    if(keys.length === 0){
      var msg = document.createElement('span');
      msg.style.cssText = 'font-size:12px;color:#888;';
      msg.textContent = 'None — click an element above to start';
      elTags.appendChild(msg);
    } else {
      keys.forEach(function(sym){
        var sp = document.createElement('span'); sp.className = 'ss-tag';
        sp.setAttribute('data-tag', sym);
        var rm = document.createElement('span');
        rm.style.cssText = 'cursor:pointer;opacity:.7;font-size:11px;margin-left:3px;';
        rm.textContent = '✕'; rm.setAttribute('data-rm', sym);
        sp.textContent = sym; sp.appendChild(rm);
        elTags.appendChild(sp);
      });
    }
  });

  /* ─── initial highlight (after a POST search) ─── */
  fullRefresh();

})();
</script>
"""


_PT_ELEMENTS = [
    # (Z, symbol, name, row, col, category)
    (1,'H','Hydrogen',1,1,'H'),
    (2,'He','Helium',1,18,'noble'),
    (3,'Li','Lithium',2,1,'alkali'),
    (4,'Be','Beryllium',2,2,'alkaline'),
    (5,'B','Boron',2,13,'metalloid'),
    (6,'C','Carbon',2,14,'nonmetal'),
    (7,'N','Nitrogen',2,15,'nonmetal'),
    (8,'O','Oxygen',2,16,'nonmetal'),
    (9,'F','Fluorine',2,17,'halogen'),
    (10,'Ne','Neon',2,18,'noble'),
    (11,'Na','Sodium',3,1,'alkali'),
    (12,'Mg','Magnesium',3,2,'alkaline'),
    (13,'Al','Aluminum',3,13,'post'),
    (14,'Si','Silicon',3,14,'metalloid'),
    (15,'P','Phosphorus',3,15,'nonmetal'),
    (16,'S','Sulfur',3,16,'nonmetal'),
    (17,'Cl','Chlorine',3,17,'halogen'),
    (18,'Ar','Argon',3,18,'noble'),
    (19,'K','Potassium',4,1,'alkali'),
    (20,'Ca','Calcium',4,2,'alkaline'),
    (21,'Sc','Scandium',4,3,'transition'),
    (22,'Ti','Titanium',4,4,'transition'),
    (23,'V','Vanadium',4,5,'transition'),
    (24,'Cr','Chromium',4,6,'transition'),
    (25,'Mn','Manganese',4,7,'transition'),
    (26,'Fe','Iron',4,8,'transition'),
    (27,'Co','Cobalt',4,9,'transition'),
    (28,'Ni','Nickel',4,10,'transition'),
    (29,'Cu','Copper',4,11,'transition'),
    (30,'Zn','Zinc',4,12,'transition'),
    (31,'Ga','Gallium',4,13,'post'),
    (32,'Ge','Germanium',4,14,'metalloid'),
    (33,'As','Arsenic',4,15,'metalloid'),
    (34,'Se','Selenium',4,16,'nonmetal'),
    (35,'Br','Bromine',4,17,'halogen'),
    (36,'Kr','Krypton',4,18,'noble'),
    (37,'Rb','Rubidium',5,1,'alkali'),
    (38,'Sr','Strontium',5,2,'alkaline'),
    (39,'Y','Yttrium',5,3,'transition'),
    (40,'Zr','Zirconium',5,4,'transition'),
    (41,'Nb','Niobium',5,5,'transition'),
    (42,'Mo','Molybdenum',5,6,'transition'),
    (43,'Tc','Technetium',5,7,'transition'),
    (44,'Ru','Ruthenium',5,8,'transition'),
    (45,'Rh','Rhodium',5,9,'transition'),
    (46,'Pd','Palladium',5,10,'transition'),
    (47,'Ag','Silver',5,11,'transition'),
    (48,'Cd','Cadmium',5,12,'transition'),
    (49,'In','Indium',5,13,'post'),
    (50,'Sn','Tin',5,14,'post'),
    (51,'Sb','Antimony',5,15,'metalloid'),
    (52,'Te','Tellurium',5,16,'metalloid'),
    (53,'I','Iodine',5,17,'halogen'),
    (54,'Xe','Xenon',5,18,'noble'),
    (55,'Cs','Cesium',6,1,'alkali'),
    (56,'Ba','Barium',6,2,'alkaline'),
    (72,'Hf','Hafnium',6,4,'transition'),
    (73,'Ta','Tantalum',6,5,'transition'),
    (74,'W','Tungsten',6,6,'transition'),
    (75,'Re','Rhenium',6,7,'transition'),
    (76,'Os','Osmium',6,8,'transition'),
    (77,'Ir','Iridium',6,9,'transition'),
    (78,'Pt','Platinum',6,10,'transition'),
    (79,'Au','Gold',6,11,'transition'),
    (80,'Hg','Mercury',6,12,'transition'),
    (81,'Tl','Thallium',6,13,'post'),
    (82,'Pb','Lead',6,14,'post'),
    (83,'Bi','Bismuth',6,15,'post'),
    (84,'Po','Polonium',6,16,'post'),
    (85,'At','Astatine',6,17,'halogen'),
    (86,'Rn','Radon',6,18,'noble'),
    (87,'Fr','Francium',7,1,'alkali'),
    (88,'Ra','Radium',7,2,'alkaline'),
    (104,'Rf','Rutherfordium',7,4,'transition'),
    (105,'Db','Dubnium',7,5,'transition'),
    (106,'Sg','Seaborgium',7,6,'transition'),
    (107,'Bh','Bohrium',7,7,'transition'),
    (108,'Hs','Hassium',7,8,'transition'),
    (109,'Mt','Meitnerium',7,9,'unknown'),
    (110,'Ds','Darmstadtium',7,10,'unknown'),
    (111,'Rg','Roentgenium',7,11,'unknown'),
    (112,'Cn','Copernicium',7,12,'unknown'),
    (113,'Nh','Nihonium',7,13,'unknown'),
    (114,'Fl','Flerovium',7,14,'unknown'),
    (115,'Mc','Moscovium',7,15,'unknown'),
    (116,'Lv','Livermorium',7,16,'unknown'),
    (117,'Ts','Tennessine',7,17,'unknown'),
    (118,'Og','Oganesson',7,18,'unknown'),
    # Lanthanides — row 9, cols 3–17
    (57,'La','Lanthanum',9,3,'lanthanide'),
    (58,'Ce','Cerium',9,4,'lanthanide'),
    (59,'Pr','Praseodymium',9,5,'lanthanide'),
    (60,'Nd','Neodymium',9,6,'lanthanide'),
    (61,'Pm','Promethium',9,7,'lanthanide'),
    (62,'Sm','Samarium',9,8,'lanthanide'),
    (63,'Eu','Europium',9,9,'lanthanide'),
    (64,'Gd','Gadolinium',9,10,'lanthanide'),
    (65,'Tb','Terbium',9,11,'lanthanide'),
    (66,'Dy','Dysprosium',9,12,'lanthanide'),
    (67,'Ho','Holmium',9,13,'lanthanide'),
    (68,'Er','Erbium',9,14,'lanthanide'),
    (69,'Tm','Thulium',9,15,'lanthanide'),
    (70,'Yb','Ytterbium',9,16,'lanthanide'),
    (71,'Lu','Lutetium',9,17,'lanthanide'),
    # Actinides — row 10, cols 3–17
    (89,'Ac','Actinium',10,3,'actinide'),
    (90,'Th','Thorium',10,4,'actinide'),
    (91,'Pa','Protactinium',10,5,'actinide'),
    (92,'U','Uranium',10,6,'actinide'),
    (93,'Np','Neptunium',10,7,'actinide'),
    (94,'Pu','Plutonium',10,8,'actinide'),
    (95,'Am','Americium',10,9,'actinide'),
    (96,'Cm','Curium',10,10,'actinide'),
    (97,'Bk','Berkelium',10,11,'actinide'),
    (98,'Cf','Californium',10,12,'actinide'),
    (99,'Es','Einsteinium',10,13,'actinide'),
    (100,'Fm','Fermium',10,14,'actinide'),
    (101,'Md','Mendelevium',10,15,'actinide'),
    (102,'No','Nobelium',10,16,'actinide'),
    (103,'Lr','Lawrencium',10,17,'actinide'),
]

_PT_CATS = [
    ('H',          'Hydrogen',         '#2b6cb0'),
    ('alkali',     'Alkali metals',    '#c53030'),
    ('alkaline',   'Alkaline earth',   '#c05621'),
    ('transition', 'Transition metals','#2b4c8c'),
    ('post',       'Post-transition',  '#276749'),
    ('metalloid',  'Metalloids',       '#744210'),
    ('nonmetal',   'Nonmetals',        '#22543d'),
    ('halogen',    'Halogens',         '#285e61'),
    ('noble',      'Noble gases',      '#553c9a'),
    ('lanthanide', 'Lanthanides',      '#822727'),
    ('actinide',   'Actinides',        '#44337a'),
    ('unknown',    'Unknown',          '#2d3748'),
]

def _build_pt_html(selected_syms=()):
    """Return pre-rendered HTML for the CSS-grid periodic table."""
    sel_set = set(selected_syms)
    parts = []
    # Placeholder cells at (row6,col3) and (row7,col3)
    for row, col, label in [(6, 3, '57–71'), (7, 3, '89–103')]:
        parts.append(
            f'<div class="pt-ph" style="grid-row:{row};grid-column:{col}">{label}</div>'
        )
    # F-block row labels
    parts.append('<div class="pt-lbl" style="grid-row:9;grid-column:1/3">Lanthanides →</div>')
    parts.append('<div class="pt-lbl" style="grid-row:10;grid-column:1/3">Actinides →</div>')
    # Element cells — NO inline event handlers; all handled via JS delegation
    for z, sym, name, row, col, cat in _PT_ELEMENTS:
        sel_cls = ' sel' if sym in sel_set else ''
        parts.append(
            f'<div class="pt-cell el-{cat}{sel_cls}" '
            f'style="grid-row:{row};grid-column:{col}" '
            f'data-sym="{sym}" data-z="{z}" data-name="{name}">'
            f'<span class="pt-num">{z}</span>'
            f'<span class="pt-sym">{sym}</span>'
            f'</div>'
        )
    return ''.join(parts)


def _build_pt_legend():
    parts = []
    for cat, label, color in _PT_CATS:
        parts.append(
            f'<span><i style="background:{color};border:1px solid rgba(255,255,255,.2);'
            f'border-radius:2px;"></i>{label}</span>'
        )
    return ''.join(parts)


def _search_samples_by_elements(root, elements, search_in="both"):
    """
    Walk `root` recursively.  For each folder name or .nxs filename,
    check whether it contains ALL supplied element symbols as
    case-sensitive substrings (e.g. "Fe" and "Te" both in "FeTe2").

    Returns list of dicts: {name, path, type}
    """
    import re
    results = []
    if not elements:
        return results

    # Build a simple "all-element" checker
    def _matches(name):
        return all(el in name for el in elements)

    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # --- check folder names ---
            if search_in in ("both", "folders"):
                for dname in dirnames:
                    if _matches(dname):
                        results.append({
                            "name": dname,
                            "path": os.path.join(dirpath, dname),
                            "type": "folder",
                        })
            # --- check .nxs filenames ---
            if search_in in ("both", "files"):
                for fname in filenames:
                    if fname.endswith(".nxs") and _matches(os.path.splitext(fname)[0]):
                        results.append({
                            "name": fname,
                            "path": os.path.join(dirpath, fname),
                            "type": "nxs_file",
                        })
    except Exception:
        pass

    # sort: folders first, then by name
    results.sort(key=lambda r: (0 if r["type"] == "folder" else 1, r["name"]))
    return results


@app.route("/sample_search", methods=["GET", "POST"])
def sample_search_page():
    from urllib.parse import quote as _urlencode
    elements_raw = ""
    elements     = []
    search_in    = "folders"   # always folders only
    results      = []
    error        = ""
    searched     = False

    if request.method == "POST":
        elements_raw = request.form.get("elements_raw", "").strip()
        searched     = True

        # parse element list: split on commas, spaces, semicolons
        import re
        raw_parts = re.split(r"[,\s;]+", elements_raw)
        elements  = [p.strip() for p in raw_parts if p.strip()]

        if not elements:
            error = "Please click at least one element on the periodic table."
        else:
            try:
                results = _search_samples_by_elements(ROOT, elements, "folders")
            except Exception as exc:
                import traceback
                error = f"Search error: {exc}\n{traceback.format_exc()}"

    content = render_template_string(
        SAMPLE_SEARCH_CONTENT,
        root=ROOT,
        elements_raw=elements_raw,
        elements=elements,
        results=results,
        error=error,
        searched=searched,
        urlencode=_urlencode,
        pt_html=_build_pt_html(elements),
        pt_legend=_build_pt_legend(),
    )
    return render_base(content, "sample_search")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
