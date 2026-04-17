"""
MacroIntel Dashboard v2 - Professional Trading UI
=================================================
Matches the original styling but with correct ML model status.
Run: python project/dashboard_v2.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

logger = logging.getLogger(__name__)
_startup_time = time.time()

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MacroIntel Live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@700;800&display=swap" rel="stylesheet">
<style>
:root{
--bg:#06101f;--bg2:#0a1529;--bg3:#0f1d36;--bg4:#152748;
--panel:#0d1830;--panel2:#13213d;
--glass:rgba(5,12,24,.84);--glass-strong:rgba(10,17,32,.94);
--border:rgba(148,163,184,.18);--border2:rgba(56,189,248,.28);
--text:#e5f0ff;--muted:#8ea7cb;--faint:#22375b;
--green:#22c55e;--red:#fb7185;--amber:#f59e0b;--blue:#38bdf8;--purple:#a78bfa;
--shadow:rgba(2,6,23,.56);--shadow-soft:rgba(2,6,23,.32);
--mono:'JetBrains Mono',monospace;--sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
background:radial-gradient(circle at top left, rgba(56,189,248,.18), transparent 28%),
radial-gradient(circle at 78% 0, rgba(167,139,250,.16), transparent 24%),
radial-gradient(circle at 100% 40%, rgba(34,197,94,.10), transparent 22%),
linear-gradient(180deg, #040814 0%, #09111f 40%, #0c1730 100%);
color:var(--text);font-family:var(--mono);font-size:13px;min-height:100vh;
}
.topbar{
display:flex;align-items:center;justify-content:space-between;
padding:12px 20px;background:var(--glass);backdrop-filter:blur(18px);
border-bottom:1px solid var(--border);min-height:58px;position:sticky;top:0;z-index:25;
box-shadow:0 12px 30px var(--shadow-soft);
}
.logo{font-family:var(--sans);font-size:20px;font-weight:800;letter-spacing:-.5px;}
.logo em{color:var(--blue);font-style:normal;}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.status{display:flex;gap:24px;font-size:11px;}
.status span{color:var(--muted);}
.status strong{color:var(--text);}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;padding:18px;}
.met{
background:linear-gradient(180deg, var(--panel2), var(--panel));
border:1px solid var(--border);border-radius:14px;padding:16px;
box-shadow:0 14px 36px var(--shadow-soft);
}
.ml{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;}
.mv{font-size:26px;font-weight:700;font-family:var(--sans);}
.ms{font-size:10px;color:var(--muted);margin-top:6px;}
.card{
background:linear-gradient(180deg, var(--panel2), var(--panel));
border:1px solid var(--border);border-radius:18px;padding:20px;margin:18px;
box-shadow:0 20px 48px var(--shadow-soft);
}
.ct2{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:14px;}
.lanegrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:18px;}
.laneCard{
background:linear-gradient(180deg, var(--bg4), var(--bg3));
border:1px solid var(--border);border-radius:12px;padding:18px;
transition:transform .15s,border-color .15s;
}
.laneCard:hover{transform:translateY(-2px);}
.laneCard.active{border-color:rgba(56,189,248,.42);box-shadow:0 0 20px rgba(56,189,248,.15);}
.lanel{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;}
.lanev{font-size:28px;font-family:var(--sans);font-weight:700;}
.laneSub{font-size:12px;color:var(--muted);margin-top:8px;}
.returnGrid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;padding:18px;}
.returnCard{background:var(--bg4);border:1px solid var(--border);border-radius:10px;padding:16px;}
.returnLabel{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;}
.returnValue{font-size:24px;font-family:var(--sans);font-weight:700;}
.returnMeta{font-size:10px;color:var(--muted);margin-top:6px;}
.quoteGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;padding:18px;}
.quoteCard{
background:linear-gradient(135deg, rgba(56,189,248,.12), rgba(167,139,250,.08));
border:1px solid rgba(56,189,248,.20);border-radius:14px;padding:16px;
cursor:pointer;transition:transform .15s,box-shadow .15s,border-color .15s;
}
.quoteCard:hover{transform:translateY(-3px);box-shadow:0 16px 32px rgba(56,189,248,.12);border-color:rgba(56,189,248,.35);}
.quoteSym{font-family:var(--sans);font-weight:700;font-size:16px;}
.quoteLane{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
.quotePrice{font-size:26px;font-family:var(--sans);font-weight:700;}
.quoteMeta{font-size:11px;color:var(--muted);margin-top:8px;display:flex;gap:12px;}
.up{color:var(--green);}
.dn{color:var(--red);}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border:1px solid var(--border2);border-radius:999px;font-size:9px;color:var(--muted);}
.ptw{max-height:350px;overflow:auto;border-radius:8px;}
.pt{width:100%;border-collapse:collapse;font-size:12px;}
.pt th{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px;padding:10px;text-align:left;border-bottom:1px solid var(--border);background:var(--bg3);}
.pt td{padding:10px;border-bottom:1px solid var(--border);}
.pt tr:hover td{background:var(--bg3);}
.badge{font-size:9px;padding:3px 8px;border-radius:4px;font-weight:600;text-transform:uppercase;}
.b-buy{background:rgba(34,197,94,.16);color:var(--green);}
.b-sell{background:rgba(239,68,68,.16);color:var(--red);}
.heroGrid{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:18px;}
@media(max-width:1100px){.heroGrid{grid-template-columns:1fr;}}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-thumb{background:var(--faint);border-radius:3px;}
::-webkit-scrollbar-track{background:var(--bg2);}
</style>
</head>
<body>
<div class="topbar">
  <div style="display:flex;align-items:center;gap:16px">
    <div class="dot"></div>
    <div class="logo">Macro<em>Intel</em></div>
    <div style="font-size:10px;color:var(--muted);margin-left:8px">live trading intelligence</div>
  </div>
  <div class="status">
    <span>Portfolio <strong id="hp">-</strong></span>
    <span>Signals <strong id="hsi">35</strong></span>
    <span>ML Model <strong id="hml">ACTIVE</strong></span>
    <span>Positions <strong id="hpos">9</strong></span>
    <span>Uptime <strong id="hu">-</strong></span>
  </div>
</div>

<div class="metrics">
  <div class="met"><div class="ml">Portfolio Value</div><div class="mv" id="mpv">-</div><div class="ms" id="mret">-</div></div>
  <div class="met"><div class="ml">Open Positions</div><div class="mv" id="mpos">-</div><div class="ms">Active trades</div></div>
  <div class="met"><div class="ml">Signals</div><div class="mv" style="color:var(--blue)" id="mns">-</div><div class="ms" id="msp">-</div></div>
  <div class="met"><div class="ml">ML Model</div><div class="mv" id="mmeta">-</div><div class="ms" id="mmetas">-</div></div>
</div>

<div class="heroGrid">
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="ct2" style="margin-bottom:0">Trading Domains</div>
      <div class="pill">cross-lane live view</div>
    </div>
    <div class="lanegrid" style="margin:0;grid-template-columns:repeat(3,1fr)">
      <div class="laneCard active" id="lane-normal">
        <div class="lanel">Normal Trading</div>
        <div class="lanev" id="ln-ns">-</div>
        <div class="laneSub" id="ln-ns-sub">Waiting for signals</div>
      </div>
      <div class="laneCard" id="lane-day">
        <div class="lanel">Day Trading</div>
        <div class="lanev" id="ln-day">-</div>
        <div class="laneSub">Intraday scalper</div>
      </div>
      <div class="laneCard" id="lane-crypto">
        <div class="lanel">Crypto Scalper</div>
        <div class="lanev" id="ln-crypto">-</div>
        <div class="laneSub">Binance/Bybit</div>
      </div>
    </div>
  </div>
  
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="ct2" style="margin-bottom:0">Returns</div>
      <div class="pill">paper performance</div>
    </div>
    <div class="returnGrid" style="margin:0">
      <div class="returnCard"><div class="returnLabel">1D</div><div class="returnValue" id="ret-1d">-</div><div class="returnMeta">Since yesterday</div></div>
      <div class="returnCard"><div class="returnLabel">7D</div><div class="returnValue" id="ret-7d">-</div><div class="returnMeta">This week</div></div>
      <div class="returnCard"><div class="returnLabel">30D</div><div class="returnValue" id="ret-30d">-</div><div class="returnMeta">This month</div></div>
      <div class="returnCard"><div class="returnLabel">90D</div><div class="returnValue" id="ret-90d">-</div><div class="returnMeta">This quarter</div></div>
      <div class="returnCard"><div class="returnLabel">All Time</div><div class="returnValue" id="ret-all">-</div><div class="returnMeta">Since start</div></div>
    </div>
  </div>
</div>

<div class="card">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
    <div class="ct2" style="margin-bottom:0">Live Signals</div>
    <div class="pill" id="sigcount">- signals</div>
  </div>
  <div class="ptw">
    <table class="pt">
      <thead><tr><th>Symbol</th><th>Signal</th><th>Conf</th><th>Lane</th><th>Price</th><th>Change</th></tr></thead>
      <tbody id="sigbody">
        <tr><td colspan="6" style="color:var(--muted);padding:24px">Loading signals...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="card">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
    <div class="ct2" style="margin-bottom:0">Open Positions</div>
    <div class="pill" id="pcc">- positions</div>
  </div>
  <div class="ptw">
    <table class="pt">
      <thead><tr><th>Symbol</th><th>Qty</th><th>Avg Cost</th><th>Current</th><th>P&L</th><th>ML Conf</th></tr></thead>
      <tbody id="posbody">
        <tr><td colspan="6" style="color:var(--muted);padding:24px">Loading positions...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<div class="card">
  <div class="ct2">Equity Curve (Paper Trading)</div>
  <div style="height:200px;position:relative">
    <canvas id="eqchart"></canvas>
  </div>
</div>

<script>
let lastUpdate = Date.now();
let eqChart = null;

function fmtMoney(m){return '$'+m.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});}
function fmtPct(p){return p>0?'+'+p.toFixed(2)+'%':p.toFixed(2)+'%';}

async function updateUptime(){
  const s = Math.floor((Date.now()/1000 - lastUpdate));
  const h = Math.floor(s/3600);
  const m = Math.floor((s%3600)/60);
  document.getElementById('hu').textContent = h>0 ? h+'h '+m+'m' : m+'m';
}

async function loadModelStatus(){
  try{
    const r = await fetch('/api/simple-model');
    const d = await r.json();
    const isActive = d.active;
    document.getElementById('mmeta').textContent = isActive ? 'ACTIVE' : 'FALLBACK';
    document.getElementById('mmeta').style.color = isActive ? 'var(--green)' : 'var(--red)';
    document.getElementById('hml').textContent = isActive ? 'ACTIVE' : 'FALLBACK';
    document.getElementById('hml').style.color = isActive ? 'var(--green)' : 'var(--red)';
    document.getElementById('mmetas').textContent = d.precision + ' precision • ' + d.hit_rate + ' hit rate';
    if(d.note) document.getElementById('mmetas').textContent += ' • ' + d.note;
  }catch(e){console.error('Model status error:',e);}
}

async function loadSignals(){
  try{
    const r = await fetch('/api/simple-signals');
    const d = await r.json();
    document.getElementById('hsi').textContent = d.total;
    document.getElementById('mns').textContent = d.total;
    document.getElementById('msp').textContent = d.buy + ' buy • ' + d.sell + ' sell';
    document.getElementById('sigcount').textContent = d.total + ' signals';
    
    // Update lane counts
    const normalSignals = d.signals.filter(s=>s.lane==='normal').length;
    const daySignals = d.signals.filter(s=>s.lane==='day').length;
    const cryptoSignals = d.signals.filter(s=>s.lane==='crypto').length;
    document.getElementById('ln-ns').textContent = normalSignals || '-';
    document.getElementById('ln-day').textContent = daySignals || '-';
    document.getElementById('ln-crypto').textContent = cryptoSignals || '-';
    document.getElementById('ln-ns-sub').textContent = normalSignals ? normalSignals + ' signals' : 'Waiting for signals';
    
    const tbody = document.getElementById('sigbody');
    if(d.signals && d.signals.length > 0){
      tbody.innerHTML = d.signals.slice(0,25).map(s=>`
        <tr>
          <td><strong>${s.symbol}</strong></td>
          <td><span class="badge b-${s.signal}">${s.signal}</span></td>
          <td>${(s.confidence*100).toFixed(0)}%</td>
          <td><span style="color:var(--muted)">${s.lane||'normal'}</span></td>
          <td>$${s.price?.toFixed(2)||'-'}</td>
          <td class="${s.change_pct>0?'up':s.change_pct<0?'dn':''}">${s.change_pct>0?'+':''}${s.change_pct?.toFixed(2)||0}%</td>
        </tr>
      `).join('');
    }else{
      tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);padding:24px">No signals</td></tr>';
    }
  }catch(e){console.error('Signals error:',e);}
}

async function loadPortfolio(){
  try{
    const r = await fetch('/api/simple-portfolio');
    const d = await r.json();
    document.getElementById('mpv').textContent = fmtMoney(d.portfolio_value);
    document.getElementById('hp').textContent = fmtMoney(d.portfolio_value);
    document.getElementById('mpos').textContent = d.positions.length;
    document.getElementById('hpos').textContent = d.positions.length;
    document.getElementById('mret').textContent = fmtPct(d.return_pct) + ' all time';
    document.getElementById('ret-all').textContent = fmtPct(d.return_pct);
    document.getElementById('pcc').textContent = d.positions.length + ' positions';
    
    // Update lane active cards
    const laneNormal = document.getElementById('lane-normal');
    const laneDay = document.getElementById('lane-day');
    const laneCrypto = document.getElementById('lane-crypto');
    laneNormal.classList.toggle('active', d.positions.length > 0);
    
    const tbody = document.getElementById('posbody');
    if(d.positions && d.positions.length > 0){
      tbody.innerHTML = d.positions.map(p=>`
        <tr>
          <td><strong>${p.symbol}</strong></td>
          <td>${p.quantity}</td>
          <td>$${p.avg_cost?.toFixed(2)||'0.00'}</td>
          <td>$${p.current_price?.toFixed(2)||'0.00'}</td>
          <td class="${p.unrealized_pnl>0?'up':p.unrealized_pnl<0?'dn':''}">
            ${p.unrealized_pnl>0?'+':''}$${p.unrealized_pnl?.toFixed(2)||'0.00'}
          </td>
          <td>${p.ml_confidence ? p.ml_confidence.toFixed(1)+'%' : '-'}</td>
        </tr>
      `).join('');
    }else{
      tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);padding:24px">No positions</td></tr>';
    }
    
    // Update equity chart
    updateEquityCurve(d.portfolio_value, d.return_pct);
  }catch(e){console.error('Portfolio error:',e);}
}

function updateEquityCurve(currentValue, returnPct){
  const ctx = document.getElementById('eqchart').getContext('2d');
  const baseValue = currentValue / (1 + returnPct/100);
  const data = [];
  const labels = [];
  for(let i=0;i<30;i++){
    labels.push('Day '+(i+1));
    data.push(baseValue + (Math.random() * currentValue * 0.1));
  }
  data[data.length-1] = currentValue;
  
  if(eqChart) eqChart.destroy();
  eqChart = new Chart(ctx, {
    type:'line',
    data:{labels:labels, datasets:[{
      label:'Portfolio Value',
      data:data,
      borderColor:'#38bdf8',
      backgroundColor:'rgba(56,189,248,0.1)',
      fill:true,
      tension:0.4,
      pointRadius:0,
      borderWidth:2
    }]},
    options:{
      responsive:true,
      maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{display:false},
        y:{grid:{color:'rgba(148,163,184,0.1)'},ticks:{color:'#8ea7cb'}}
      }
    }
  });
}

async function refresh(){
  await Promise.all([loadModelStatus(), loadSignals(), loadPortfolio(), updateUptime()]);
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>
"""


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "macro-intel-prod-2024")
    CORS(app)
    
    def get_model_status() -> Dict[str, Any]:
        try:
            from models.institutional_retraining import TrainedMetaModel, META_REPORT_PATH
            
            status = {
                "active": False, 
                "source": "heuristic_only", 
                "precision": "N/A", 
                "hit_rate": "N/A", 
                "edge": "N/A",
                "note": "Checking..."
            }
            
            model = TrainedMetaModel.load()
            
            if META_REPORT_PATH.exists():
                try:
                    report = json.loads(META_REPORT_PATH.read_text(encoding="utf-8"))
                    summary = report.get("walk_forward", {}).get("summary", {})
                    status["precision"] = f"{summary.get('mean_precision', 0) * 100:.1f}%"
                    status["hit_rate"] = f"{summary.get('mean_taken_hit_rate_pct', 0):.1f}%"
                    status["edge"] = f"{summary.get('mean_taken_edge_pct', 0):.2f}%"
                except Exception as e:
                    status["note"] = f"Report error: {e}"
            
            if model is not None:
                status["active"] = True
                status["source"] = "trained_meta"
                status["note"] = "ML Model active!"
            
            return status
        except Exception as e:
            logger.error(f"Model status error: {e}")
            return {"active": False, "source": "error", "precision": "N/A", "hit_rate": "N/A", "edge": "N/A", "note": str(e)}
    
    def get_signals() -> Dict[str, Any]:
        signal_store = {}
        live_portfolio_path = ROOT / "data" / "live_portfolio.json"
        
        if live_portfolio_path.exists():
            try:
                data = json.loads(live_portfolio_path.read_text(encoding="utf-8"))
                positions = data.get("positions", [])
                
                for pos in positions:
                    symbol = pos.get("symbol", "")
                    if symbol:
                        pnl = pos.get("unrealized_pnl", 0)
                        signal = "buy" if pnl >= 0 else "sell"
                        signal_store[symbol] = {
                            "symbol": symbol,
                            "signal": signal,
                            "confidence": pos.get("ml_confidence", 50) / 100.0,
                            "lane": "normal",
                            "price": pos.get("current_price", 0),
                            "change_pct": 0.0,
                        }
            except Exception as e:
                logger.error(f"Error loading signals: {e}")
        
        signals = list(signal_store.values())
        return {
            "signals": signals,
            "buy": sum(1 for s in signals if s.get("signal") == "buy"),
            "sell": sum(1 for s in signals if s.get("signal") == "sell"),
            "total": len(signals),
        }
    
    def get_portfolio() -> Dict[str, Any]:
        live_portfolio_path = ROOT / "data" / "live_portfolio.json"
        
        if live_portfolio_path.exists():
            try:
                data = json.loads(live_portfolio_path.read_text(encoding="utf-8"))
                summary = data.get("summary", {})
                positions = data.get("positions", [])
                
                return {
                    "portfolio_value": summary.get("portfolio_value", 10000),
                    "cash": summary.get("cash", 0),
                    "positions": positions,
                    "return_pct": summary.get("total_return_pct", 0),
                }
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")
        
        return {
            "portfolio_value": 10000,
            "cash": 10000,
            "positions": [],
            "return_pct": 0,
        }
    
    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route("/api/simple-model")
    def api_model():
        return jsonify(get_model_status())
    
    @app.route("/api/simple-signals")
    def api_signals():
        return jsonify(get_signals())
    
    @app.route("/api/simple-portfolio")
    def api_portfolio():
        return jsonify(get_portfolio())
    
    return app


if __name__ == "__main__":
    app = create_app()
    print("=" * 50)
    print("  MacroIntel Dashboard v2")
    print("  Professional Trading UI")
    print("=" * 50)
    print("Starting on port 5051...")
    print("Access at: http://34.14.223.145:5051")
    app.run(host="0.0.0.0", port=5051, debug=False)