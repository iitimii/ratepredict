"""Build an offline, interactive chart for every daily numerical feature."""

import json
from pathlib import Path

import pandas as pd
from plotly.offline import get_plotlyjs


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/dataset/views/daily_quantitative_observed.parquet"
OUTPUT = ROOT / "outputs/numerical-feature-plots"


def main():
    frame = pd.read_parquet(SOURCE).sort_values("date")
    dates = frame["date"].dt.strftime("%Y-%m-%d").tolist()
    features = frame.drop(columns="date").select_dtypes(include="number")
    series = []
    coverage = []
    for name in features:
        values = features[name]
        present = values.notna()
        observed_dates = frame.loc[present, "date"]
        first = observed_dates.min().strftime("%Y-%m-%d") if present.any() else None
        last = observed_dates.max().strftime("%Y-%m-%d") if present.any() else None
        series.append({
            "name": name,
            "y": [float(v) if pd.notna(v) else None for v in values],
            "count": int(present.sum()), "first": first, "last": last,
        })
        coverage.append({"feature": name, "observations": int(present.sum()),
                         "missing_days": int((~present).sum()),
                         "first_observation": first, "last_observation": last})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(OUTPUT / "feature_coverage.csv", index=False)
    payload = json.dumps({"dates": dates, "series": series}, allow_nan=False)
    template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RatePredict — numerical feature history</title>
<style>
body{margin:0;background:#f3f5f8;color:#172b43;font:15px system-ui,sans-serif}
header,main{max-width:1400px;margin:auto;padding:24px}h1{font-size:30px;margin:0 0 12px}
p{line-height:1.6;max-width:1000px}.controls{display:flex;gap:20px;flex-wrap:wrap;align-items:center}
input[type=search]{padding:12px;width:min(440px,85vw);border:1px solid #b9c6d5;border-radius:6px}
section{background:white;margin:0 0 24px;border:1px solid #dbe2eb;border-radius:10px;padding:18px}
h2{font:600 16px ui-monospace,monospace;overflow-wrap:anywhere;margin:0 0 8px}
.meta{color:#5a6b7d;font-size:13px}.chart{height:340px}#count{font-weight:600}
@media print{.controls{display:none}section{break-inside:avoid}.chart{height:300px}}
</style><script>__PLOTLY__</script></head><body>
<header><h1>Numerical feature history</h1>
<p><strong>63 features · 7,550 calendar days · January 2006 – September 2026</strong><br>
Every numerical column in the daily observed dataset has its own graph. All graphs share the full date range;
each vertical axis uses its own scale and original stored values. No normalization or missing-value filling is applied.</p>
<p>Dots show observations; lines join adjacent observed calendar days. Gaps remain empty, so monthly series may appear as dots.
Use “Connect observations across gaps” to draw lines between available observations, including monthly releases;
these connecting lines are visual guides, not additional data. Monthly dates describe observation periods, not verified release dates.</p>
<div class="controls"><input id="search" type="search" aria-label="Filter features" placeholder="Filter features, e.g. inflation, usdtngn, fred">
<label><input id="connect" type="checkbox"> Connect observations across gaps</label><span id="count"></span></div>
<p>Hover for exact values, drag to zoom, double-click to reset, or use a chart’s camera button to export a PNG.
Charts load as you scroll. Data snapshot: September 2, 2026.</p></header><main id="charts"></main>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('dataset').textContent);
const cards=[];const charts=document.getElementById('charts');
function render(card){
 if(card.loaded)return;card.loaded=true;
 Plotly.newPlot(card.chart,[{x:data.dates,y:card.series.y,type:'scatter',mode:'lines+markers',
   connectgaps:document.getElementById('connect').checked,line:{color:'#167c98',width:1.4},
   marker:{size:3,color:'#167c98'},hovertemplate:'%{x|%Y-%m-%d}<br>%{y:,.6~g}<extra></extra>'}],
 {margin:{l:90,r:25,t:15,b:55},paper_bgcolor:'white',plot_bgcolor:'white',
  xaxis:{type:'date',range:[data.dates[0],data.dates.at(-1)],dtick:'M24',tickformat:'%Y',title:{text:'Date'},gridcolor:'#edf0f5'},
  yaxis:{title:{text:'Stored value'},automargin:true,gridcolor:'#edf0f5'},showlegend:false},
 {responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:card.series.name,width:1500,height:500,scale:2}});
}
const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){render(e.target.card);observer.unobserve(e.target)}}),{rootMargin:'500px'});
for(const [i,s] of data.series.entries()){
 const section=document.createElement('section');const heading=document.createElement('h2');heading.textContent=`${i+1}. ${s.name}`;
 const meta=document.createElement('div');meta.className='meta';meta.textContent=`${s.count.toLocaleString()} observations / ${data.dates.length.toLocaleString()} calendar days · Available: ${s.first || 'none'} → ${s.last || 'none'}`;
 const chart=document.createElement('div');chart.className='chart';chart.setAttribute('aria-label',s.name+' against time');
 section.append(heading,meta,chart);charts.append(section);
 const card={section,chart,series:s,loaded:false};section.card=card;cards.push(card);observer.observe(section);
}
document.getElementById('search').addEventListener('input',e=>{
 const q=e.target.value.toLowerCase();let visible=0;
 cards.forEach(c=>{c.section.hidden=!c.series.name.toLowerCase().includes(q);if(!c.section.hidden)visible++});
 document.getElementById('count').textContent=`${visible} / ${cards.length} features`;
 window.dispatchEvent(new Event('resize'));
});
document.getElementById('connect').addEventListener('change',e=>cards.filter(c=>c.loaded).forEach(c=>Plotly.restyle(c.chart,{connectgaps:e.target.checked})));
document.getElementById('count').textContent=`${cards.length} / ${cards.length} features`;
</script></body></html>"""
    html = template.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", payload)
    destination = OUTPUT / "all_numerical_features.html"
    destination.write_text(html)
    print(f"Wrote {len(series)} graphs to {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
