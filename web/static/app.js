/**
 * Cyber Threat Forecasting World Model — Interactive SOC Dashboard
 * 1. Threat Type Classification & Exact Percentage Breakdown
 * 2. Proper Graphical Representation of Threat Levels (Canvas)
 * 3. Simplified, Intuitive Buttons
 */

const state = {
  activeScenario: 'kill_chain',
  isAnalyzing: false,
  analysisData: null,
};

const STAGE_COLORS = {
  'Benign': '#10b981',            // Emerald
  'Reconnaissance': '#eab308',    // Amber
  'Initial Access': '#f97316',    // Orange
  'Lateral Movement': '#a855f7',  // Purple
  'Command & Control': '#38bdf8', // Blue
  'Exfiltration': '#ef4444',      // Crimson
};

function showToast(msg, type = 'success') {
  const banner = document.getElementById('global-alert');
  if (!banner) return;
  banner.style.display = 'flex';
  banner.className = `alert-banner alert-${type === 'error' ? 'danger' : 'success'}`;
  banner.innerHTML = `<span>${type === 'error' ? '[ALERT]' : '[SUCCESS]'} ${msg}</span>`;
  setTimeout(() => { banner.style.display = 'none'; }, 5000);
}

// =========================================================================
// 1. DATASET INGESTION & SIMPLIFIED BUTTONS
// =========================================================================

function initDropZone() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  if (!dropZone || !fileInput) return;

  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFileUpload(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileUpload(e.target.files[0]);
  });
}

async function handleFileUpload(file) {
  if (!file.name.endsWith('.csv')) {
    showToast('Please upload a valid .csv network telemetry file.', 'error');
    return;
  }

  showToast(`Uploading '${file.name}'...`, 'success');
  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/dataset/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!data.success) {
      showToast(data.error || 'Failed to parse CSV.', 'error');
      return;
    }

    document.getElementById('ds-name').textContent = data.filename;
    document.getElementById('ds-flows').textContent = data.num_flows.toLocaleString();
    document.getElementById('ds-sessions').textContent = `${data.sessions.length} session(s)`;
    document.getElementById('ds-duration').textContent = `${data.time_span_sec}s`;
    document.getElementById('ds-status-pill').textContent = 'Custom Data Loaded';

    document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
    showToast(`Loaded ${data.num_flows} flows. Analyzing...`);
    runForecast();
  } catch (err) {
    showToast(`Upload error: ${err.message}`, 'error');
  }
}

function initScenarioButtons() {
  const btns = document.querySelectorAll('.scenario-btn');
  btns.forEach(btn => {
    btn.addEventListener('click', async () => {
      btns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const scenario = btn.getAttribute('data-scenario');
      state.activeScenario = scenario;

      try {
        const res = await fetch('/api/dataset/load-sample', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scenario }),
        });
        const data = await res.json();
        if (data.success) {
          document.getElementById('ds-name').textContent = data.scenario_name;
          document.getElementById('ds-flows').textContent = data.num_flows.toLocaleString();
          document.getElementById('ds-sessions').textContent = `${data.sessions.length} session(s)`;
          document.getElementById('ds-duration').textContent = `${data.time_span_sec}s`;
          document.getElementById('ds-status-pill').textContent = 'Sample Active';
          showToast(`Loaded '${data.scenario_name}'. Analyzing threat...`);
          runForecast();
        }
      } catch (err) {
        showToast(`Failed to load: ${err.message}`, 'error');
      }
    });
  });
}

function initTemplateDownload() {
  const btn = document.getElementById('btn-download-template');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/dataset/template');
      const data = await res.json();
      if (data.sample_csv) {
        const blob = new Blob([data.sample_csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'sample_telemetry_flows.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        showToast('Downloaded sample CSV template.');
      }
    } catch (err) {
      showToast('Could not fetch template.', 'error');
    }
  });
}

// =========================================================================
// 2. RUN ANALYSIS & PROGRESS
// =========================================================================

async function runForecast() {
  if (state.isAnalyzing) return;
  state.isAnalyzing = true;

  const progBox = document.getElementById('execution-progress');
  const animBar = document.getElementById('anim-progress-bar');
  const btnRun = document.getElementById('btn-run-forecast');
  
  if (progBox) progBox.style.display = 'block';
  if (btnRun) {
    btnRun.disabled = true;
    btnRun.innerHTML = 'Analyzing Threat...';
  }

  const steps = [
    { pct: '25%', activeId: 'step-1-label' },
    { pct: '50%', activeId: 'step-2-label' },
    { pct: '75%', activeId: 'step-3-label' },
    { pct: '95%', activeId: 'step-4-label' },
  ];

  let currentStep = 0;
  const stepInterval = setInterval(() => {
    if (currentStep < steps.length) {
      animBar.style.width = steps[currentStep].pct;
      document.querySelectorAll('.step-lbl').forEach(l => l.classList.remove('active'));
      const activeLbl = document.getElementById(steps[currentStep].activeId);
      if (activeLbl) activeLbl.classList.add('active');
      currentStep++;
    }
  }, 200);

  try {
    const res = await fetch('/api/dataset/analyze', { method: 'POST' });
    const data = await res.json();
    clearInterval(stepInterval);

    if (!data.success) {
      showToast(data.error || 'Analysis failed.', 'error');
      return;
    }

    state.analysisData = data;
    renderResults(data);
    showToast('Threat classification & forecast successfully updated.');
  } catch (err) {
    clearInterval(stepInterval);
    showToast(`Error: ${err.message}`, 'error');
  } finally {
    state.isAnalyzing = false;
    if (animBar) animBar.style.width = '100%';
    setTimeout(() => {
      if (progBox) progBox.style.display = 'none';
    }, 500);
    if (btnRun) {
      btnRun.disabled = false;
      btnRun.innerHTML = 'Analyze Threat';
    }
  }
}

// =========================================================================
// 3. GRAPHICAL RENDERING & THREAT LEVEL VISUALIZATIONS
// =========================================================================

function renderResults(data) {
  const sum = data.summary;
  const threatClass = sum.threat_classification || {};

  // 1. Top KPI Metrics
  const riskPct = sum.current_risk_pct;
  const kpiRisk = document.getElementById('kpi-risk-score');
  const badgeLevel = document.getElementById('badge-threat-level');
  const kpiBar = document.getElementById('kpi-risk-bar');

  if (kpiRisk) kpiRisk.textContent = `${riskPct.toFixed(1)}%`;
  if (badgeLevel) {
    badgeLevel.textContent = sum.threat_level;
    const color = sum.threat_level === 'CRITICAL' ? '#ef4444' :
                  sum.threat_level === 'HIGH' ? '#f97316' :
                  sum.threat_level === 'ELEVATED' ? '#f59e0b' : '#10b981';
    badgeLevel.style.color = color;
    badgeLevel.style.borderColor = color;
    badgeLevel.style.background = `${color}20`;
    if (kpiRisk) kpiRisk.style.color = color;
    if (kpiBar) {
      kpiBar.style.width = `${Math.min(100, riskPct)}%`;
      kpiBar.style.background = color;
    }
  }

  // Primary Threat Type
  const stageTag = document.getElementById('kpi-stage-tag');
  if (stageTag) {
    stageTag.textContent = sum.predicted_stage;
    const c = STAGE_COLORS[sum.predicted_stage] || '#00e5ff';
    stageTag.style.color = c;
    stageTag.style.borderColor = `${c}60`;
    stageTag.style.background = `${c}20`;
  }

  // Trend
  const kpiTrend = document.getElementById('kpi-trend');
  if (kpiTrend) {
    kpiTrend.textContent = sum.forecast_trend;
    kpiTrend.className = sum.trend_class;
    kpiTrend.style.color = sum.forecast_trend.includes('Escalating') ? '#ef4444' :
                           sum.forecast_trend.includes('De-escalating') ? '#10b981' : '#f59e0b';
  }

  // Horizon
  const kpiHorizon = document.getElementById('kpi-horizon');
  if (kpiHorizon) kpiHorizon.textContent = `+${sum.forecast_horizon_sec}s`;

  // 2. GRAPH 1: Threat Level Progression Over Time (Past + Future Rollout)
  drawThreatLevelProgression('chart-forecast-rollout', data.forecast_steps, sum.current_risk_score);
  renderRolloutStepCards(data.forecast_steps);

  // 3. GRAPH 2: Threat Classification by Type (% Breakdown)
  const threatPercentages = threatClass.threat_percentages || {
    'Benign': 0, 'Reconnaissance': 0, 'Initial Access': 0,
    'Lateral Movement': 0, 'Command & Control': 0, 'Exfiltration': 0
  };
  drawThreatPercentagesChart('chart-threat-types', threatPercentages);
  renderThreatTypesLegend(threatPercentages);

  // 4. GRAPH 3: Root-Cause Threat Drivers (SHAP Attribution)
  drawShapDriversChart('chart-threat-shap', data.top_threat_drivers);
  renderShapDriversList(data.top_threat_drivers);

  // 5. Timeline Table
  renderTimelineTable(data.timeline);

  // Background diagnostics
  loadDiagnosticsData();
}

// GRAPH 1: Proper Graphical Threat Level Progression with Safety/Danger Bands
function drawThreatLevelProgression(canvasId, steps, currentRisk) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, w, h);
  if (!steps || steps.length === 0) return;

  const pad = { top: 30, right: 30, bottom: 40, left: 55 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  const pts = [
    { lookahead_sec: 0, risk: currentRisk || 0, uncertainty: 0.03, stage: 'Now' },
    ...steps.map(s => ({
      lookahead_sec: s.lookahead_sec,
      risk: s.risk_score,
      uncertainty: s.uncertainty || 0.05,
      stage: s.predicted_stage,
    }))
  ];

  const getX = (idx) => pad.left + (idx / (pts.length - 1)) * plotW;
  const getY = (risk) => pad.top + plotH - (Math.max(0, Math.min(1, risk)) * plotH);

  // Color-coded Threat Severity Bands
  // 1. Critical Band (> 75%) - Red
  ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
  ctx.fillRect(pad.left, pad.top, plotW, plotH * 0.25);
  ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';
  ctx.font = '10px Inter';
  ctx.fillText('CRITICAL DANGER ZONE (> 75%)', pad.left + 10, pad.top + 15);

  // 2. High Band (50% - 75%) - Orange
  ctx.fillStyle = 'rgba(249, 115, 22, 0.08)';
  ctx.fillRect(pad.left, pad.top + plotH * 0.25, plotW, plotH * 0.25);

  // 3. Guarded Band (25% - 50%) - Amber
  ctx.fillStyle = 'rgba(245, 158, 11, 0.05)';
  ctx.fillRect(pad.left, pad.top + plotH * 0.50, plotW, plotH * 0.25);

  // 4. Safe Band (< 25%) - Emerald
  ctx.fillStyle = 'rgba(16, 185, 129, 0.04)';
  ctx.fillRect(pad.left, pad.top + plotH * 0.75, plotW, plotH * 0.25);

  // Horizontal Grid Lines & Percent Axis
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  [1.0, 0.75, 0.50, 0.25, 0.0].forEach((val, i) => {
    const y = pad.top + (i / 4) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(w - pad.right, y);
    ctx.stroke();

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px monospace';
    ctx.fillText(`${(val * 100).toFixed(0)}%`, 14, y + 3);
  });

  // Uncertainty Envelope
  ctx.beginPath();
  pts.forEach((p, idx) => {
    const x = getX(idx);
    const yTop = getY(Math.min(1.0, p.risk + p.uncertainty * 0.35));
    if (idx === 0) ctx.moveTo(x, yTop); else ctx.lineTo(x, yTop);
  });
  for (let idx = pts.length - 1; idx >= 0; idx--) {
    const x = getX(idx);
    const yBot = getY(Math.max(0.0, pts[idx].risk - pts[idx].uncertainty * 0.35));
    ctx.lineTo(x, yBot);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(0, 229, 255, 0.15)';
  ctx.fill();

  // Forecast Line
  ctx.beginPath();
  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 3.5;
  pts.forEach((p, idx) => {
    const x = getX(idx);
    const y = getY(p.risk);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw Node Dots with Color Coding
  pts.forEach((p, idx) => {
    const x = getX(idx);
    const y = getY(p.risk);
    const nodeColor = p.risk >= 0.75 ? '#ef4444' : p.risk >= 0.50 ? '#f97316' : p.risk >= 0.25 ? '#f59e0b' : '#10b981';

    ctx.fillStyle = nodeColor;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#0d1322';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Node percentage text
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 10px monospace';
    ctx.fillText(`${(p.risk * 100).toFixed(0)}%`, x - 10, y - 10);

    // X-axis label
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px monospace';
    const lbl = idx === 0 ? 'Now' : `+${p.lookahead_sec}s`;
    ctx.fillText(lbl, x - 12, h - 12);
  });
}

function renderRolloutStepCards(steps) {
  const container = document.getElementById('rollout-steps-container');
  if (!container || !steps) return;
  container.innerHTML = steps.map(s => {
    const color = s.risk_score >= 0.75 ? '#ef4444' : s.risk_score >= 0.50 ? '#f97316' : s.risk_score >= 0.25 ? '#f59e0b' : '#10b981';
    return `
      <div class="rollout-step-card">
        <div class="step-time-badge">+${s.lookahead_sec}s</div>
        <div class="step-risk-val" style="color: ${color};">${(s.risk_score * 100).toFixed(0)}%</div>
        <div class="step-stage-name" title="${s.predicted_stage}">${s.predicted_stage}</div>
      </div>
    `;
  }).join('');
}

// GRAPH 2: Threat Type Classification & Percentage Breakdown Chart
function drawThreatPercentagesChart(canvasId, stagePercentages) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, w, h);

  const stages = Object.keys(stagePercentages);
  if (stages.length === 0) return;

  const pad = { top: 15, right: 60, bottom: 25, left: 140 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const barHeight = Math.min(24, (plotH / stages.length) - 8);

  stages.forEach((stageName, idx) => {
    const pct = stagePercentages[stageName] || 0.0;
    const y = pad.top + idx * (plotH / stages.length) + 4;
    const barW = (pct / 100.0) * plotW;
    const color = STAGE_COLORS[stageName] || '#00e5ff';

    // Label on left
    ctx.fillStyle = '#f1f5f9';
    ctx.font = '11px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(stageName, pad.left - 12, y + barHeight / 2 + 4);

    // Track
    ctx.fillStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.fillRect(pad.left, y, plotW, barHeight);

    // Colored Percentage Bar
    if (barW > 0) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(pad.left, y, Math.max(4, barW), barHeight, 3);
      ctx.fill();
    }

    // Percentage value text on right
    ctx.fillStyle = pct > 0 ? color : '#64748b';
    ctx.font = 'bold 11px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`${pct.toFixed(1)}%`, pad.left + Math.max(4, barW) + 8, y + barHeight / 2 + 4);
  });
}

function renderThreatTypesLegend(stagePercentages) {
  const container = document.getElementById('threat-types-legend');
  if (!container) return;

  container.innerHTML = Object.entries(stagePercentages).map(([stage, pct]) => {
    const color = STAGE_COLORS[stage] || '#00e5ff';
    return `
      <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 4px; padding: 0.3rem 0.6rem; font-size: 0.75rem; display: flex; align-items: center; gap: 0.4rem;">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:${color};"></span>
        <span style="color:#cbd5e1;">${stage}:</span>
        <strong style="color:${color}; font-family:var(--font-mono);">${pct.toFixed(1)}%</strong>
      </div>
    `;
  }).join('');
}

// GRAPH 3: Root-Cause Threat Drivers (SHAP Attribution)
function drawShapDriversChart(canvasId, drivers) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, w, h);
  if (!drivers || drivers.length === 0) return;

  const pad = { top: 15, right: 55, bottom: 20, left: 160 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const maxImpact = Math.max(...drivers.map(d => Math.abs(d.shap_impact))) || 0.1;
  const barHeight = Math.min(20, (plotH / drivers.length) - 6);

  drivers.forEach((d, idx) => {
    const y = pad.top + idx * (plotH / drivers.length) + 4;
    const barW = (Math.abs(d.shap_impact) / maxImpact) * plotW;

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px Inter';
    ctx.textAlign = 'right';
    const shortLabel = d.label.length > 22 ? d.label.substring(0, 20) + '...' : d.label;
    ctx.fillText(shortLabel, pad.left - 10, y + barHeight / 2 + 4);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.fillRect(pad.left, y, plotW, barHeight);

    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.roundRect(pad.left, y, Math.max(3, barW), barHeight, 3);
    ctx.fill();

    ctx.fillStyle = '#f1f5f9';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`+${d.shap_impact.toFixed(3)}`, pad.left + barW + 6, y + barHeight / 2 + 3);
  });
}

function renderShapDriversList(drivers) {
  const container = document.getElementById('shap-drivers-list');
  if (!container || !drivers) return;

  container.innerHTML = drivers.slice(0, 4).map(d => `
    <div class="driver-bar-row" style="background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.2); padding: 0.4rem 0.8rem; border-radius: 4px; margin-bottom: 0.4rem;">
      <span class="driver-label" style="color:#f1f5f9; font-size:0.8rem;"> ${d.label}</span>
      <span class="driver-val" style="color:var(--cyan); font-family:var(--font-mono); font-size:0.8rem;">value: ${d.actual_value} (&Delta; +${d.shap_impact.toFixed(3)})</span>
    </div>
  `).join('');
}

function renderTimelineTable(timeline) {
  const tbody = document.getElementById('timeline-table-body');
  if (!tbody || !timeline) return;

  if (timeline.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No windows recorded.</td></tr>';
    return;
  }

  tbody.innerHTML = timeline.map(row => {
    const badgeColor = row.threat_level === 'CRITICAL' ? 'var(--crimson)' :
                       row.threat_level === 'HIGH' ? '#f97316' :
                       row.threat_level === 'ELEVATED' ? 'var(--amber)' : 'var(--emerald)';
    return `
      <tr>
        <td style="font-family: var(--font-mono); color: var(--cyan);">#${row.step}</td>
        <td>${row.window_time}</td>
        <td><code>${row.session_id}</code></td>
        <td><strong style="color: ${STAGE_COLORS[row.stage] || '#fff'}">${row.stage}</strong></td>
        <td>
          <span style="color: ${badgeColor}; font-weight: 700;">${row.risk_pct.toFixed(1)}%</span>
        </td>
        <td>
          <span class="badge" style="color: ${badgeColor}; border-color: ${badgeColor}; background: ${badgeColor}15; font-size: 0.7rem;">
            ${row.threat_level}
          </span>
        </td>
        <td>${row.flow_count}</td>
        <td style="font-family: var(--font-mono);">${row.byte_rate.toLocaleString()} B/s</td>
      </tr>
    `;
  }).join('');
}

function initAccordion() {
  const btn = document.getElementById('btn-toggle-diagnostics');
  const content = document.getElementById('diagnostics-content');
  const arrow = document.getElementById('accordion-arrow');
  if (!btn || !content) return;

  btn.addEventListener('click', () => {
    const isHidden = content.style.display === 'none';
    content.style.display = isHidden ? 'block' : 'none';
    if (arrow) arrow.textContent = isHidden ? '▲ Click to Collapse' : '▼ Click to Expand';
    if (isHidden) loadDiagnosticsData();
  });
}

async function loadDiagnosticsData() {
  try {
    const trainRes = await fetch('/api/model/train', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ num_epochs: 10 }),
    });
    const trainData = await trainRes.json();
    if (trainData.success) {
      drawDiagnosticsLoss('chart-training-loss', trainData.epochs, trainData.train_losses, trainData.val_losses);
    }

    const mitreRes = await fetch('/api/mitre/evaluate', { method: 'POST' });
    const mitreData = await mitreRes.json();
    if (mitreData.success) {
      renderConfusionMatrix(mitreData.confusion_matrix, mitreData.confusion_matrix_labels);
    }
  } catch (err) {
    console.warn('Diagnostics background load:', err);
  }
}

function drawDiagnosticsLoss(canvasId, epochs, trainLoss, valLoss) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, w, h);
  if (!epochs || epochs.length === 0) return;

  const pad = { top: 20, right: 20, bottom: 25, left: 45 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;
  const all = [...trainLoss, ...valLoss];
  const minV = Math.min(...all), maxV = Math.max(...all);
  const range = (maxV - minV) || 1.0;

  const getX = (i) => pad.left + (i / (epochs.length - 1)) * plotW;
  const getY = (v) => pad.top + plotH - ((v - minV) / range) * plotH;

  ctx.strokeStyle = '#00e5ff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  trainLoss.forEach((v, i) => { const x = getX(i), y = getY(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();

  ctx.strokeStyle = '#c084fc';
  ctx.lineWidth = 2;
  ctx.beginPath();
  valLoss.forEach((v, i) => { const x = getX(i), y = getY(v); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
  ctx.stroke();
}

function renderConfusionMatrix(matrix, labels) {
  const tbody = document.getElementById('diag-cm-body');
  if (!tbody || !matrix || !labels) return;

  let html = '<tr><th style="font-size: 0.7rem;">True \\ Pred</th>';
  labels.forEach(l => {
    const shortL = l.length > 8 ? l.substring(0, 7) + '.' : l;
    html += `<th style="font-size: 0.7rem;" title="${l}">${shortL}</th>`;
  });
  html += '</tr>';

  matrix.forEach((row, rIdx) => {
    const rowLabel = labels[rIdx] || `C${rIdx}`;
    const shortR = rowLabel.length > 8 ? rowLabel.substring(0, 7) + '.' : rowLabel;
    html += `<tr><td style="font-size: 0.7rem; font-weight: 700;" title="${rowLabel}">${shortR}</td>`;
    row.forEach((cnt, cIdx) => {
      const isDiag = rIdx === cIdx;
      const bg = isDiag && cnt > 0 ? 'rgba(16, 185, 129, 0.25)' : (!isDiag && cnt > 0 ? 'rgba(239, 68, 68, 0.2)' : 'transparent');
      html += `<td style="background: ${bg}; font-size: 0.75rem; text-align: center;">${cnt}</td>`;
    });
    html += '</tr>';
  });
  tbody.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', () => {
  initDropZone();
  initScenarioButtons();
  initTemplateDownload();
  initAccordion();

  const btnRun = document.getElementById('btn-run-forecast');
  if (btnRun) btnRun.addEventListener('click', runForecast);

  runForecast();
});
