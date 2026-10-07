/* ═══════════════════════════════════════════════════════
   PhishGuard – main.js
   Handles: tab switching, URL scanning, charts, history
═══════════════════════════════════════════════════════ */

"use strict";

// ──────────────────────────────────────────────────────
// Tab navigation
// ──────────────────────────────────────────────────────
const navBtns    = document.querySelectorAll(".nav-btn");
const tabSections = document.querySelectorAll(".tab-content");

navBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;

    navBtns.forEach(b => b.classList.remove("active"));
    tabSections.forEach(s => { s.classList.remove("active"); s.classList.add("hidden"); });

    btn.classList.add("active");
    const section = document.getElementById("tab-" + target);
    if (section) { section.classList.remove("hidden"); section.classList.add("active"); }

    // Lazy-load data for specific tabs
    if (target === "performance") loadPerformance();
    if (target === "history")     loadHistory();
  });
});

// ──────────────────────────────────────────────────────
// URL Scanner
// ──────────────────────────────────────────────────────
const urlInput   = document.getElementById("url-input");
const scanBtn    = document.getElementById("scan-btn");
const scanText   = document.getElementById("scan-btn-text");
const scanSpinner = document.getElementById("scan-btn-spinner");
const scanError  = document.getElementById("scan-error");
const resultCard = document.getElementById("result-card");

function setScanning(active) {
  scanBtn.disabled = active;
  scanText.textContent  = active ? "Scanning…" : "Scan URL";
  scanSpinner.classList.toggle("hidden", !active);
}

function showError(msg) {
  scanError.textContent = msg;
  scanError.classList.remove("hidden");
}

function hideError() {
  scanError.classList.add("hidden");
}

async function doScan() {
  const url = urlInput.value.trim();
  if (!url) { showError("Please enter a URL."); return; }
  hideError();
  setScanning(true);
  resultCard.classList.add("hidden");

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();

    if (!res.ok) { showError(data.error || "Server error."); return; }
    renderResult(data);
    refreshStats();
  } catch (err) {
    showError("Network error – is the server running?");
  } finally {
    setScanning(false);
  }
}

scanBtn.addEventListener("click", doScan);
urlInput.addEventListener("keydown", e => { if (e.key === "Enter") doScan(); });

// ──────────────────────────────────────────────────────
// Render scan result
// ──────────────────────────────────────────────────────
function renderResult(data) {
  const isPhishing = data.result === "Phishing";

  // Banner
  const banner = document.getElementById("result-banner");
  banner.className = "result-banner " + (isPhishing ? "phishing" : "legit");
  document.getElementById("result-icon").textContent  = isPhishing ? "🔴" : "🟢";
  document.getElementById("result-label").textContent =
    isPhishing ? "PHISHING URL DETECTED" : "LEGITIMATE URL";

  // ML prediction
  renderPredBox("ml", data.ml);
  renderPredBox("nn", data.nn);

  // Reasons
  const ul = document.getElementById("reasons-list");
  ul.innerHTML = "";
  (data.reasons || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    ul.appendChild(li);
  });

  // Features table
  const tbody = document.getElementById("features-tbody");
  tbody.innerHTML = "";
  (data.features || []).forEach(f => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${escHtml(f.name)}</td><td><strong>${escHtml(String(f.value))}</strong></td>`;
    tbody.appendChild(tr);
  });

  resultCard.classList.remove("hidden");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function renderPredBox(prefix, pred) {
  const isPhishing = pred.prediction === "Phishing";
  const labelEl   = document.getElementById(prefix + "-pred");
  const barEl     = document.getElementById(prefix + "-conf-bar");
  const textEl    = document.getElementById(prefix + "-conf-text");

  labelEl.textContent  = pred.prediction;
  labelEl.className    = "pred-label " + (isPhishing ? "phishing" : "legit");
  barEl.style.width    = pred.confidence + "%";
  barEl.className      = "confidence-bar " + (isPhishing ? "phishing" : "legit");
  textEl.textContent   = `Confidence: ${pred.confidence}%`;
}

// ──────────────────────────────────────────────────────
// Refresh dashboard stats
// ──────────────────────────────────────────────────────
async function refreshStats() {
  try {
    const s = await fetch("/api/stats").then(r => r.json());
    document.getElementById("stat-total").textContent    = s.total    ?? 0;
    document.getElementById("stat-phishing").textContent = s.phishing ?? 0;
    document.getElementById("stat-legit").textContent    = s.legitimate ?? 0;
  } catch (_) {}
}

// ──────────────────────────────────────────────────────
// Model Performance tab
// ──────────────────────────────────────────────────────
let perfLoaded  = false;
let chartAcc    = null;
let chartPrf    = null;

async function loadPerformance() {
  if (perfLoaded) return;

  let m = {};
  try {
    m = await fetch("/api/metrics").then(r => r.json());
  } catch (_) { return; }

  if (!m.random_forest && !m.neural_network) {
    document.getElementById("metrics-tbody").innerHTML =
      "<tr><td colspan='5' style='text-align:center'>No metrics found – train the models first.</td></tr>";
    return;
  }

  const rf = m.random_forest || {};
  const nn = m.neural_network || {};

  // Table
  const tbody = document.getElementById("metrics-tbody");
  tbody.innerHTML = `
    <tr>
      <td>🌲 Random Forest</td>
      <td>${pct(rf.accuracy)}</td>
      <td>${pct(rf.precision)}</td>
      <td>${pct(rf.recall)}</td>
      <td>${pct(rf.f1)}</td>
    </tr>
    <tr>
      <td>🧠 Neural Network</td>
      <td>${pct(nn.accuracy)}</td>
      <td>${pct(nn.precision)}</td>
      <td>${pct(nn.recall)}</td>
      <td>${pct(nn.f1)}</td>
    </tr>`;

  // Accuracy chart
  const ctxAcc = document.getElementById("chart-accuracy").getContext("2d");
  if (chartAcc) chartAcc.destroy();
  chartAcc = new Chart(ctxAcc, {
    type: "bar",
    data: {
      labels: ["Random Forest", "Neural Network"],
      datasets: [{
        label: "Accuracy",
        data: [rf.accuracy ?? 0, nn.accuracy ?? 0],
        backgroundColor: ["rgba(88,166,255,0.7)", "rgba(188,140,255,0.7)"],
        borderColor:     ["#58a6ff", "#bc8cff"],
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: chartOptions("Accuracy", 1),
  });

  // Precision / Recall / F1 chart
  const ctxPrf = document.getElementById("chart-prf").getContext("2d");
  if (chartPrf) chartPrf.destroy();
  chartPrf = new Chart(ctxPrf, {
    type: "bar",
    data: {
      labels: ["Precision", "Recall", "F1-Score"],
      datasets: [
        {
          label: "Random Forest",
          data: [rf.precision ?? 0, rf.recall ?? 0, rf.f1 ?? 0],
          backgroundColor: "rgba(88,166,255,0.7)",
          borderColor: "#58a6ff",
          borderWidth: 2,
          borderRadius: 5,
        },
        {
          label: "Neural Network",
          data: [nn.precision ?? 0, nn.recall ?? 0, nn.f1 ?? 0],
          backgroundColor: "rgba(188,140,255,0.7)",
          borderColor: "#bc8cff",
          borderWidth: 2,
          borderRadius: 5,
        },
      ],
    },
    options: chartOptions("Score", 1),
  });

  perfLoaded = true;
}

function chartOptions(yLabel, maxY) {
  return {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#e6edf3", font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(2)}%`,
        },
      },
    },
    scales: {
      x: { ticks: { color: "#8b949e" }, grid: { color: "#30363d" } },
      y: {
        min: 0, max: maxY,
        ticks: { color: "#8b949e", callback: v => (v * 100).toFixed(0) + "%" },
        grid: { color: "#30363d" },
        title: { display: true, text: yLabel, color: "#8b949e" },
      },
    },
  };
}

function pct(v) {
  if (v == null || v === "") return "–";
  return (parseFloat(v) * 100).toFixed(2) + "%";
}

// ──────────────────────────────────────────────────────
// Scan History tab
// ──────────────────────────────────────────────────────
async function loadHistory() {
  let scans = [];
  try {
    scans = await fetch("/api/history").then(r => r.json());
  } catch (_) { return; }

  const tbody = document.getElementById("history-tbody");
  if (!scans.length) {
    tbody.innerHTML = "<tr><td colspan='8' style='text-align:center'>No scans yet.</td></tr>";
    return;
  }

  tbody.innerHTML = scans.map((s, i) => `
    <tr>
      <td>${i + 1}</td>
      <td title="${escHtml(s.url)}">${escHtml(s.url)}</td>
      <td><span class="badge ${s.result === 'Phishing' ? 'badge-phishing' : 'badge-legit'}">${escHtml(s.result)}</span></td>
      <td>${escHtml(s.ml_pred)}</td>
      <td>${(s.ml_conf * 100).toFixed(1)}%</td>
      <td>${escHtml(s.nn_pred)}</td>
      <td>${(s.nn_conf * 100).toFixed(1)}%</td>
      <td>${escHtml(s.scanned_at)}</td>
    </tr>`).join("");
}

// ──────────────────────────────────────────────────────
// Utility
// ──────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
