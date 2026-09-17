/**
 * CyberShield AML — Streamlined Client-Side Controller
 * Fast, lightweight, and focused URL security scanner and benchmark viewer.
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initScanner();
  initBenchmarks();
  initModal();
});

/* ==========================================================================
   1. Tab Navigation Controller
   ========================================================================== */
function initTabs() {
  const tabs = document.querySelectorAll(".nav-tab");
  const panes = document.querySelectorAll(".tab-pane");

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      
      tabs.forEach(t => t.classList.remove("active"));
      panes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      const targetPane = document.getElementById(targetId);
      if (targetPane) {
        targetPane.classList.add("active");
      }
    });
  });
}

/* ==========================================================================
   2. URL Scanner Controller
   ========================================================================== */
function initScanner() {
  const urlInput = document.getElementById("url-input");
  const scanBtn = document.getElementById("btn-scan-url");
  const clearBtn = document.getElementById("btn-clear-url");
  const modelSelector = document.getElementById("model-selector");
  const modelLabel = document.getElementById("active-model-label");
  const sampleChips = document.querySelectorAll(".sample-chip");
  const toggleFeaturesBtn = document.getElementById("toggle-features-btn");
  const allFeaturesDrawer = document.getElementById("all-features-drawer");

  // Model select change updates header label
  if (modelSelector && modelLabel) {
    modelSelector.addEventListener("change", () => {
      const optText = modelSelector.options[modelSelector.selectedIndex].text;
      modelLabel.textContent = optText;
    });
  }

  // Clear button
  if (clearBtn && urlInput) {
    clearBtn.addEventListener("click", () => {
      urlInput.value = "";
      urlInput.focus();
    });
  }

  // Enter key trigger
  if (urlInput) {
    urlInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") performScan();
    });
  }

  // Scan button trigger
  if (scanBtn) {
    scanBtn.addEventListener("click", performScan);
  }

  // Sample chips
  sampleChips.forEach(chip => {
    chip.addEventListener("click", () => {
      const sampleUrl = chip.getAttribute("data-url");
      if (sampleUrl && urlInput) {
        urlInput.value = sampleUrl;
        performScan();
      }
    });
  });

  // Collapsible toggle for 25 features table
  if (toggleFeaturesBtn && allFeaturesDrawer) {
    toggleFeaturesBtn.addEventListener("click", () => {
      const isHidden = allFeaturesDrawer.style.display === "none";
      allFeaturesDrawer.style.display = isHidden ? "block" : "none";
      toggleFeaturesBtn.textContent = isHidden ? "Hide 25 Features ▴" : "View All 25 Features ▾";
    });
  }
}

async function performScan() {
  const urlInput = document.getElementById("url-input");
  const scanBtn = document.getElementById("btn-scan-url");
  const spinner = document.getElementById("scan-spinner");
  const modelSelector = document.getElementById("model-selector");
  const resultsSection = document.getElementById("scan-results-section");

  const rawUrl = urlInput.value.trim();
  if (!rawUrl) {
    urlInput.focus();
    return;
  }

  // Loading state
  scanBtn.disabled = true;
  spinner.style.display = "inline-block";

  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: rawUrl,
        model: modelSelector ? modelSelector.value : "rf"
      })
    });

    const data = await res.json();

    if (!res.ok) {
      alert(data.error || "Failed to analyze URL.");
      return;
    }

    renderScanResults(data);
    resultsSection.style.display = "block";
    resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

  } catch (err) {
    console.error("Scan error:", err);
    alert("Connection to ML engine failed.");
  } finally {
    scanBtn.disabled = false;
    spinner.style.display = "none";
  }
}

function renderScanResults(data) {
  // Latency & Target URL
  document.getElementById("result-latency").textContent = `${data.latency_ms} ms`;
  document.getElementById("result-target-url").textContent = data.url;

  // Verdict Card
  const verdictTitle = document.getElementById("verdict-title");
  const verdictDesc = document.getElementById("verdict-desc");
  const verdictIcon = document.getElementById("verdict-icon");
  const verdictCard = document.getElementById("verdict-card");

  verdictCard.classList.remove("verdict-danger", "verdict-warning", "verdict-safe");
  verdictTitle.textContent = data.verdict;
  verdictDesc.textContent = data.status_desc;

  if (data.status_code === "danger") {
    verdictCard.classList.add("verdict-danger");
    verdictIcon.textContent = "🚨";
  } else if (data.status_code === "warning") {
    verdictCard.classList.add("verdict-warning");
    verdictIcon.textContent = "⚠️";
  } else {
    verdictCard.classList.add("verdict-safe");
    verdictIcon.textContent = "🛡️";
  }

  // Circular Gauge (r = 50, circumference = 2 * PI * 50 = ~314.16)
  const circle = document.getElementById("risk-progress-bar");
  const riskNum = document.getElementById("gauge-risk-val");
  const circumference = 314.16;
  const offset = circumference - (data.risk_score / 100) * circumference;
  circle.style.strokeDashoffset = offset;
  riskNum.textContent = `${data.risk_score}%`;

  if (data.risk_score >= 70) {
    circle.setAttribute("stroke", "#ff2a5f");
  } else if (data.risk_score >= 35) {
    circle.setAttribute("stroke", "#ffb703");
  } else {
    circle.setAttribute("stroke", "#00f59b");
  }

  // Threat Indicators List
  const indContainer = document.getElementById("indicators-list-container");
  const indCount = document.getElementById("indicator-count");
  indContainer.innerHTML = "";
  indCount.textContent = `${data.indicators.length} Flag${data.indicators.length === 1 ? '' : 's'}`;

  data.indicators.forEach(item => {
    const el = document.createElement("div");
    el.className = `indicator-item level-${item.level}`;
    let icon = "🛡️";
    if (item.level === "danger") icon = "⚠️";
    if (item.level === "warning") icon = "⚡";
    if (item.level === "info") icon = "ℹ️";

    el.innerHTML = `
      <div class="ind-icon">${icon}</div>
      <div class="ind-text">
        <div class="ind-title">${item.title}</div>
        <div class="ind-desc">${item.desc}</div>
      </div>
    `;
    indContainer.appendChild(el);
  });

  // Diagnostic 6-Tile Feature Highlights
  const f = data.features;
  document.getElementById("diag-length").textContent = `${f.url_length || 0}`;
  document.getElementById("diag-length-sub").textContent = `Domain: ${f.domain_length || 0} | Path: ${f.path_length || 0}`;

  document.getElementById("diag-entropy").textContent = `${f.domain_entropy ? f.domain_entropy.toFixed(2) : '0.00'}`;

  document.getElementById("diag-tokens").textContent = `${f.n_suspicious_words || 0}`;
  document.getElementById("diag-tokens-sub").textContent = f.has_suspicious_word ? "Keywords Found" : "No Flags";

  document.getElementById("diag-host").textContent = f.has_ip ? "Raw IPv4" : `${f.subdomain_level || 0} Sub`;
  document.getElementById("diag-host-sub").textContent = f.has_ip ? "Dangerous" : "Standard Domain";

  document.getElementById("diag-proto").textContent = f.is_https ? "HTTPS" : "HTTP";

  document.getElementById("diag-redirect").textContent = f.uses_shortener ? "Shortener" : (f.has_redirect ? "Redirect" : "Direct");

  // Populate Collapsible 25-Feature Table
  const tbody = document.getElementById("features-table-body");
  tbody.innerHTML = "";
  data.detailed_features.forEach(item => {
    const tr = document.createElement("tr");
    const riskBadge = item.risky
      ? `<span class="badge-risk badge-risk-danger">RISK TRIGGER</span>`
      : `<span class="badge-risk badge-risk-safe">SAFE</span>`;

    tr.innerHTML = `
      <td style="font-family: var(--font-mono); font-weight: 600; color: #fff;">${item.name}</td>
      <td style="font-family: var(--font-mono); color: var(--cyan-primary);">${item.value}</td>
      <td>${riskBadge}</td>
      <td>${item.description}</td>
    `;
    tbody.appendChild(tr);
  });
}

/* ==========================================================================
   3. Benchmarks Controller
   ========================================================================== */
async function initBenchmarks() {
  const tbody = document.getElementById("metrics-table-body");
  if (!tbody) return;

  try {
    const res = await fetch("/api/metrics");
    const json = await res.json();

    if (json.status === "ok" && json.metrics) {
      tbody.innerHTML = "";
      for (const [expName, m] of Object.entries(json.metrics)) {
        const tr = document.createElement("tr");
        const isCross = expName.includes("to") || expName.includes("exp2");
        const label = isCross
          ? `<strong style="color: var(--danger-crimson);">⚠️ ${expName}</strong>`
          : `<strong>${expName}</strong>`;

        tr.innerHTML = `
          <td>${label}</td>
          <td style="font-family: var(--font-mono);">${(m.accuracy !== undefined ? (m.accuracy * 100).toFixed(1) : "-")}%</td>
          <td style="font-family: var(--font-mono);">${(m.precision !== undefined ? (m.precision * 100).toFixed(1) : "-")}%</td>
          <td style="font-family: var(--font-mono);">${(m.recall !== undefined ? (m.recall * 100).toFixed(1) : "-")}%</td>
          <td style="font-family: var(--font-mono); color: var(--safe-emerald); font-weight: 700;">${(m.f1 !== undefined ? (m.f1 * 100).toFixed(1) : "-")}%</td>
          <td style="font-family: var(--font-mono);">${m.roc_auc !== undefined ? m.roc_auc.toFixed(3) : "-"}</td>
          <td style="font-family: var(--font-mono); color: var(--text-muted);">${m.false_positive_rate !== undefined ? (m.false_positive_rate * 100).toFixed(1) + "%" : "-"}</td>
        `;
        tbody.appendChild(tr);
      }
    }
  } catch (err) {
    console.warn("Metrics load failed:", err);
  }
}

/* ==========================================================================
   4. Modal Controller
   ========================================================================== */
function initModal() {
  const modal = document.getElementById("image-modal");
  const closeBtn = document.getElementById("modal-close-btn");
  const modalImg = document.getElementById("modal-img-src");
  const modalTitle = document.getElementById("modal-img-title");

  // Bind curated chart tiles
  document.querySelectorAll(".chart-tile").forEach(tile => {
    tile.addEventListener("click", () => {
      const img = tile.getAttribute("data-img");
      const title = tile.getAttribute("data-title");
      if (img) {
        modalImg.src = img;
        modalTitle.textContent = title || "Visualization Preview";
        modal.style.display = "flex";
      }
    });
  });

  if (closeBtn) {
    closeBtn.addEventListener("click", () => { modal.style.display = "none"; });
  }

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.style.display = "none";
    });
  }
}
