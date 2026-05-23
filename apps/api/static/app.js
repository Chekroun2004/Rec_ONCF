/**
 * ONCF — Recommandation de trajet · Demo UI
 * ES module — no build step, no external dependencies.
 *
 * Privacy: code_client goes ONLY in the POST body.
 *          It is never stored in localStorage/sessionStorage/cookies
 *          and never appears in any URL or query string.
 */

// ── Model selector state ───────────────────────────────────────
let selectedVariant = "d"; // default: most recent

async function fetchModels() {
  try {
    const res = await fetch("/models");
    if (!res.ok) return;
    const models = await res.json();
    buildModelSelector(models);
  } catch (_) {
    // server not up yet — leave loading text
  }
}

function buildModelSelector(models) {
  const grid = document.getElementById("model-grid");
  if (!grid) return;
  grid.innerHTML = "";

  for (const model of models) {
    const hr1 = model.metrics?.["hit_rate@1"];
    const hr1Pct = hr1 !== undefined ? (hr1 * 100).toFixed(1) + "%" : "—";
    const isSelected = model.variant === selectedVariant;

    const lbl = document.createElement("label");
    lbl.className = "model-card-label";
    lbl.innerHTML = `
      <input type="radio" name="model_variant" value="${escapeHtml(model.variant)}"
        ${isSelected ? "checked" : ""} ${!model.available ? "disabled" : ""}
        aria-label="${escapeHtml(model.label)}" />
      <div class="model-card">
        <div class="model-card-top-row">
          <span class="model-variant-badge">${escapeHtml(model.variant.toUpperCase())}</span>
          ${model.is_default ? '<span class="model-default-chip">Défaut</span>' : ""}
        </div>
        <div class="model-name">${escapeHtml(model.label)}</div>
        <div class="model-metrics-row">
          <span class="model-metric-val">${escapeHtml(hr1Pct)}</span>
          <span class="model-metric-key">HR@1</span>
        </div>
        <div class="model-desc">${escapeHtml(model.description)}</div>
      </div>
    `;

    const radio = lbl.querySelector("input[type='radio']");
    radio.addEventListener("change", () => {
      selectedVariant = model.variant;
      updateMetricsBar(model);
    });

    grid.appendChild(lbl);
    if (isSelected) updateMetricsBar(model);
  }
}

function updateMetricsBar(model) {
  const bar = document.getElementById("selected-model-metrics");
  const hr1El = document.getElementById("hr1-val");
  const hr3El = document.getElementById("hr3-val");
  const mrr3El = document.getElementById("mrr3-val");
  if (!bar) return;

  const m = model.metrics || {};
  const fmt = (v) => v ? (v * 100).toFixed(2) + "%" : "—";
  if (hr1El) hr1El.textContent = fmt(m["hit_rate@1"]);
  if (hr3El) hr3El.textContent = fmt(m["hit_rate@3"]);
  if (mrr3El) mrr3El.textContent = fmt(m["mrr@3"]);
  bar.removeAttribute("hidden");
}

// ── DOM refs ───────────────────────────────────────────────────
const form          = /** @type {HTMLFormElement}     */ (document.getElementById("recommend-form"));
const codeInput     = /** @type {HTMLInputElement}    */ (document.getElementById("code-client"));
const codeError     = /** @type {HTMLElement}         */ (document.getElementById("code-client-error"));
const submitBtn     = /** @type {HTMLButtonElement}   */ (document.getElementById("submit-btn"));
const errorBanner   = /** @type {HTMLElement}         */ (document.getElementById("error-banner"));
const errorMessage  = /** @type {HTMLElement}         */ (document.getElementById("error-message"));
const emptyState    = /** @type {HTMLElement}         */ (document.getElementById("empty-state"));
const cardsContainer = /** @type {HTMLElement}        */ (document.getElementById("cards-container"));
const noResults     = /** @type {HTMLElement}         */ (document.getElementById("no-results"));
const resultFooter  = /** @type {HTMLElement}         */ (document.getElementById("result-footer"));
const modeBadge     = /** @type {HTMLElement}         */ (document.getElementById("mode-badge"));
const requestIdEl   = /** @type {HTMLElement}         */ (document.getElementById("request-id-display"));
const latencyEl     = /** @type {HTMLElement}         */ (document.getElementById("latency-display"));

// ── Helpers ────────────────────────────────────────────────────

/**
 * Show or hide a DOM element using the `hidden` attribute.
 * @param {HTMLElement} el
 * @param {boolean} visible
 */
function setVisible(el, visible) {
  if (visible) { el.removeAttribute("hidden"); }
  else         { el.setAttribute("hidden", ""); }
}

/**
 * Clear all validation state on the code-client input.
 */
function clearValidation() {
  codeInput.classList.remove("is-invalid");
  codeError.textContent = "";
}

/**
 * Show an inline validation error on the code-client input.
 * @param {string} msg
 */
function showValidationError(msg) {
  codeInput.classList.add("is-invalid");
  codeError.textContent = msg;
  codeInput.focus();
}

/**
 * Show the error banner with a given message; hide on empty.
 * @param {string} msg
 */
function showError(msg) {
  if (!msg) {
    setVisible(errorBanner, false);
    errorMessage.textContent = "";
    return;
  }
  errorMessage.textContent = msg;
  setVisible(errorBanner, true);
}

/**
 * Set the submit button loading state.
 * @param {boolean} loading
 */
function setLoading(loading) {
  submitBtn.setAttribute("aria-busy", loading ? "true" : "false");
  submitBtn.disabled = loading;
}

/**
 * Get the currently selected k value from the segmented group.
 * @returns {number}
 */
function getK() {
  const checked = /** @type {HTMLInputElement|null} */ (
    form.querySelector("input[name='k']:checked")
  );
  return checked ? parseInt(checked.value, 10) : 3;
}

/**
 * Get the include_schedule checkbox state.
 * @returns {boolean}
 */
function getIncludeSchedule() {
  const cb = /** @type {HTMLInputElement|null} */ (
    document.getElementById("include-schedule")
  );
  return cb ? cb.checked : false;
}

// ── Mode badge rendering ───────────────────────────────────────

/** @type {Record<string, {cssClass: string, label: string, dot: string}>} */
const MODE_META = {
  model:         { cssClass: "mode-model",     label: "Modèle",                    dot: "#1E8E5A" },
  cold_start_cf: { cssClass: "mode-cold-cf",   label: "Démarrage à froid (CF)",    dot: "#C9821A" },
  popularity:    { cssClass: "mode-popularity", label: "Routes populaires",         dot: "#6B7280" },
  cold_start:    { cssClass: "mode-cold",       label: "Aucune recommandation",     dot: "#6B7280" },
};

/**
 * Render the mode badge element.
 * @param {string} mode
 */
function renderModeBadge(mode) {
  const meta = MODE_META[mode] ?? { cssClass: "mode-cold", label: mode, dot: "#6B7280" };

  // Remove previous mode classes
  modeBadge.className = "mode-badge";
  modeBadge.classList.add(meta.cssClass);

  // Dot indicator + label
  modeBadge.innerHTML = `
    <svg width="8" height="8" viewBox="0 0 8 8" fill="${meta.dot}" aria-hidden="true">
      <circle cx="4" cy="4" r="4"/>
    </svg>
    ${escapeHtml(meta.label)}
  `;
}

// ── Route label parsing ────────────────────────────────────────

/**
 * Given a raw label string (e.g. "CASA VOYAGEURS → MEKNES") or fallback id,
 * return an HTML string with the arrow styled as a distinct span.
 * @param {string} raw
 * @returns {string}
 */
function formatRouteLabel(raw) {
  // Normalize arrow variants (→, ->, –>, —>)
  const normalised = raw.replace(/\s*(→|->|–>|—>)\s*/g, " → ");
  const parts = normalised.split(" → ");
  if (parts.length === 2) {
    return `${escapeHtml(parts[0])}<span class="arrow" aria-hidden="true">→</span>${escapeHtml(parts[1])}`;
  }
  return escapeHtml(normalised);
}

// ── Schedule rendering ─────────────────────────────────────────

/**
 * Render a list of schedule items into an HTML string.
 * @param {Array<Record<string, string>>} items
 * @returns {string}
 */
function renderScheduleItems(items) {
  if (!items || items.length === 0) {
    return `<p class="schedule-unavailable">Horaire non disponible (trajet LGV ou avec correspondance)</p>`;
  }

  const rows = items.map((item) => {
    // Build display text from whatever keys exist — depart, arrive, train
    const parts = [];
    if (item.depart) {
      parts.push(`<span class="schedule-time">${escapeHtml(item.depart)}</span>`);
    }
    if (item.arrive) {
      parts.push(`<span class="schedule-sep">→</span>`);
      parts.push(`<span class="schedule-time">${escapeHtml(item.arrive)}</span>`);
    }
    // Extra fields beyond depart/arrive/train
    const knownKeys = new Set(["depart", "arrive", "train"]);
    for (const [key, val] of Object.entries(item)) {
      if (!knownKeys.has(key) && val) {
        parts.push(`<span class="schedule-sep">·</span><span>${escapeHtml(String(val))}</span>`);
      }
    }

    let trainBadge = "";
    if (item.train) {
      trainBadge = `<span class="schedule-train">Train ${escapeHtml(String(item.train))}</span>`;
    }

    return `<div class="schedule-row">${parts.join("")}${trainBadge}</div>`;
  });

  return `<div class="schedule-list">${rows.join("")}</div>`;
}

// ── Card rendering ─────────────────────────────────────────────

/**
 * Update the schedule slot for a liaison card once the lazy fetch completes.
 * @param {string} liaisonId
 * @param {Array<Record<string, string>>} items
 */
function updateScheduleSlot(liaisonId, items) {
  const slot = [...cardsContainer.querySelectorAll("[data-schedule-slot]")]
    .find(el => el.getAttribute("data-schedule-slot") === liaisonId);
  if (!slot) return;
  const loading = slot.querySelector(".schedule-loading");
  if (loading) {
    loading.insertAdjacentHTML("afterend", renderScheduleItems(items));
    loading.remove();
  }
}

/**
 * Fetch a single liaison's schedule from GET /schedule/{id} and update its card.
 * @param {string} liaisonId
 */
function lazyLoadSchedule(liaisonId) {
  fetch(`/schedule/${encodeURIComponent(liaisonId)}`)
    .then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(data => updateScheduleSlot(liaisonId, data.schedule ?? []))
    .catch(() => updateScheduleSlot(liaisonId, []));
}

/**
 * Render all route cards into the cards container.
 * @param {string[]} recommendations
 * @param {Record<string, string>} labels
 * @param {Record<string, Array<Record<string, string>>> | undefined} schedules
 * @param {boolean} [showScheduleSlots] - when true and schedules is absent, render loading placeholders
 */
function renderCards(recommendations, labels, schedules, showScheduleSlots = false) {
  cardsContainer.innerHTML = "";

  if (!recommendations || recommendations.length === 0) {
    setVisible(cardsContainer, false);
    setVisible(noResults, true);
    return;
  }

  setVisible(noResults, false);

  const rankLabels = ["#1", "#2", "#3"];
  const rankClasses = ["rank-1", "rank-2", "rank-3"];

  recommendations.forEach((id, idx) => {
    const rawLabel = (labels && labels[id]) ? labels[id] : `Liaison ${id}`;
    const headline = formatRouteLabel(rawLabel);
    const rankText = rankLabels[idx] ?? `#${idx + 1}`;
    const rankClass = rankClasses[idx] ?? "rank-3";

    let scheduleHtml = "";
    if (schedules) {
      const items = schedules[id];
      const scheduleSectionHtml = renderScheduleItems(items);
      scheduleHtml = `
        <div class="schedule-section">
          <div class="schedule-header">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
            Prochains départs
          </div>
          ${scheduleSectionHtml}
        </div>
      `;
    } else if (showScheduleSlots) {
      scheduleHtml = `
        <div class="schedule-section" data-schedule-slot="${escapeHtml(id)}">
          <div class="schedule-header">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
            Prochains départs
          </div>
          <p class="schedule-loading">Chargement…</p>
        </div>
      `;
    }

    const card = document.createElement("article");
    card.className = "route-card";
    card.setAttribute("aria-label", `Recommandation ${idx + 1} : ${rawLabel}`);
    card.innerHTML = `
      <div class="route-card-inner">
        <div class="route-card-top">
          <div class="rank-pill ${rankClass}" aria-label="Rang ${idx + 1}">${rankText}</div>
          <div class="route-info">
            <h2 class="route-headline">${headline}</h2>
            <div class="route-id">ID: ${escapeHtml(id)}</div>
          </div>
        </div>
        ${scheduleHtml}
      </div>
    `;

    cardsContainer.appendChild(card);
  });

  setVisible(cardsContainer, true);
}

// ── Main fetch logic ───────────────────────────────────────────

/**
 * Submit the recommendation request.
 */
async function submitRecommendation() {
  // 1. Validation
  clearValidation();
  showError("");

  const codeClient = codeInput.value.trim();
  if (!codeClient) {
    showValidationError("Saisissez un code client");
    return;
  }

  // 2. Collect params
  const k = getK();
  const includeSchedule = getIncludeSchedule();

  // 3. Hide previous results / show loading
  setVisible(emptyState, false);
  setVisible(cardsContainer, false);
  setVisible(noResults, false);
  setVisible(resultFooter, false);
  setLoading(true);

  // 4. Timing — client-side round trip
  const t0 = performance.now();

  try {
    // code_client goes ONLY in the POST body — never in URL or query string.
    // Schedules are NOT requested here — they lazy-load per card via GET /schedule/{id}.
    const response = await fetch(`/recommend?variant=${encodeURIComponent(selectedVariant)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code_client: codeClient, k: k }),
    });

    const latencyMs = Math.round(performance.now() - t0);

    if (!response.ok) {
      let detail = "";
      try {
        const err = await response.json();
        detail = err.detail ?? err.message ?? "";
      } catch (_) {
        // ignore parse error
      }
      const msg = detail
        ? `Erreur ${response.status} : ${detail}`
        : `Erreur HTTP ${response.status} — ${response.statusText}`;
      showError(msg);
      setVisible(emptyState, true);
      return;
    }

    /** @type {{
     *   mode: string,
     *   variant: string,
     *   request_id: string,
     *   recommendations: string[],
     *   labels?: Record<string, string>
     * }} */
    const data = await response.json();

    // 5. Render cards (with schedule loading slots if checkbox is checked)
    renderCards(data.recommendations ?? [], data.labels ?? {}, undefined, includeSchedule);

    // 6. Lazy-load schedules per card after recommendations are rendered
    if (includeSchedule && data.recommendations?.length) {
      data.recommendations.forEach(id => lazyLoadSchedule(id));
    }

    // 7. Render footer
    renderModeBadge(data.mode ?? "cold_start");
    requestIdEl.textContent = data.request_id ? truncateId(data.request_id) : "";
    requestIdEl.setAttribute("title", data.request_id ?? "");
    latencyEl.textContent = `${latencyMs} ms`;

    setVisible(resultFooter, true);
    // Re-trigger footer animation
    resultFooter.style.animation = "none";
    // eslint-disable-next-line no-unused-expressions
    resultFooter.offsetHeight; // reflow
    resultFooter.style.animation = "";

  } catch (err) {
    const msg = err instanceof TypeError
      ? "Impossible de contacter le serveur. Vérifiez que l'API est démarrée."
      : `Erreur inattendue : ${String(err)}`;
    showError(msg);
    setVisible(emptyState, true);
  } finally {
    setLoading(false);
  }
}

// ── Event listeners ────────────────────────────────────────────

form.addEventListener("submit", (e) => {
  e.preventDefault();
  submitRecommendation();
});

// Load model selector on page load
fetchModels();

// Clear validation error on input
codeInput.addEventListener("input", () => {
  if (codeInput.classList.contains("is-invalid")) clearValidation();
});

// ── Utility ────────────────────────────────────────────────────

/**
 * Escape HTML special characters to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

/**
 * Truncate a UUID for display (show first 8 chars + ellipsis).
 * @param {string} id
 * @returns {string}
 */
function truncateId(id) {
  return id.length > 8 ? `${id.slice(0, 8)}…` : id;
}
