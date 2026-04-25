"use strict";

// ── Global state ────────────────────────────────────────────────────────────
const charts = {};

const fmt = {
    pct: (v) => `${(v * 100).toFixed(2)}%`,
    money: (v) => `₹${Number(v).toLocaleString("en-IN", { maximumFractionDigits: 2 })}`,
    num: (v, d = 2) => Number(v).toFixed(d),
    date: (s) => new Date(s).toLocaleString(),
};

// ── API helper ──────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
    const res = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...opts,
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status} ${res.statusText}: ${text.slice(0, 200)}`);
    }
    return res.json();
}

function setBusy(btn, busy) {
    if (!btn) return;
    if (busy) {
        btn.disabled = true;
        btn.dataset.label = btn.textContent;
        btn.innerHTML = `<span class="spinner"></span>${btn.dataset.label}`;
    } else {
        btn.disabled = false;
        if (btn.dataset.label) btn.textContent = btn.dataset.label;
    }
}

function showError(panelId, message) {
    const panel = document.getElementById(panelId);
    let box = panel.querySelector(".error");
    if (!box) {
        box = document.createElement("div");
        box.className = "error";
        panel.querySelector(".panel-header").after(box);
    }
    box.textContent = `Error: ${message}`;
    setTimeout(() => box.remove(), 6000);
}

function metric(label, value, klass = "") {
    return `<div class="metric"><div class="label">${label}</div><div class="value ${klass}">${value}</div></div>`;
}

// ── Tab switching ───────────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        const target = tab.dataset.tab;
        document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
        document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${target}`));
        Object.values(charts).forEach((c) => c && c.resize());
    });
});

// ── Chart helpers ───────────────────────────────────────────────────────────
function initChart(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (charts[id]) charts[id].dispose();
    charts[id] = echarts.init(el, "dark", { renderer: "canvas" });
    return charts[id];
}

window.addEventListener("resize", () => {
    Object.values(charts).forEach((c) => c && c.resize());
});

// ── Health pill ─────────────────────────────────────────────────────────────
async function refreshHealth() {
    try {
        const h = await api("/api/health");
        const pillH = document.getElementById("pill-health");
        const ppoOk = h.policy && h.policy.ppo && h.policy.ppo.exists;
        const bcOk = h.policy && h.policy.bc && h.policy.bc.exists;
        const mlOk = h.ml_forecaster && h.ml_forecaster.available;
        const policyTag = ppoOk ? "PPO" : bcOk ? "BC" : "heuristic";
        pillH.textContent = `health: ${h.status} · policy: ${policyTag}${mlOk ? " · LSTM✓" : ""}`;
        pillH.classList.toggle("ok", h.status === "ok");
        pillH.classList.toggle("bad", h.status !== "ok");

        const cutoff = h.data || {};
        document.getElementById("pill-cutoff").textContent =
            `cutoff: ${cutoff.train_cutoff || "—"}  · last train: ${cutoff.train_last_date || "—"}  · live rows: ${cutoff.live_rows ?? "—"}`;
        if (cutoff.train_cutoff) {
            document.getElementById("cutoff-date-inline").textContent = cutoff.train_cutoff;
        }

        const dbPill = document.getElementById("pill-db");
        dbPill.textContent = `db: ${h.database.ready ? "connected" : "offline"}`;
        dbPill.classList.toggle("ok", h.database.ready);
        dbPill.classList.toggle("bad", !h.database.ready);
    } catch (e) {
        document.getElementById("pill-health").textContent = "health: down";
        document.getElementById("pill-health").classList.add("bad");
    }
}

// ── Forecast tab ────────────────────────────────────────────────────────────
function renderForecast(data) {
    const chart = initChart("chart-forecast");
    if (!chart) return;

    const histDates = data.history.dates;
    const histClose = data.history.close;
    const fcDates = data.forecast.dates;

    const allDates = [...histDates, ...fcDates];
    const histSeries = histClose.concat(new Array(fcDates.length).fill(null));
    const padNulls = (arr) => new Array(histDates.length).fill(null).concat(arr);

    const median = padNulls(data.forecast.median);
    const p05 = padNulls(data.forecast.p05);
    const p25 = padNulls(data.forecast.p25);
    const p75 = padNulls(data.forecast.p75);
    const p95 = padNulls(data.forecast.p95);

    const band95Lower = p05;
    const band95Width = padNulls(data.forecast.p95.map((v, i) => v - data.forecast.p05[i]));
    const band50Lower = p25;
    const band50Width = padNulls(data.forecast.p75.map((v, i) => v - data.forecast.p25[i]));

    let overlaySeries = null;
    if (data.holdout_overlay && data.holdout_overlay.dates && data.holdout_overlay.dates.length) {
        const overlayMap = {};
        data.holdout_overlay.dates.forEach((d, i) => { overlayMap[d] = data.holdout_overlay.close[i]; });
        overlaySeries = allDates.map(d => overlayMap[d] ?? null);
    }

    const legendData = ["History", "Median", "5–95%", "25–75%"];
    if (overlaySeries) legendData.push("Actual (holdout)");

    chart.setOption({
        backgroundColor: "transparent",
        textStyle: { color: "#e6edf3" },
        title: { text: `TATAGOLD.NS — ${data.horizon_days}d forecast`, left: "center", textStyle: { fontSize: 14, color: "#e6edf3" } },
        tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
        legend: { data: legendData, top: 28, textStyle: { color: "#8b949e" } },
        grid: { left: 60, right: 30, top: 70, bottom: 60 },
        xAxis: { type: "category", data: allDates, axisLabel: { color: "#8b949e" } },
        yAxis: {
            type: "value",
            scale: true,
            axisLabel: { color: "#8b949e", formatter: (v) => `₹${v.toFixed(0)}` },
            splitLine: { lineStyle: { color: "#21262d" } },
        },
        dataZoom: [{ type: "inside", start: 50, end: 100 }, { type: "slider", start: 50, end: 100, height: 20, bottom: 10 }],
        series: [
            {
                name: "5–95%",
                type: "line",
                data: band95Lower,
                lineStyle: { opacity: 0 },
                stack: "band95",
                symbol: "none",
                tooltip: { show: false },
            },
            {
                name: "5–95%",
                type: "line",
                data: band95Width,
                lineStyle: { opacity: 0 },
                stack: "band95",
                areaStyle: { color: "rgba(240,180,41,0.10)" },
                symbol: "none",
            },
            {
                name: "25–75%",
                type: "line",
                data: band50Lower,
                lineStyle: { opacity: 0 },
                stack: "band50",
                symbol: "none",
                tooltip: { show: false },
            },
            {
                name: "25–75%",
                type: "line",
                data: band50Width,
                lineStyle: { opacity: 0 },
                stack: "band50",
                areaStyle: { color: "rgba(240,180,41,0.22)" },
                symbol: "none",
            },
            {
                name: "Median",
                type: "line",
                data: median,
                smooth: true,
                lineStyle: { color: "#f0b429", width: 2 },
                itemStyle: { color: "#f0b429" },
                symbol: "none",
            },
            {
                name: "History",
                type: "line",
                data: histSeries,
                smooth: false,
                lineStyle: { color: "#58a6ff", width: 1.5 },
                itemStyle: { color: "#58a6ff" },
                symbol: "none",
            },
            ...(overlaySeries ? [{
                name: "Actual (holdout)",
                type: "line",
                data: overlaySeries,
                connectNulls: true,
                smooth: false,
                lineStyle: { color: "#f85149", width: 2.5 },
                itemStyle: { color: "#f85149" },
                symbol: "circle",
                symbolSize: 6,
                z: 10,
            }] : []),
        ],
    });

    const last = data.last_close;
    const medTerm = data.forecast.median[data.forecast.median.length - 1];
    const p05Term = data.forecast.p05[data.forecast.p05.length - 1];
    const p95Term = data.forecast.p95[data.forecast.p95.length - 1];
    const expReturn = (medTerm - last) / last;

    document.getElementById("forecast-metrics").innerHTML = [
        metric("Last close (cutoff)", fmt.money(last)),
        metric("Median terminal", fmt.money(medTerm), expReturn >= 0 ? "green" : "red"),
        metric("5% terminal", fmt.money(p05Term), "red"),
        metric("95% terminal", fmt.money(p95Term), "green"),
        metric("Expected return", fmt.pct(expReturn), expReturn >= 0 ? "green" : "red"),
        metric("σ (daily)", fmt.pct(data.sigma_daily), "accent"),
    ].join("");
}

document.getElementById("btn-forecast").addEventListener("click", async (ev) => {
    setBusy(ev.target, true);
    try {
        const method = document.getElementById("forecast-method").value;
        const horizon = document.getElementById("forecast-horizon").value;
        const sims = parseInt(document.getElementById("forecast-sims").value, 10);
        const lookback = parseInt(document.getElementById("forecast-lookback").value, 10);
        const data = await api("/api/forecast", {
            method: "POST",
            body: JSON.stringify({ method, horizon, n_simulations: sims, lookback_days: lookback }),
        });
        renderForecast(data);
    } catch (e) {
        showError("panel-forecast", e.message);
    } finally {
        setBusy(ev.target, false);
    }
});

document.getElementById("btn-eval-ml").addEventListener("click", async (ev) => {
    setBusy(ev.target, true);
    const out = document.getElementById("forecast-eval-output");
    out.innerHTML = "";
    try {
        const r = await api("/api/forecast-eval");
        if (r.error) {
            out.innerHTML = `<div class="error">${r.error}</div>`;
        } else {
            out.innerHTML = `
                <div class="metrics-row">
                    ${metric("Horizon", `${r.horizon_days}d`)}
                    ${metric("Predicted log-ret", fmt.num(r.predicted_log_return, 4), r.predicted_log_return >= 0 ? "green" : "red")}
                    ${metric("Realized log-ret", fmt.num(r.realized_log_return, 4), r.realized_log_return >= 0 ? "green" : "red")}
                    ${metric("Abs error", fmt.num(r.abs_error, 4))}
                    ${metric("Direction OK", r.directional_correct ? "yes" : "no", r.directional_correct ? "green" : "red")}
                </div>
                <div class="muted" style="margin-top:8px;font-size:12px">
                    Forecaster fed only training-cutoff bars (≤ ${fmt.date}). Compared against
                    realized post-cutoff bars from <code>${r.first_holdout_date || "—"}</code>.
                </div>`;
        }
    } catch (e) {
        out.innerHTML = `<div class="error">${e.message}</div>`;
    } finally {
        setBusy(ev.target, false);
    }
});

// ── Live Action tab ─────────────────────────────────────────────────────────
async function runLive() {
    const lookback = parseInt(document.getElementById("live-lookback").value, 10);
    const isLong = document.getElementById("live-is-long").checked;
    const data = await api("/api/live-action", {
        method: "POST",
        body: JSON.stringify({ lookback_days: lookback, is_long: isLong }),
    });

    const card = document.getElementById("live-action-card");
    card.innerHTML = `
        <div class="action-label">Recommended action</div>
        <div class="action-name ${data.action_name}">${data.action_name}</div>
        <div>Last close: <strong>${fmt.money(data.last_close)}</strong>
             · Last bar: <code>${data.last_date}</code>
             · Source: <code>${data.source}</code></div>
    `;

    const chart = initChart("chart-live");
    if (!chart) return;
    chart.setOption({
        backgroundColor: "transparent",
        textStyle: { color: "#e6edf3" },
        title: { text: `Last ${data.history.dates.length} bars`, left: "center", textStyle: { fontSize: 14, color: "#e6edf3" } },
        tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
        grid: { left: 60, right: 30, top: 50, bottom: 50 },
        xAxis: { type: "category", data: data.history.dates, axisLabel: { color: "#8b949e" } },
        yAxis: { type: "value", scale: true, axisLabel: { color: "#8b949e", formatter: (v) => `₹${v.toFixed(0)}` }, splitLine: { lineStyle: { color: "#21262d" } } },
        dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 5 }],
        series: [
            { name: "Close", type: "line", data: data.history.close, smooth: false, lineStyle: { color: "#58a6ff", width: 2 }, symbol: "none" },
            { name: "SMA(20)", type: "line", data: data.history.sma_20, smooth: true, lineStyle: { color: "#f0b429", width: 1.5, type: "dashed" }, symbol: "none" },
        ],
    });
}

document.getElementById("btn-live").addEventListener("click", async (ev) => {
    setBusy(ev.target, true);
    try { await runLive(); } catch (e) { showError("panel-live", e.message); } finally { setBusy(ev.target, false); }
});

// ── Backtest tab ────────────────────────────────────────────────────────────
async function loadTasks() {
    try {
        const tasks = await api("/api/tasks");
        const sel = document.getElementById("backtest-task");
        sel.innerHTML = tasks.map((t) => `<option value="${t.task_type}">${t.task_type} · ${t.start} → ${t.end}</option>`).join("");
    } catch (e) {
        showError("panel-backtest", e.message);
    }
}

async function runBacktest() {
    const taskType = document.getElementById("backtest-task").value;
    const data = await api("/api/backtest", {
        method: "POST",
        body: JSON.stringify({ task_type: taskType }),
    });

    document.getElementById("backtest-metrics").innerHTML = [
        metric("Score", fmt.num(data.score, 3), data.score >= 0.5 ? "green" : "red"),
        metric("Total return", fmt.pct(data.summary.total_return), data.summary.total_return >= 0 ? "green" : "red"),
        metric("Buy & hold", fmt.pct(data.summary.buy_and_hold_return)),
        metric("Sharpe", fmt.num(data.summary.sharpe), "accent"),
        metric("Max drawdown", fmt.pct(data.summary.max_drawdown), "red"),
        metric("Trades", String(data.summary.n_trades)),
    ].join("");

    const chart = initChart("chart-backtest");
    if (!chart) return;
    chart.setOption({
        backgroundColor: "transparent",
        textStyle: { color: "#e6edf3" },
        title: { text: `Backtest — ${data.task_type}`, left: "center", textStyle: { fontSize: 14, color: "#e6edf3" } },
        tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
        legend: { data: ["Agent", "Buy & Hold"], top: 28, textStyle: { color: "#8b949e" } },
        grid: { left: 70, right: 30, top: 70, bottom: 60 },
        xAxis: { type: "category", data: data.dates, axisLabel: { color: "#8b949e" } },
        yAxis: { type: "value", scale: true, axisLabel: { color: "#8b949e", formatter: (v) => `₹${(v / 1000).toFixed(0)}k` }, splitLine: { lineStyle: { color: "#21262d" } } },
        dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 10 }],
        series: [
            { name: "Agent", type: "line", data: data.equity_curve, smooth: false, lineStyle: { color: "#f0b429", width: 2 }, itemStyle: { color: "#f0b429" }, symbol: "none" },
            { name: "Buy & Hold", type: "line", data: data.buy_and_hold, smooth: false, lineStyle: { color: "#58a6ff", width: 1.5, type: "dashed" }, itemStyle: { color: "#58a6ff" }, symbol: "none" },
        ],
    });
}

document.getElementById("btn-backtest").addEventListener("click", async (ev) => {
    setBusy(ev.target, true);
    try { await runBacktest(); } catch (e) { showError("panel-backtest", e.message); } finally { setBusy(ev.target, false); }
});

// ── Holdout tab ─────────────────────────────────────────────────────────────
async function refreshHoldout() {
    const data = await api("/api/live");
    const dates = data.bars.map((b) => b.date);
    const close = data.bars.map((b) => b.close);

    document.getElementById("holdout-metrics").innerHTML = [
        metric("Live start", data.live_start),
        metric("Bars available", String(data.bars.length)),
        metric("Last bar", data.last_date || "—"),
        metric("Last close", data.bars.length ? fmt.money(close[close.length - 1]) : "—"),
    ].join("");

    const chart = initChart("chart-holdout");
    if (!chart) return;
    if (!data.bars.length) {
        chart.setOption({ title: { text: "No holdout bars yet (post-cutoff data not available)", left: "center", textStyle: { color: "#8b949e", fontSize: 14 } } });
        return;
    }
    chart.setOption({
        backgroundColor: "transparent",
        textStyle: { color: "#e6edf3" },
        title: { text: `Holdout — TATAGOLD.NS post ${data.live_start}`, left: "center", textStyle: { fontSize: 14, color: "#e6edf3" } },
        tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
        grid: { left: 60, right: 30, top: 50, bottom: 60 },
        xAxis: { type: "category", data: dates, axisLabel: { color: "#8b949e" } },
        yAxis: { type: "value", scale: true, axisLabel: { color: "#8b949e", formatter: (v) => `₹${v.toFixed(0)}` }, splitLine: { lineStyle: { color: "#21262d" } } },
        dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 10 }],
        series: [{ name: "Holdout close", type: "line", data: close, smooth: false, lineStyle: { color: "#3fb950", width: 2 }, areaStyle: { color: "rgba(63,185,80,0.10)" }, symbol: "none" }],
    });
}

document.getElementById("btn-holdout").addEventListener("click", async (ev) => {
    setBusy(ev.target, true);
    try { await refreshHoldout(); } catch (e) { showError("panel-holdout", e.message); } finally { setBusy(ev.target, false); }
});

// ── History tab ─────────────────────────────────────────────────────────────
async function refreshHistory() {
    const tbodyP = document.querySelector("#table-predictions tbody");
    const tbodyB = document.querySelector("#table-backtests tbody");
    const tbodyA = document.querySelector("#table-actions tbody");
    tbodyP.innerHTML = tbodyB.innerHTML = tbodyA.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-dim)">Loading…</td></tr>`;

    try {
        const [preds, bts, acts] = await Promise.all([
            api("/api/history/predictions"),
            api("/api/history/backtests"),
            api("/api/history/actions"),
        ]);

        tbodyP.innerHTML = preds.length
            ? preds.map((r) => `<tr><td>${fmt.date(r.created_at)}</td><td>${r.horizon_label || r.horizon_days + "d"}</td><td>${fmt.money(r.last_close)}</td><td>${fmt.money(r.median_terminal)}</td><td>${fmt.money(r.p05_terminal)} / ${fmt.money(r.p95_terminal)}</td></tr>`).join("")
            : `<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No forecasts yet</td></tr>`;

        tbodyB.innerHTML = bts.length
            ? bts.map((r) => `<tr><td>${fmt.date(r.created_at)}</td><td>${r.task_type}</td><td>${fmt.num(r.score, 3)}</td><td>${fmt.num(r.sharpe)}</td><td>${fmt.pct(r.total_return)}</td><td>${r.n_trades}</td></tr>`).join("")
            : `<tr><td colspan="6" style="text-align:center;color:var(--text-dim)">No backtests yet</td></tr>`;

        tbodyA.innerHTML = acts.length
            ? acts.map((r) => `<tr><td>${fmt.date(r.created_at)}</td><td><strong style="color:${r.action_name === 'BUY' ? 'var(--green)' : r.action_name === 'SELL' ? 'var(--red)' : 'var(--accent)'}">${r.action_name}</strong></td><td>${fmt.money(r.last_close)}</td><td>${r.is_long ? "yes" : "no"}</td><td>${r.source}</td></tr>`).join("")
            : `<tr><td colspan="5" style="text-align:center;color:var(--text-dim)">No live actions yet</td></tr>`;
    } catch (e) {
        showError("panel-history", e.message);
    }
}

document.getElementById("btn-history").addEventListener("click", refreshHistory);

// ── Grade Tasks tab ─────────────────────────────────────────────────────────
async function runGrading() {
    const methodsSel = document.getElementById("grade-methods");
    const methods = Array.from(methodsSel.selectedOptions).map(o => o.value);
    const samples = parseInt(document.getElementById("grade-samples").value, 10);
    const data = await api("/api/grade-tasks", {
        method: "POST",
        body: JSON.stringify({ methods: methods.length ? methods : null, n_samples: samples }),
    });

    if (data.error) {
        document.getElementById("grade-aggregates").innerHTML = `<div class="error">${data.error}</div>`;
        return;
    }

    const aggCards = Object.entries(data.aggregates || {}).map(([m, a]) => {
        if (a.error) return `<div class="metric"><div class="label">${m}</div><div class="value red">${a.error}</div></div>`;
        return `
            <div class="metric">
                <div class="label">${m}</div>
                <div class="value accent">${(a.mape * 100).toFixed(2)}% MAPE</div>
                <div style="font-size:11px;color:var(--text-dim);margin-top:4px">
                    Dir: ${(a.directional_accuracy * 100).toFixed(0)}% · Cal95: ${(a.calibration_95 * 100).toFixed(0)}% · n=${a.n_tasks}
                </div>
            </div>`;
    }).join("");
    document.getElementById("grade-aggregates").innerHTML = aggCards;

    const tasks = data.tasks || [];
    const dates = tasks.map(t => t.target_date);
    const actual = tasks.map(t => t.actual);
    const series = [{
        name: "Actual",
        type: "line",
        data: actual,
        lineStyle: { color: "#3fb950", width: 3 },
        symbol: "circle", symbolSize: 8,
    }];
    const colors = { gbm: "#58a6ff", ml: "#bc8cff", chronos_zs: "#f0b429", chronos_ft: "#ff7b72" };
    const methodSet = new Set();
    tasks.forEach(t => Object.keys(t.predictions).forEach(m => methodSet.add(m)));
    methodSet.forEach(m => {
        series.push({
            name: m,
            type: "line",
            data: tasks.map(t => t.predictions[m]?.predicted ?? null),
            lineStyle: { color: colors[m] || "#8b949e", width: 1.5, type: "dashed" },
            symbol: "diamond", symbolSize: 6,
        });
    });

    const chart = initChart("chart-grade");
    if (chart) {
        chart.setOption({
            backgroundColor: "transparent",
            textStyle: { color: "#e6edf3" },
            title: { text: "Predictions vs. actual on holdout dates", left: "center", textStyle: { fontSize: 14, color: "#e6edf3" } },
            tooltip: { trigger: "axis" },
            legend: { data: series.map(s => s.name), top: 28, textStyle: { color: "#8b949e" } },
            grid: { left: 60, right: 30, top: 70, bottom: 50 },
            xAxis: { type: "category", data: dates, axisLabel: { color: "#8b949e" } },
            yAxis: { type: "value", scale: true, axisLabel: { color: "#8b949e", formatter: v => `₹${v.toFixed(0)}` }, splitLine: { lineStyle: { color: "#21262d" } } },
            series,
        });
    }

    const cols = ["Date", "Actual", ...Array.from(methodSet).map(m => `${m} (pred / err%)`)];
    const rows = tasks.map(t => {
        const cells = [t.target_date, fmt.money(t.actual)];
        Array.from(methodSet).forEach(m => {
            const p = t.predictions[m] || {};
            if (p.error) cells.push(`<span style="color:var(--red)">${p.error.slice(0, 30)}</span>`);
            else cells.push(`${fmt.money(p.predicted)} <span style="color:var(--text-dim)">(${(p.ape * 100).toFixed(2)}%)</span>`);
        });
        return `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
    }).join("");
    document.getElementById("grade-table-wrap").innerHTML = `
        <table>
            <thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
}

document.getElementById("btn-grade").addEventListener("click", async (ev) => {
    setBusy(ev.target, true);
    try { await runGrading(); } catch (e) { showError("panel-grade", e.message); } finally { setBusy(ev.target, false); }
});

// ── Boot ────────────────────────────────────────────────────────────────────
(async () => {
    await refreshHealth();
    await loadTasks();
    document.getElementById("btn-forecast").click();
    setInterval(refreshHealth, 30000);
})();
