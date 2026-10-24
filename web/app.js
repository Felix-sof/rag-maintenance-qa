/* Maintenance Copilot - the web front end of api.py. Plain JavaScript, no build step.
 *
 * Security: everything the server returns - the model's answer above all - is
 * treated as untrusted text. It is escaped before the few markdown forms the
 * answers use (bold, lists, citations) are turned into markup, and every other
 * value goes in through textContent. The page's Content-Security-Policy (set
 * by api.py) forbids inline scripts, so injected markup couldn't run anyway.
 */
"use strict";

// The pipeline's fixed sentences (rag.py): shown in English on this page.
const NO_ANSWER = "Bilgi bankasında bu sorunun cevabı yok.";
const ERROR_PREFIX = "Cevap üretilemedi:";
const MAX_HISTORY_TURNS = 4;    // api.py accepts at most 4 turns...
const MAX_TURN_CHARS = 2000;    // ...of at most 2,000 characters each

const MODE_NAMES = { HDF: "Heat dissipation", PWF: "Power", OSF: "Overstrain", TWF: "Tool wear", RNF: "Random" };
const OUTCOME = {
  answered: ["Answered", "badge-ok"],
  refused_gate: ["Out of scope", "badge-warn"],
  refused_llm: ["Not in the documents", "badge-warn"],
  error: ["Error", "badge-danger"],
};

const state = { turns: [], selected: -1, busy: false };
const $ = (selector, root = document) => root.querySelector(selector);

// ---------- small DOM helpers ----------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

const ICON_WARN = '<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" d="M8 5.5v3.2M8 11h.01M7.1 2.6 1.6 12a1 1 0 0 0 .9 1.5h11a1 1 0 0 0 .9-1.5L8.9 2.6a1 1 0 0 0-1.8 0Z"/></svg>';

// ---------- safe markdown, with citation chips ----------

function inline(escaped) {
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // [1], [1, 3], [T1] - each number its own chip
    .replace(/\[((?:T?\d+)(?:\s*,\s*T?\d+)*)\]/g, (_, group) =>
      group.split(/\s*,\s*/).map((ref) =>
        `<button type="button" class="cite${ref.startsWith("T") ? " cite-tool" : ""}" data-ref="${ref}">${ref}</button>`
      ).join(""));
}

function renderMarkdown(text) {
  const lines = escapeHtml(text).split(/\r?\n/);
  const out = [];
  let list = null;
  let paragraph = [];
  const flushParagraph = () => { if (paragraph.length) { out.push(`<p>${inline(paragraph.join(" "))}</p>`); paragraph = []; } };
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of lines) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)$/);
    const numbered = line.match(/^\d+[.)]\s+(.*)$/);
    const heading = line.match(/^#{1,4}\s+(.*)$/);
    if (bullet || numbered) {
      flushParagraph();
      const kind = bullet ? "ul" : "ol";
      if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline((bullet || numbered)[1])}</li>`);
    } else if (heading) {
      flushParagraph(); closeList();
      out.push(`<h4>${inline(heading[1])}</h4>`);
    } else if (!line) {
      flushParagraph(); closeList();
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  flushParagraph(); closeList();
  return out.join("");
}

function displayAnswer(result) {
  const answer = (result.answer || "").trim();
  if (answer === NO_ANSWER) return "The knowledge base doesn't answer this question.";
  if (result.error) return "Couldn't generate an answer: " + answer.replace(ERROR_PREFIX, "").trim();
  return answer;
}

// ---------- tool result cards ----------

function meter(condition) {
  // the value against its threshold: the tick in the middle is the limit
  const ratio = condition.limit ? Math.max(0, Math.min(condition.value / condition.limit, 2)) : 0;
  const bar = h("i");
  bar.style.width = `${(ratio / 2) * 100}%`;
  return h("div", { class: `meter${condition.met ? " met" : ""}`, title: `${condition.value} vs ${condition.limit}` }, bar);
}

function assessmentCard(n, result) {
  if (result.error) return errorCard(n, "assess_process", result.error);
  const modes = Object.entries(result.modes).map(([code, mode]) => {
    let state, label, body = null;
    if (mode.triggered === null) {
      const rate = mode.dataset_rate ?? mode.probability;
      state = "rate"; label = `${(rate * 100).toFixed(1)}%`;
      body = h("ul", { class: "conditions" }, h("li", {},
        code === "TWF" ? `Share of recorded processes at this wear that failed by TWF` : `Random, at any reading`));
    } else {
      state = mode.triggered ? "on" : "off"; label = mode.triggered ? "Triggered" : "Clear";
      body = h("ul", { class: "conditions" }, mode.conditions.map((c) =>
        h("li", {}, c.condition, h("br"), h("span", { class: "val" }, fmt(c.value)), ` · margin ${fmt(c.margin)}`, meter(c))));
    }
    return h("div", { class: `mode ${state}` },
      h("div", { class: "mode-top" }, h("span", { class: "mode-code" }, code), h("span", { class: `mode-state ${state}` }, label)),
      h("div", { class: "mode-name" }, MODE_NAMES[code] || code),
      body);
  });
  const verdict = result.triggered.length
    ? h("span", { class: "badge badge-danger" }, `Failure expected: ${result.triggered.join(", ")}`)
    : h("span", { class: "badge badge-ok" }, "No rule-based failure");
  const similar = result.similar_recorded_processes;
  const foot = [];
  if (similar && similar.compared) {
    const byMode = Object.entries(similar.failed_by_mode).filter(([, count]) => count).map(([m, c]) => `${m} ${c}`).join(", ");
    foot.push(h("span", {}, h("strong", {}, `${similar.failed} of ${similar.compared}`),
      ` most similar recorded processes failed${byMode ? ` (${byMode})` : ""}`));
  }
  const d = result.derived;
  foot.push(h("span", { class: "muted" }, `ΔT ${fmt(d.temperature_difference_k)} K · power ${fmt(d.power_w)} W · wear × torque ${fmt(d.strain_min_nm)} minNm`));
  if (result.outside_data_range.length) {
    foot.push(h("span", { class: "badge badge-warn" }, `Outside the recorded range: ${result.outside_data_range.join(", ")}`));
  }
  return h("div", { class: "tool-card", id: null, dataset: { ref: `T${n}` } },
    h("div", { class: "tool-card-head" }, h("span", { class: "cite cite-tool" }, `T${n}`), h("strong", {}, "Process assessment"),
      h("span", { class: "muted" }, `Type ${result.inputs.product_type}`), h("span", { class: "spacer" }), verdict),
    h("div", { class: "modes" }, modes),
    h("div", { class: "tool-card-foot" }, foot));
}

const NUMERIC = (value) => typeof value === "number";
const fmt = (value) => (typeof value === "number" ? value.toLocaleString("en-US", { maximumFractionDigits: 3 }) : String(value));

function tableOf(rows, { barColumn } = {}) {
  if (!rows.length) return h("p", { class: "empty-row" }, "No rows yet.");
  const columns = Object.keys(rows[0]);
  const max = barColumn ? Math.max(...rows.map((r) => Number(r[barColumn]) || 0), 1) : 0;
  return h("table", { class: "data" },
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: NUMERIC(rows[0][c]) ? "num" : null }, c.replaceAll("_", " "))))),
    h("tbody", {}, rows.map((row) => h("tr", {}, columns.map((c) => {
      const value = row[c];
      const text = value === null || value === undefined ? "–" : Array.isArray(value) ? value.join(", ") : fmt(value);
      if (c === barColumn) {
        const bar = h("span", { class: "bar" });
        bar.style.width = `${Math.round((Number(value) / max) * 80)}px`;
        return h("td", {}, h("span", { class: "bar-cell" }, bar, text));
      }
      return h("td", { class: NUMERIC(value) ? "num" : null }, text);
    })))));
}

function datasetCard(n, result) {
  if (result.error) return errorCard(n, "query_dataset", result.error);
  return h("div", { class: "tool-card", dataset: { ref: `T${n}` } },
    h("div", { class: "tool-card-head" }, h("span", { class: "cite cite-tool" }, `T${n}`), h("strong", {}, "Dataset query"),
      h("code", {}, result.query), h("span", { class: "spacer" }), h("span", { class: "muted" }, "AI4I 2020 · synthetic")),
    tableOf(result.rows || []));
}

function errorCard(n, name, message) {
  return h("div", { class: "tool-card", dataset: { ref: `T${n}` } },
    h("div", { class: "tool-card-head" }, h("span", { class: "cite cite-tool" }, `T${n}`), h("strong", {}, name),
      h("span", { class: "spacer" }), h("span", { class: "badge badge-danger" }, "Tool error")),
    h("div", { class: "tool-card-foot" }, message));
}

function toolCard(n, call) {
  return call.name === "assess_process" ? assessmentCard(n, call.result) : datasetCard(n, call.result);
}

// ---------- messages ----------

function notice(kind, html) {
  const el = h("div", { class: `notice notice-${kind}`, role: "note" });
  el.innerHTML = ICON_WARN + `<div>${html}</div>`; // html is built only from escaped values below
  return el;
}

function answerElement(index) {
  const { result } = state.turns[index];
  const [label, badge] = OUTCOME[result.outcome] || ["", ""];
  const timing = result.timings && result.timings.total_ms ? `${(result.timings.total_ms / 1000).toFixed(1)} s` : null;
  const body = h("div", { class: "answer-body" });
  body.innerHTML = renderMarkdown(displayAnswer(result));

  const foot = [];
  if (result.conflict) {
    foot.push(notice("warn", "<strong>The sources disagree.</strong> Both versions are cited above, with the reason one was taken to apply."));
  }
  if (result.unsupported_numbers && result.unsupported_numbers.length) {
    foot.push(notice("danger", `<strong>Unverified values:</strong> ${result.unsupported_numbers.map(escapeHtml).join(", ")} ` +
      "appear in none of the cited sources. Check them before acting on them."));
  }
  (result.tool_calls || []).forEach((call, i) => foot.push(toolCard(i + 1, call)));

  const el = h("article", { class: "answer", dataset: { index } },
    h("div", { class: "answer-head" },
      h("strong", {}, "Copilot"),
      label ? h("span", { class: `badge ${badge}` }, label) : null,
      h("span", { class: "spacer" }),
      result.model ? h("span", {}, result.model) : null,
      timing ? h("span", {}, timing) : null,
      result.sources && result.sources.length ? h("span", {}, `${(result.cited || []).length}/${result.sources.length} sources cited`) : null),
    body,
    foot.length ? h("div", { class: "answer-foot" }, foot) : null);
  el.addEventListener("click", (event) => {
    const chip = event.target.closest(".cite[data-ref]");
    select(index, chip ? chip.dataset.ref : null);
  });
  return el;
}

function addUserMessage(text) {
  $("#thread").append(h("div", { class: "msg msg-user" }, h("div", { class: "bubble" }, text)));
}

function addWorking() {
  const el = h("div", { class: "msg" }, h("div", { class: "answer" }, h("div", { class: "working" },
    h("span", { class: "dots" }, h("i"), h("i"), h("i")),
    h("span", {}, "Searching the documents and checking the rules… this can take 10–30 seconds."))));
  $("#thread").append(el);
  return el;
}

function scrollToEnd() {
  const thread = $("#thread");
  thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });
}

// ---------- inspector ----------

function select(index, ref = null) {
  state.selected = index;
  document.querySelectorAll(".answer.selected").forEach((el) => el.classList.remove("selected"));
  const answer = document.querySelector(`.answer[data-index="${index}"]`);
  if (answer) answer.classList.add("selected");
  renderInspector(index);
  if (ref) {
    const target = document.querySelector(`#inspector-body [data-ref="${ref}"]`);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "nearest" });
      target.classList.add("flash", "open");
      setTimeout(() => target.classList.remove("flash"), 1600);
    }
  }
}

function renderInspector(index) {
  const body = $("#inspector-body");
  body.replaceChildren();
  const turn = state.turns[index];
  if (!turn) return;
  const { result } = turn;
  $("#inspector-sub").textContent = result.search_query && result.search_query !== result.question
    ? `Searched for: “${result.search_query}”` : "Sources and tool results for the selected answer.";

  const calls = result.tool_calls || [];
  if (calls.length) {
    body.append(h("p", { class: "section-label" }, "Tool results"));
    calls.forEach((call, i) => {
      const r = call.result;
      const summary = r.error ? `Error: ${r.error}`
        : call.name === "assess_process" ? (r.triggered.length ? `Triggered: ${r.triggered.join(", ")}` : "No rule-based failure")
        : `${(r.rows || []).length} rows from ${r.query}`;
      body.append(h("div", { class: "source cited", dataset: { ref: `T${i + 1}` } },
        h("div", { class: "source-top" }, h("span", { class: "source-n" }, `T${i + 1}`), h("span", { class: "source-title" }, call.name)),
        h("div", { class: "source-text" }, summary),
        h("div", { class: "source-meta" }, Object.entries(call.args || {}).map(([k, v]) => h("span", {}, `${k} = ${v}`)))));
    });
  }

  const sources = result.sources || [];
  if (!sources.length) {
    body.append(h("div", { class: "inspector-empty" }, "No passages were retrieved for this question."));
    return;
  }
  body.append(h("p", { class: "section-label" }, "Passages"));
  const cited = new Set(result.cited || []);
  for (const source of sources) {
    const ranks = source.ranks || {};
    const meta = [`similarity ${source.score.toFixed(3)}`];
    if (ranks.rerank !== null && ranks.rerank !== undefined) meta.push(`rerank ${ranks.rerank.toFixed(2)}`);
    if (ranks.vector) meta.push(`vector #${ranks.vector}`);
    if (ranks.bm25) meta.push(`keyword #${ranks.bm25}`);
    if (source.doc_type === "example") meta.push("example document");
    const card = h("div", { class: `source${cited.has(source.n) ? " cited" : ""}`, dataset: { ref: String(source.n) } },
      h("div", { class: "source-top" }, h("span", { class: "source-n" }, String(source.n)),
        h("span", { class: "source-title" }, source.section.split(" > ").pop()),
        cited.has(source.n) ? h("span", { class: "badge badge-ok" }, "cited") : null),
      h("div", { class: "source-path" }, `${source.source} › ${source.section}`),
      h("div", { class: "source-text" }, source.text.replace(/^[^\n]*\n+/, "").replaceAll("**", "")),
      h("div", { class: "source-meta" }, meta.map((m) => h("span", {}, m))));
    card.append(h("button", { type: "button", class: "more", onclick: () => card.classList.toggle("open") }, "Show full passage"));
    body.append(card);
  }
}

// ---------- asking ----------

async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  $("#send").disabled = true;
  $("#empty")?.remove();
  addUserMessage(question);
  const working = addWorking();
  scrollToEnd();

  const history = state.turns.slice(-MAX_HISTORY_TURNS).map((t) => ({
    question: t.question.slice(0, MAX_TURN_CHARS),
    answer: (t.result.answer || "").slice(0, MAX_TURN_CHARS),
  }));
  let result;
  try {
    const response = await fetch("ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    if (response.status === 429) {
      const wait = Number(response.headers.get("Retry-After") || 60);
      throw new Error(`The demo's limit is reached (5 questions a minute, 20 answers a day). Try again in about ${Math.max(1, Math.round(wait / 60))} min.`);
    }
    if (response.status === 422) throw new Error("The question is too long: at most 500 characters.");
    if (!response.ok) throw new Error(`The server answered ${response.status}.`);
    result = await response.json();
  } catch (error) {
    result = { question, answer: error.message, outcome: "error", error: true, sources: [], cited: [], tool_calls: [], network: true };
  }
  working.remove();

  const index = state.turns.length;
  state.turns.push({ question, result });
  if (result.network) state.turns.pop(); // a failed request has no answer to follow up on
  const el = result.network
    ? h("div", { class: "msg" }, notice("danger", escapeHtml(result.answer)))
    : h("div", { class: "msg" }, answerElement(index));
  $("#thread").append(el);
  if (!result.network) select(index);
  scrollToEnd();
  state.busy = false;
  $("#send").disabled = false;
  $("#question").focus();
}

// ---------- analytics ----------

async function loadAnalytics() {
  const root = $("#analytics-body");
  let stats;
  try {
    const response = await fetch("stats");
    if (response.status === 503) {
      root.replaceChildren(notice("info", "Query logging is off on this deployment: it needs a PostgreSQL database (<code>DATABASE_URL</code>)."));
      return;
    }
    if (!response.ok) throw new Error(`The server answered ${response.status}.`);
    stats = await response.json();
  } catch (error) {
    root.replaceChildren(notice("danger", escapeHtml(error.message)));
    return;
  }
  const daily = stats.daily_volume || [];
  const total = (stats.refusal_rate || []).find((r) => r.source === "all");
  const answered = (stats.latency_percentiles || []).find((r) => r.outcome === "answered");
  const kpi = (value, label) => h("div", { class: "kpi" }, h("span", { class: "kpi-value" }, value), h("span", { class: "kpi-label" }, label));
  const panel = (title, note, content) => h("div", { class: "panel" },
    h("div", { class: "panel-head" }, h("h3", {}, title), h("span", {}, note)), content);
  root.replaceChildren(
    h("div", { class: "kpis" },
      kpi(String(daily.reduce((sum, d) => sum + d.questions, 0)), "questions in the last 30 days"),
      kpi(total && total.questions ? `${Math.round(total.refusal_rate * 100)}%` : "–", "refused as not covered by the documents"),
      kpi(answered ? `${(answered.p50_ms / 1000).toFixed(1)} s` : "–", "median time to an answer (p50)"),
      kpi(answered ? `${(answered.p95_ms / 1000).toFixed(1)} s` : "–", "slowest 5% of answers (p95)")),
    h("div", { class: "grid-2" },
      panel("Questions per day", "01_daily_volume.sql", tableOf(daily, { barColumn: "questions" })),
      panel("Most retrieved documents", "04_top_documents.sql", tableOf(stats.top_documents || [], { barColumn: "retrieved" }))),
    h("div", { class: "grid-2" },
      panel("Latency by outcome (ms)", "03_latency_percentiles.sql · percentile_cont", tableOf(stats.latency_percentiles || [])),
      panel("Refusals by source", "02_refusal_rate.sql · ROLLUP", tableOf(stats.refusal_rate || []))));
}

// ---------- status, routing, composer ----------

async function loadStatus() {
  const status = $("#status");
  try {
    const response = await fetch("health");
    const health = await response.json();
    const index = health.index || {};
    status.classList.add("ok");
    $(".status-text", status).textContent =
      `${index.backend || "chroma"} · ${index.chunks ?? "?"} chunks · ${health.retrieval_mode} · ${health.model}`;
  } catch {
    status.classList.add("down");
    $(".status-text", status).textContent = "API unreachable";
  }
}

function route() {
  const view = (location.hash || "#copilot").slice(1);
  const known = ["copilot", "analytics", "about"].includes(view) ? view : "copilot";
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${known}`; });
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === known));
  if (known === "analytics") loadAnalytics();
}

function setupComposer() {
  const textarea = $("#question");
  const resize = () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
    $("#counter").textContent = `${textarea.value.length} / 500`;
  };
  textarea.addEventListener("input", resize);
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("#composer").requestSubmit();
    }
  });
  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    const question = textarea.value.trim();
    if (!question) return;
    textarea.value = "";
    resize();
    ask(question);
  });
  document.querySelectorAll(".suggestion").forEach((button) =>
    button.addEventListener("click", () => ask(button.dataset.question)));
}

window.addEventListener("hashchange", route);
document.addEventListener("DOMContentLoaded", () => {
  setupComposer();
  route();
  loadStatus();
});
