"use strict";

const byId = (id) => document.getElementById(id);
const actions = document.querySelectorAll(".action");
const positionNames = {
  GK: "Goalkeeper",
  DEF: "Defender",
  MID: "Midfielder",
  FWD: "Forward",
};
const positionOrder = Object.keys(positionNames);
const state = {
  status: null,
  result: null,
  busy: false,
  refreshing: false,
  refreshRequested: false,
  pollTimer: null,
  lastRequest: null,
  file: null,
};
const dateFormat = new Intl.DateTimeFormat(undefined, {
  weekday: "short",
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
  timeZoneName: "short",
});
const pointsFormat = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const percentageFormat = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 0,
});

function create(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content != null) element.textContent = content;
  return element;
}

function showText(id, message) {
  const element = byId(id);
  element.textContent = message || "";
  element.hidden = !message;
}

function showRequest(message, error = false) {
  showText("request-message", message);
  byId("request-message").classList.toggle("is-error", error);
}

function formatDate(value) {
  const date = new Date(value);
  return value && Number.isFinite(date.getTime()) ? dateFormat.format(date) : null;
}

function points(value) {
  return Number.isFinite(value) ? pointsFormat.format(value) : "—";
}

function setButtonState() {
  const inactive = state.status?.state !== "ready" || state.busy || state.refreshing;
  for (const action of actions) action.disabled = inactive;
  byId("recommendation").setAttribute("aria-busy", String(state.busy));
  byId("refresh").disabled = state.busy || state.refreshing || state.status?.state === "loading";
}

async function api(url, payload) {
  const response = await fetch(url, payload === undefined ? { cache: "no-store" } : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) {
    const error = new Error(data?.error || `The application returned HTTP ${response.status}.`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function renderStatus(status) {
  state.status = status;
  const phase = status.state;
  const loading = phase === "loading";
  const timestamp = formatDate(status.updated_at);
  const deadline = formatDate(status.deadline);
  byId("data-status").dataset.state = phase;
  const label = phase === "ready" ? "Data ready" : loading ? "Updating data" : "Data unavailable";
  byId("data-status-label").textContent = label;
  byId("data-status").setAttribute("aria-label", label);
  byId("season").textContent = status.season ? `${status.season} season` : "Upcoming Gameweek";
  byId("gameweek").textContent = status.gameweek ? `Gameweek ${status.gameweek}` : "—";
  byId("deadline").textContent = deadline ? `Deadline ${deadline}` : "Deadline unavailable";
  showText("data-date", timestamp && `FPL data updated ${timestamp}`);
  showText("data-message", loading || phase === "error" ? status.message || (
    loading ? "Preparing Gameweek projections. The first run may take several minutes." : "FPL data is unavailable. Use Refresh data to retry."
  ) : null);
  byId("data-message").classList.toggle("notice-error", phase === "error");
  showText("data-warning", status.warning);

  const coverage = status.coverage;
  if (coverage) {
    const parts = [`${coverage.seasons} seasons`, `${coverage.first}–${coverage.last}`];
    byId("coverage").textContent = parts.join(" · ");
  } else {
    byId("coverage").textContent = loading ? "Waiting for historical coverage…" : "Historical coverage unavailable.";
  }
  const metrics = status.metrics;
  byId("model-metrics").hidden = !metrics;
  byId("metric-help").hidden = !metrics;
  if (metrics) {
    byId("model-mae").textContent = points(metrics.mae);
    byId("baseline-mae").textContent = points(metrics.baseline_mae);
  }
  if (state.result) renderLineupWarning(state.result);
  setButtonState();
}

function schedulePoll(delay = 3500) {
  window.clearTimeout(state.pollTimer);
  state.pollTimer = window.setTimeout(fetchStatus, delay);
}

function applyStatus(status) {
  renderStatus(status);
  if (status.state === "loading") schedulePoll();
  if (status.state === "ready" && state.refreshRequested) {
    state.refreshRequested = false;
    if (state.lastRequest) optimize(state.lastRequest, false);
  }
  if (status.state === "error") state.refreshRequested = false;
}

async function fetchStatus() {
  try {
    applyStatus(await api("/api/status"));
  } catch (error) {
    renderStatus({
      ...state.status,
      state: "error",
      message: `Could not load Gameweek status. ${error.message}`,
    });
    schedulePoll(10000);
  }
}

function fixtureLabel(fixture) {
  return `${fixture.opponent} (${fixture.home ? "H" : "A"})`;
}

function openPlayer(player) {
  const position = positionNames[player.position] || player.position;
  const role = player.role === "C" ? "Captain" : player.role === "VC" ? "Vice-captain" : null;
  byId("detail-name").textContent = player.name;
  byId("detail-position").textContent = [position, player.team, role].filter(Boolean).join(" · ");

  const availability = byId("detail-availability");
  const metrics = [
    ["Any minutes", player.appearance_chance],
    ["Starts", player.start_chance],
    ["60+ minutes", player.sixty_chance],
  ];
  availability.replaceChildren(...metrics.map(([label, value]) => {
    const metric = create("div");
    metric.append(create("dt", "", label), create(
      "dd", "", Number.isFinite(value) ? percentageFormat.format(value) : "—",
    ));
    return metric;
  }));

  const fixtures = byId("detail-fixtures");
  fixtures.replaceChildren();
  for (const fixture of player.fixtures) {
    const row = create("div", "fixture-row");
    const venue = fixture.home ? "home" : "away";
    const difficulty = Number.isFinite(fixture.difficulty) ? ` · FDR ${fixture.difficulty}/5` : "";
    row.append(
      create("span", "", `${fixture.opponent_name || fixture.opponent} (${venue})${difficulty}`),
      create("strong", "", `${points(fixture.points)} pts`),
    );
    fixtures.append(row);
  }
  const total = create("div", "fixture-row");
  total.append(
    create("span", "", player.fixtures.length ? "Gameweek projection" : "Blank Gameweek"),
    create("strong", "", `${points(player.points)} pts`),
  );
  fixtures.append(total);
  byId("player-dialog").showModal();
}

function fixtureText(player) {
  return player.fixtures.length ? player.fixtures.map(fixtureLabel).join(", ") : "Blank";
}

function addPlayerInteraction(tile, player, label = null) {
  tile.type = "button";
  tile.addEventListener("click", () => openPlayer(player));
  const role = player.role === "C" ? "captain" : player.role === "VC" ? "vice-captain" : null;
  const summary = [player.name, role, label, fixtureText(player), `${points(player.points)} projected points`];
  tile.setAttribute("aria-label", summary.filter(Boolean).join(", "));
}

function starterTile(player, position) {
  const tile = create(player ? "button" : "div", player ? "player-tile" : "player-tile placeholder");
  const kit = create("span", "kit");
  kit.dataset.position = position;
  kit.setAttribute("aria-hidden", "true");
  tile.append(kit);
  if (!player) {
    tile.append(create("span", "placeholder-card", position));
    return tile;
  }

  addPlayerInteraction(tile, player);
  if (player.role) {
    const badge = create("span", "role-badge", player.role);
    badge.dataset.role = player.role;
    badge.title = player.role === "C" ? "Captain" : "Vice-captain";
    tile.dataset.role = player.role;
    tile.append(badge);
  }

  const card = create("span", "player-card");
  const name = create("span", "player-name", player.name);
  name.title = player.name;
  const fixture = create("span", "player-fixtures", fixtureText(player));
  fixture.title = fixtureText(player);
  const detail = create("span", "player-detail");
  detail.append(fixture, create("span", "player-points", points(player.points)));
  card.append(name, detail);
  tile.append(card);
  return tile;
}

function benchTile(player, label) {
  const tile = create(player ? "button" : "div", "bench-tile");
  const top = create("span", "bench-top");
  top.append(create("span", "bench-label", label));
  if (player) {
    addPlayerInteraction(tile, player, label);
    top.append(create("span", "", player.position));
    const detail = create("span", "bench-detail");
    detail.append(
      create("span", "", fixtureText(player)),
      create("span", "player-points", points(player.points)),
    );
    tile.append(top, create("span", "player-name", player.name), detail);
  } else {
    tile.append(top, create("span", "player-name", "—"));
  }
  return tile;
}

function renderTiles(lineup) {
  const placeholderCounts = [1, 4, 4, 2];
  const rows = positionOrder.map((position, index) => {
    const row = create("div", "pitch-row");
    const players = lineup ? lineup.starters.filter((player) => player.position === position) :
      Array.from({ length: placeholderCounts[index] }, () => null);
    row.append(...players.map((player) => starterTile(player, position)));
    return row;
  });
  const pitch = byId("pitch-rows");
  pitch.replaceChildren(...rows);
  if (!lineup) pitch.setAttribute("aria-hidden", "true");
  else pitch.removeAttribute("aria-hidden");

  const outfield = lineup?.bench || [null, null, null];
  const bench = outfield.map((player, index) => ({ player, label: `Priority ${index + 1}` }));
  bench.push({ player: lineup?.goalkeeper, label: "Reserve GK" });
  if (lineup) {
    bench.sort((a, b) => b.player.points - a.player.points || a.player.id - b.player.id);
  }
  byId("bench-grid").replaceChildren(...bench.map(({ player, label }) => benchTile(player, label)));
}

function renderLineupWarning(lineup) {
  const status = state.status;
  const stale = status?.gameweek && (status.season !== lineup.season || status.gameweek !== lineup.gameweek);
  let warning = lineup.warning;
  if (stale) {
    warning = `This lineup is for Gameweek ${lineup.gameweek}. Run the optimizer for Gameweek ${status.gameweek}.`;
  }
  showText("lineup-warning", warning && warning !== status?.warning ? warning : null);
}

function renderLineup(lineup) {
  state.result = lineup;
  showText("formation", lineup.formation);
  let source = lineup.source.label;
  if (lineup.source.gameweek && lineup.source.type === "manager") {
    source += ` · squad from GW ${lineup.source.gameweek}`;
  }
  byId("lineup-source").textContent = source;
  byId("projected-score").textContent = points(lineup.projected_points);
  showText("lineup-breakdown", `${points(lineup.starting_points)} starter pts + ${points(lineup.captain_bonus)} captain pts`);
  renderLineupWarning(lineup);
  renderTiles(lineup);
}

async function optimize(payload, reveal = true) {
  if (state.busy) return;
  state.busy = true;
  setButtonState();
  showRequest("Projecting points and comparing legal lineups…");
  try {
    const lineup = await api("/api/optimize", payload);
    renderLineup(lineup);
    state.lastRequest = payload;
    if (payload.manager_id) {
      try { localStorage.setItem("fpl-manager-id", payload.manager_id); } catch { /* Storage can be disabled. */ }
    }
    showRequest(`Gameweek ${lineup.gameweek} lineup ready.`);
    if (reveal && window.matchMedia("(max-width: 790px)").matches) {
      byId("recommendation").scrollIntoView({ block: "start" });
    }
  } catch (error) {
    showRequest(error.message, true);
    if (error.status === 409) schedulePoll(500);
  } finally {
    state.busy = false;
    setButtonState();
  }
}

function changeMethod(isManager) {
  byId("manager-toggle").setAttribute("aria-pressed", String(isManager));
  byId("upload-toggle").setAttribute("aria-pressed", String(!isManager));
  byId("manager-form").hidden = !isManager;
  byId("upload-form").hidden = isManager;
  if (!state.busy) showRequest("");
}

function selectFile(file) {
  state.file = null;
  byId("file-label").textContent = "Choose a CSV or JSON file";
  showRequest("");
  if (!file) return;
  if (!/\.(csv|json)$/i.test(file.name) || file.size > 128 * 1024) {
    showRequest("Choose a CSV or JSON file up to 128 KB.", true);
    byId("squad-file").value = "";
    return;
  }
  state.file = file;
  byId("file-label").textContent = file.name;
}

byId("manager-toggle").addEventListener("click", () => changeMethod(true));
byId("upload-toggle").addEventListener("click", () => changeMethod(false));
byId("manager-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const managerId = byId("manager-id").value.trim();
  if (!/^\d+$/.test(managerId) || Number(managerId) < 1) {
    showRequest("Enter a valid FPL Manager ID.", true);
    return;
  }
  optimize({ manager_id: managerId });
});
byId("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = state.file;
  if (!file) {
    showRequest("Choose your squad file first.", true);
    return;
  }
  try {
    const contents = await file.text();
    await optimize({ filename: file.name, contents });
  } catch (error) {
    showRequest(`Could not read the squad file. ${error.message}`, true);
  }
});
byId("squad-file").addEventListener("change", (event) => selectFile(event.target.files[0]));

const dropZone = byId("drop-zone");
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
});
dropZone.addEventListener("dragleave", (event) => {
  if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove("dragging");
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
  byId("squad-file").value = "";
  selectFile(event.dataTransfer.files[0]);
});
byId("demo").addEventListener("click", () => optimize({ demo: true }));
byId("refresh").addEventListener("click", async () => {
  window.clearTimeout(state.pollTimer);
  state.refreshing = true;
  state.refreshRequested = true;
  setButtonState();
  showRequest("");
  try {
    const status = await api("/api/refresh", {});
    state.refreshing = false;
    applyStatus(status);
  } catch (error) {
    state.refreshRequested = false;
    showRequest(`Could not refresh FPL data. ${error.message}`, true);
    if (error.status === 409) schedulePoll(500);
  } finally {
    state.refreshing = false;
    setButtonState();
  }
});

try { byId("manager-id").value = localStorage.getItem("fpl-manager-id") || ""; } catch { /* Storage can be disabled. */ }
renderTiles(null);
fetchStatus();
