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
  transferRequest: null,
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
const gainFormat = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
  signDisplay: "exceptZero",
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
  const currentSquad = state.result?.season === state.status?.season && state.result?.gameweek === state.status?.gameweek;
  byId("transfer-submit").disabled = inactive || !currentSquad || Boolean(state.transferRequest);
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
  if (state.result) {
    renderLineupWarning(state.result);
    if (status.gameweek && (status.season !== state.result.season || status.gameweek !== state.result.gameweek)) {
      clearTransferResults();
    }
  }
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
  if (player.fixtures.length && Number.isFinite(player.expected_minutes)) {
    const minutes = create("div", "fixture-row");
    minutes.append(create("span", "", "Expected minutes"), create("strong", "", Math.round(player.expected_minutes)));
    fixtures.append(minutes);
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
  const previous = state.result;
  state.result = lineup;
  showText("formation", lineup.formation);
  let source = lineup.source.label;
  if (lineup.source.gameweek && lineup.source.type === "manager") {
    source += ` · squad from GW ${lineup.source.gameweek}`;
  }
  byId("lineup-source").textContent = source;
  byId("projected-score").textContent = points(lineup.evaluation?.points ?? lineup.projected_points);
  const parts = [`${points(lineup.projected_points)} XI + captain`];
  if (lineup.evaluation) {
    parts.push(`${gainFormat.format(lineup.evaluation.autosub_points)} autosubs`,
      `${gainFormat.format(lineup.evaluation.vice_captain_points)} vice-captain`);
  }
  showText("lineup-breakdown", parts.join(" · "));
  renderLineupWarning(lineup);
  renderTiles(lineup);
  renderTransferForm(lineup, previous);
}

function transferSquad(lineup) {
  return [...lineup.starters, ...lineup.bench, lineup.goalkeeper];
}

function squadKey(lineup) {
  const ids = transferSquad(lineup).map((player) => player.id).sort((left, right) => left - right);
  return JSON.stringify([lineup.season, lineup.source.type, lineup.source.label, ids]);
}

function saleInputs() {
  return byId("transfer-selling-fields").querySelectorAll("input");
}

function sellingPrices() {
  return Object.fromEntries([...saleInputs()].filter((input) => input.value !== "").map((input) => [input.name, input.value]));
}

function clearTransferResults() {
  state.transferRequest = null;
  byId("transfer-results").hidden = true;
  showText("transfer-message", null);
  byId("transfer-announcement").textContent = "";
  setButtonState();
}

function renderTransferForm(lineup, previous) {
  clearTransferResults();
  const rounds = lineup.transfer_gameweeks || [];
  byId("transfer-panel").hidden = rounds.length === 0;
  byId("transfer-jump").hidden = rounds.length === 0;
  if (!rounds.length) return;

  const sameSquad = previous && squadKey(lineup) === squadKey(previous);
  const priorPrices = sameSquad ? sellingPrices() : {};
  const priorOutgoing = sameSquad ? byId("transfer-outgoing").value : "";
  const priorHorizon = sameSquad ? Number(byId("transfer-horizon").value) : rounds.length;
  if (!sameSquad) {
    byId("transfer-form").reset();
    byId("transfer-selling").open = false;
  }

  const horizon = byId("transfer-horizon");
  horizon.replaceChildren(...rounds.map((week, index) => new Option(
    index === 0 ? `GW ${week}` : `GW ${rounds[0]}–${week}`, String(index + 1),
  )));
  horizon.value = String(Math.min(priorHorizon || rounds.length, rounds.length));

  const players = transferSquad(lineup).sort((left, right) => (
    positionOrder.indexOf(left.position) - positionOrder.indexOf(right.position) || left.name.localeCompare(right.name)
  ));
  const outgoing = byId("transfer-outgoing");
  outgoing.replaceChildren(new Option("Any player", ""), ...players.map((player) => new Option(
    `${player.name} (${player.team})`, String(player.id),
  )));
  outgoing.value = priorOutgoing;

  byId("transfer-selling-fields").replaceChildren(...players.map((player) => {
    const field = create("div");
    const input = create("input");
    input.id = `sale-price-${player.id}`;
    input.name = String(player.id);
    input.type = "number";
    input.inputMode = "decimal";
    input.min = "0.1";
    input.step = "0.1";
    input.value = priorPrices[player.id] || "";
    if (Number.isInteger(player.price)) {
      input.max = (player.price / 10).toFixed(1);
      input.placeholder = `${input.max} est.`;
    }
    const label = create("label", "", `${player.name} (£m)`);
    label.htmlFor = input.id;
    field.append(label, input);
    return field;
  }));
}

function money(tenths) {
  return `£${(tenths / 10).toFixed(1)}m`;
}

function transferPlayer(label, player, price, estimated = false) {
  const block = create("div", "transfer-player");
  const names = create("div", "transfer-player-name");
  names.append(create("strong", "", player.name), create("span", "", player.team));
  const flag = { d: "Doubtful", i: "Injured", s: "Suspended" }[player.status];
  if (flag) names.append(create("span", "transfer-player-flag", flag));
  block.append(
    create("div", "transfer-label", label), names,
    create("div", "transfer-player-price", `${money(price)}${estimated ? " estimated" : ""}`),
  );
  return block;
}

function transferCandidate(candidate) {
  const card = create("li", "transfer-card");
  const headline = create("div", "transfer-card-headline");
  const gain = create("div", "transfer-gain");
  gain.append(create("strong", "", `${gainFormat.format(candidate.net_gain)} pts`), create("span", "", "Net gain"));
  headline.append(
    transferPlayer("Buy", candidate.incoming, candidate.incoming.price),
    transferPlayer("Sell", candidate.outgoing, candidate.sale_price, candidate.estimated_sale_price),
    gain,
  );

  const details = create("div", "transfer-card-details");
  const rounds = create("ul", "transfer-weeks");
  for (const week of candidate.weeks) {
    const role = week.captain ? " · Captain" : week.starts ? " · Starts" : "";
    const fixtures = candidate.incoming.fixtures.filter((fixture) => fixture.gameweek === week.gameweek);
    const opponents = fixtures.length ? fixtures.map(fixtureLabel).join(", ") : "Blank";
    rounds.append(create("li", "", `GW ${week.gameweek} · ${opponents} · ${gainFormat.format(week.gain)} pts${role}`));
  }
  details.append(rounds, create("span", "transfer-bank-after", `${money(candidate.bank_after)} left in bank`));
  card.append(headline, details);
  return card;
}

function renderTransferResults(result) {
  const first = result.gameweeks[0];
  const last = result.gameweeks.at(-1);
  const range = first === last ? `GW ${first}` : `GW ${first}–${last}`;
  const hit = result.hit ? ` · ${result.hit} pt hit included per move` : "";
  const hold = "Keep your current squad. No affordable move improves the projected total after transfer costs.";
  byId("transfer-context").textContent = `${range} · Current squad ${points(result.baseline_points)} pts${hit}`;
  showText("transfer-hold", result.candidates.length ? null : hold);
  byId("transfer-list").replaceChildren(...result.candidates.map(transferCandidate));
  byId("transfer-results").hidden = false;
  showText("transfer-message", null);
  byId("transfer-announcement").textContent = result.candidates.length ? "Ranked transfer targets ready." : hold;
}

async function findTransfers(event) {
  event.preventDefault();
  if (byId("transfer-submit").disabled) return;
  const lineup = state.result;
  const outgoing = byId("transfer-outgoing").value;
  const payload = {
    season: lineup.season,
    gameweek: lineup.gameweek,
    squad_ids: transferSquad(lineup).map((player) => player.id),
    bank: byId("transfer-bank").value,
    free_transfers: Number(byId("transfer-free").value),
    horizon: Number(byId("transfer-horizon").value),
    selling_prices: sellingPrices(),
  };
  if (outgoing) payload.outgoing_id = Number(outgoing);

  const request = {};
  state.transferRequest = request;
  byId("transfer-results").hidden = true;
  byId("transfer-message").classList.remove("is-error");
  showText("transfer-message", "Comparing affordable moves…");
  setButtonState();
  try {
    const result = await api("/api/transfers", payload);
    if (state.transferRequest === request) renderTransferResults(result);
  } catch (error) {
    if (state.transferRequest !== request) return;
    byId("transfer-message").classList.add("is-error");
    showText("transfer-message", error.message);
    if (error.status === 409) schedulePoll(500);
  } finally {
    if (state.transferRequest === request) {
      state.transferRequest = null;
      setButtonState();
    }
  }
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
byId("transfer-form").addEventListener("submit", findTransfers);
byId("transfer-form").addEventListener("input", clearTransferResults);
byId("transfer-selling-fields").addEventListener("invalid", () => { byId("transfer-selling").open = true; }, true);
byId("refresh").addEventListener("click", async () => {
  window.clearTimeout(state.pollTimer);
  state.refreshing = true;
  state.refreshRequested = true;
  clearTransferResults();
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
