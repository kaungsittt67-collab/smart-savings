// Smart Savings for GitHub Pages: the same 2 models as the Streamlit app, run in the browser.
// The models were trained in Colab and exported to JSON by export_for_web.py.

let savings, saver, customers;
const COLORS = ["#1F5F8B", "#E3A33B", "#C0392B", "#2E8B57", "#7D5BA6"];
const $ = (id) => document.getElementById(id);

// ---------- STEP 1: tabs ----------
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
  tab.classList.add("active");
  $(tab.dataset.tab).classList.add("active");
}));
document.querySelectorAll("input[type=range]").forEach((input) =>
  input.addEventListener("input", () => { $(input.id + "-out").textContent = input.value; }));

// ---------- STEP 2: load the exported models ----------
async function loadModels() {
  [savings, saver, customers] = await Promise.all(
    ["savings_model.json", "saver_model.json", "customers.json"].map((f) => fetch(f).then((r) => {
      if (!r.ok) throw new Error(f + " not found");
      return r.json();
    })));
}

// ---------- STEP 3: supervised model -> chance of opening a fixed deposit ----------
// Same steps as the scikit-learn pipeline: scale the numbers, one-hot the categories, then the forest.
function prepare(customer) {
  const x = savings.num.map((col, i) => (customer[col] - savings.mean[i]) / savings.scale[i]);
  savings.cat.forEach((col, i) => savings.categories[i].forEach((c) => x.push(customer[col] === c ? 1 : 0)));
  return x.map(Math.fround);  // scikit-learn trees compare 32-bit numbers
}
function predictChance(customer) {
  const x = prepare(customer);
  let total = 0;
  for (const tree of savings.trees) {
    let node = 0;
    while (tree.l[node] !== -1) node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    total += tree.p[node];
  }
  return total / savings.trees.length;
}

// ---------- STEP 4: unsupervised model -> saver group ----------
function predictGroup(person) {
  const x = saver.features.map((col, i) => (person[col] - saver.mean[i]) / saver.scale[i]);
  let best = 0, bestDist = Infinity;
  saver.centers.forEach((center, g) => {
    const d = center.reduce((sum, c, i) => sum + (x[i] - c) ** 2, 0);
    if (d < bestDist) { bestDist = d; best = g; }
  });
  return best;
}

// ---------- STEP 5: connect the page ----------
function fillSelect(id, values, chosen) {
  $(id).innerHTML = values.map((v) => `<option${v === chosen ? " selected" : ""}>${v}</option>`).join("");
}
function table(id, headers, rows, numeric) {
  const head = "<tr>" + headers.map((h, i) => `<th class="${numeric[i] ? "num" : ""}">${h}</th>`).join("") + "</tr>";
  $(id).innerHTML = head + rows.map((r) => `<tr class="${r.cls || ""}">` +
    r.cells.map((c, i) => `<td class="${numeric[i] ? "num" : ""}">${c}</td>`).join("") + "</tr>").join("");
}

function showDeposit() {
  const customer = { age: +$("age").value, campaign: +$("campaign").value, previous: +$("previous").value,
    euribor3m: +$("euribor3m").value };
  savings.cat.forEach((col) => { customer[col] = $(col).value; });
  const chance = predictChance(customer);
  $("chance").textContent = Math.round(chance * 100) + "%";
  $("chance-bar").style.width = (chance * 100).toFixed(1) + "%";
  const advice = $("advice");
  if (chance >= 0.7) { advice.className = "advice success"; advice.textContent = "High: call this customer first."; }
  else if (chance >= 0.5) { advice.className = "advice warning"; advice.textContent = "Medium: worth a call."; }
  else { advice.className = "advice info"; advice.textContent = "Low: send an SMS or leaflet instead of calling."; }
  $("deposit-result").hidden = false;
}

function showGroup() {
  const income = +$("income").value, spending = +$("spending").value;
  const group = predictGroup({ "Annual Income (k$)": income, "Spending Score (1-100)": spending });
  $("group-name").innerHTML = `This person is in group ${group}: <b>${saver.names[group]}</b>`;
  $("group-tip").textContent = "Savings tip: " + saver.tips[group];
  const rows = Object.entries(saver.profile).map(([g, v]) => ({ cells: [saver.names[g], v[0].toFixed(1), v[1].toFixed(1)] }));
  rows.push({ cls: "me", cells: ["➡️ This person", income.toFixed(1), spending.toFixed(1)] });
  table("profile", ["", "Annual Income (k$)", "Spending Score (1-100)"], rows, [false, true, true]);
  drawChart(income, spending);
  $("group-result").hidden = false;
}

function drawChart(income, spending) {
  const W = 640, H = 400, L = 50, B = 40, T = 15, R = 15;
  const sx = (v) => L + (v - 10) / (145 - 10) * (W - L - R);
  const sy = (v) => H - B - v / 100 * (H - B - T);
  let svg = "";
  for (let v = 20; v <= 140; v += 20) svg += `<line x1="${sx(v)}" x2="${sx(v)}" y1="${T}" y2="${H - B}" stroke="#EEF1F5"/><text x="${sx(v)}" y="${H - B + 18}" font-size="12" text-anchor="middle" fill="#646E7A">${v}</text>`;
  for (let v = 0; v <= 100; v += 20) svg += `<line x1="${L}" x2="${W - R}" y1="${sy(v)}" y2="${sy(v)}" stroke="#EEF1F5"/><text x="${L - 8}" y="${sy(v) + 4}" font-size="12" text-anchor="end" fill="#646E7A">${v}</text>`;
  svg += `<text x="${(L + W - R) / 2}" y="${H - 6}" font-size="12" text-anchor="middle" fill="#3D4654">Annual Income (k$)</text>`;
  svg += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" font-size="12" text-anchor="middle" fill="#3D4654">Spending Score (1-100)</text>`;
  for (const c of customers) svg += `<circle cx="${sx(c["Annual Income (k$)"])}" cy="${sy(c["Spending Score (1-100)"])}" r="4.5" fill="${COLORS[c.group]}" fill-opacity="0.8"/>`;
  svg += `<path d="M${sx(income) - 9} ${sy(spending)}h18M${sx(income)} ${sy(spending) - 9}v18" stroke="#1B2F4A" stroke-width="3"/>`;
  $("chart").innerHTML = svg;
  $("legend").innerHTML = Object.keys(saver.names).map((g) => `<span style="--c:${COLORS[g]}">${saver.names[g]}</span>`).join("") +
    "<span>✚ This person</span>";
}

function showAbout() {
  $("model-name").textContent = savings.model_name;
  table("scores", ["model", "accuracy", "precision", "recall", "F1"],
    savings.scores.map((s) => ({ cells: [s.model, s.accuracy.toFixed(3), s.precision.toFixed(3), s.recall.toFixed(3), s.F1.toFixed(3)] })),
    [false, true, true, true, true]);
  const counts = {};
  customers.forEach((c) => { counts[c.group_name] = (counts[c.group_name] || 0) + 1; });
  table("sizes", ["group_name", "customers"],
    Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([n, c]) => ({ cells: [n, c] })), [false, true]);
}

loadModels().then(() => {
  const c = savings.choices;
  fillSelect("job", c.job, "admin."); fillSelect("marital", c.marital, "married");
  fillSelect("education", c.education, "university.degree"); fillSelect("default", c.default, "no");
  fillSelect("housing", c.housing, "yes"); fillSelect("loan", c.loan, "no");
  fillSelect("poutcome", c.poutcome, "nonexistent");
  showAbout();
  $("predict").disabled = $("find").disabled = false;
  $("predict").addEventListener("click", showDeposit);
  $("find").addEventListener("click", showGroup);
  $("loading").hidden = true;
}).catch((err) => {
  $("loading").textContent = "Could not load the models: " + err.message +
    ". Put the 3 .json files in the same folder as index.html, and open the page through GitHub Pages (not by double-clicking).";
});
