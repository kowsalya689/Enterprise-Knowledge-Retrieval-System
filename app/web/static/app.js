// Shared helpers for both pages. All text coming from the server or the model is inserted as plain text
// (textContent), so a document or an answer can not inject HTML into the page.
// The pages never ask for or store an API key. They send "X-Web-UI: 1"; the server decides the role (WEB_UI_ROLE in .env).

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

async function api(path, options = {}) {
  const headers = Object.assign({ "X-Web-UI": "1" }, options.headers || {});
  let res;
  try { res = await fetch(path, Object.assign({}, options, { headers })); }
  catch (e) { throw new Error("Cannot reach the server. Is the API running?"); }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) {
    const e = body && body.error ? body.error : { message: res.statusText };
    const detail = e.details ? " (" + e.details.map(d => d.field + ": " + d.problem).join("; ") + ")" : "";
    const err = new Error(e.message + detail); err.status = res.status; err.code = e.code; throw err;
  }
  return body;
}

async function loadDocuments(target, countTarget) {
  target.replaceChildren();
  try {
    const data = await api("/documents");
    if (countTarget) countTarget.textContent = data.count + " document" + (data.count === 1 ? "" : "s") + " visible";
    if (!data.count) { target.append(el("p", { class: "empty", text: "No documents yet. Upload one on the Upload page." })); return data; }
    const head = el("tr", {}, ...["Document", "Department", "Access tags", "Version", "Chunks"].map(h => el("th", { text: h })));
    const rows = data.documents.map(d => el("tr", {},
      el("td", { text: d.file_name }), el("td", { text: d.department || "" }),
      el("td", { text: (d.access_tags || []).join(", ") }), el("td", { text: d.version || "" }),
      el("td", { text: String(d.chunks) })));
    target.append(el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows)));
    return data;
  } catch (e) {
    if (countTarget) countTarget.textContent = "";
    target.append(el("p", { class: "empty err", text: e.status === 401 ? "The web pages are switched off (WEB_UI_ROLE is empty in .env). Use the API with a key." : e.message }));
    return null;
  }
}
