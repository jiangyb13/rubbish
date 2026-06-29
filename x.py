const state = {
  offset: 0,
  limit: 20,
  total: 0,
  view: "pairs",
};

function mediaUrl(path) {
  if (!path) return "";
  return `/media?path=${encodeURIComponent(path)}`;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[ch]));
}

function pill(ok, label) {
  return `<span class="pill ${ok ? "ok" : "bad"}">${esc(label)}=${ok ? "true" : "false"}</span>`;
}

function metricList(meta) {
  if (!meta) return "";
  const keys = ["bucket", "emotion", "shot_key", "shot_no", "frame_idx", "yaw", "pitch", "roll", "emotion_score"];
  const bodyPose = meta.body_pose || {};
  const base = keys
    .filter(key => meta[key] !== undefined && meta[key] !== null && meta[key] !== "")
    .map(key => `<span>${esc(key)}=${esc(meta[key])}</span>`);
  if (bodyPose.label) base.push(`<span>body_label=${esc(bodyPose.label)}</span>`);
  if (bodyPose.body_part) base.push(`<span>body_part=${esc(bodyPose.body_part)}</span>`);
  return base.join("");
}

function imageTile(path, label, meta = {}, whitePath = null) {
  if (!path && !whitePath) {
    return `<div class="tile missing"><div class="thumb empty">missing</div><b>${esc(label)}</b></div>`;
  }
  const thumb = (p, tag) => p
    ? `<figure class="thumbWrap">
         <img class="thumb" src="${mediaUrl(p)}" loading="lazy" alt="${esc(label)} ${tag}" onclick="openLightbox('${mediaUrl(p)}')">
         ${tag ? `<figcaption>${esc(tag)}</figcaption>` : ""}
       </figure>`
    : `<figure class="thumbWrap"><div class="thumb empty">missing</div>${tag ? `<figcaption>${esc(tag)}</figcaption>` : ""}</figure>`;
  const imgs = whitePath
    ? `<div class="thumbPair">${thumb(path, "orig")}${thumb(whitePath, "white")}</div>`
    : thumb(path, "");
  return `
    <div class="tile">
      ${imgs}
      <b>${esc(label)}</b>
      <div class="meta">${metricList(meta)}</div>
      ${path ? `<div class="path">${esc(path)}</div>` : ""}
      ${whitePath ? `<div class="path">${esc(whitePath)}</div>` : ""}
    </div>
  `;
}

function openLightbox(src) {
  const overlay = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  img.src = src;
  overlay.classList.add("active");
}

function closeLightbox() {
  const overlay = document.getElementById("lightbox");
  const img = document.getElementById("lightboxImg");
  overlay.classList.remove("active");
  img.src = "";
}

document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeLightbox();
});

function targetVideoTile(path) {
  const url = mediaUrl(path);
  return `
    <div class="videoTile">
      <b>target_video</b>
      <div class="videoPlaceholder">
        <span>video not preloaded</span>
        ${path ? `<a href="${url}" target="_blank" rel="noopener">open video</a>` : ""}
      </div>
      <div class="path">${esc(path)}</div>
    </div>
  `;
}

function gallery(title, paths, metas, whitePaths) {
  const tiles = (paths || []).map((path, index) => {
    const meta = (metas || [])[index] || {};
    const white = (whitePaths || [])[index] || null;
    const label = meta.bucket || meta.emotion || `${title}_${index + 1}`;
    return imageTile(path, label, meta, white);
  }).join("") || `<div class="empty">No refs selected</div>`;
  return `
    <section class="block">
      <div class="blockHead">
        <h3>${esc(title)}</h3>
        <span class="reason">${(paths || []).length} refs</span>
      </div>
      <div class="gallery">${tiles}</div>
    </section>
  `;
}

function rowCard(row, index) {
  const stats = row.selection_stats || {};
  const statHtml = Object.entries(stats)
    .map(([key, value]) => `<span>${esc(key)}=${esc(value)}</span>`)
    .join("");

  return `
    <article class="card">
      <header class="cardHead">
        <div>
          <h2>#${state.offset + index + 1} ${esc(row.person_id)}</h2>
          <p>${esc(row.source_shot_key)} | uid=${esc(row.source_uid)}</p>
        </div>
      </header>
      <section class="target">
        ${imageTile(row.first_frame, "first_frame")}
        ${targetVideoTile(row.target_video)}
      </section>
      ${gallery("angle_ref", row.angle_ref, row.angle_ref_meta, row.angle_ref_white)}
      ${gallery("emo_ref", row.emo_ref, row.emo_ref_meta, row.emo_ref_white)}
      ${gallery("body_pose_ref", row.body_pose_ref, row.body_pose_ref_meta, row.body_pose_ref_white)}
      <div class="stats">${statHtml}</div>
    </article>
  `;
}

function formatDeg(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(1)} deg` : "-";
}

function poolImagePath(row) {
  const imageType = document.getElementById("poolImageTypeSelect").value || "face_orig";
  const related = row.related_images || {};
  return related[imageType] || row.pool_image_path || row.image_path || related.face_orig;
}

function poolTile(row) {
  const expression = row.expression || {};
  const pose = row.pose || {};
  const path = poolImagePath(row);
  const src = mediaUrl(path);
  const emo = expression.dominant ? `<div class="emotionBadge">${esc(expression.dominant)}</div>` : "";
  const flag = typeof expression.emo_flag === "boolean"
    ? `<div class="emoFlag ${expression.emo_flag ? "true" : "false"}">VLM ${expression.emo_flag ? "true" : "false"}</div>`
    : "";
  return `
    <div class="poolPhoto" title="${esc(path)}">
      <div class="poolImgWrap">
        <img src="${src}" loading="lazy" onclick="openLightbox('${src}')">
        ${flag}
        ${emo}
      </div>
      <div class="poolPose">
        <span>P ${esc(formatDeg(pose.pitch))}</span>
        <span>Y ${esc(formatDeg(pose.yaw))}</span>
        <span>R ${esc(formatDeg(pose.roll))}</span>
      </div>
      <div class="poolCaption">${esc(row.shot_key)} / f${esc(row.frame_idx)}</div>
    </div>
  `;
}

async function loadSummary() {
  if (state.view === "pool") {
    return loadPoolSummary();
  }
  const res = await fetch("/api/summary");
  const summary = await res.json();
  document.getElementById("summary").innerHTML = `
    <div><b>Total</b><span>${summary.total}</span></div>
    <div><b>Persons</b><span>${summary.person_count}</span></div>
    <div><b>Angle refs</b><span>${summary.angle_refs}</span></div>
    <div><b>Emo refs</b><span>${summary.emo_refs}</span></div>
    <div><b>Body refs</b><span>${summary.body_pose_refs}</span></div>
  `;

  const select = document.getElementById("personSelect");
  select.innerHTML = `<option value="">All persons</option>` + summary.persons
    .map(person => `<option value="${esc(person)}">${esc(person)}</option>`)
    .join("");
}

async function loadPoolSummary() {
  const res = await fetch("/api/pool/summary");
  const summary = await res.json();
  document.getElementById("summary").innerHTML = `
    <div><b>Pool persons</b><span>${summary.person_count}</span></div>
    <div><b>Pool frames</b><span>${summary.total_frames}</span></div>
  `;

  const select = document.getElementById("personSelect");
  select.innerHTML = `<option value="">Select person</option>` + summary.persons
    .map(person => `<option value="${esc(person.person_id)}">${esc(person.person_id)} (${person.count})</option>`)
    .join("");
  if (!select.value && summary.persons.length > 0) {
    select.value = summary.persons[0].person_id;
  }
}

async function loadRows() {
  if (state.view === "pool") {
    return loadPoolRows();
  }
  const person = document.getElementById("personSelect").value;
  const params = new URLSearchParams({
    offset: state.offset,
    limit: state.limit,
    person_id: person,
    flag: "all",
  });
  const res = await fetch(`/api/pairs?${params.toString()}`);
  const data = await res.json();
  state.total = data.total;
  document.getElementById("content").innerHTML = data.rows
    .map((row, index) => rowCard(row, index))
    .join("") || `<div class="emptyPage">No rows match the current filter.</div>`;
  document.getElementById("prevBtn").disabled = state.offset <= 0;
  document.getElementById("nextBtn").disabled = state.offset + state.limit >= state.total;
}

async function loadPoolRows() {
  const person = document.getElementById("personSelect").value;
  const params = new URLSearchParams({
    offset: state.offset,
    limit: 60,
    person_id: person,
  });
  const res = await fetch(`/api/pool?${params.toString()}`);
  const data = await res.json();
  state.total = data.total;
  document.getElementById("content").innerHTML = data.rows.length
    ? `<section class="poolPanel">
        <div class="poolPanelHead">
          <h2>${esc(person)}</h2>
          <span>${state.offset + 1}-${Math.min(state.offset + 60, state.total)} / ${state.total}</span>
        </div>
        <div class="poolGrid">${data.rows.map(row => poolTile(row)).join("")}</div>
      </section>`
    : `<div class="emptyPage">No filtered pool rows. Run TASK_NAME=pool_filted first.</div>`;
  document.getElementById("prevBtn").disabled = state.offset <= 0;
  document.getElementById("nextBtn").disabled = state.offset + 60 >= state.total;
}

function resetAndLoad() {
  state.offset = 0;
  loadRows();
}

document.getElementById("viewSelect").addEventListener("change", async event => {
  state.view = event.target.value;
  state.offset = 0;
  document.getElementById("flagSelect").style.display = "none";
  document.getElementById("poolImageTypeSelect").style.display = state.view === "pool" ? "" : "none";
  await loadSummary();
  await loadRows();
});
document.getElementById("personSelect").addEventListener("change", resetAndLoad);
document.getElementById("flagSelect").addEventListener("change", resetAndLoad);
document.getElementById("poolImageTypeSelect").addEventListener("change", resetAndLoad);
document.getElementById("prevBtn").addEventListener("click", () => {
  const step = state.view === "pool" ? 60 : state.limit;
  state.offset = Math.max(0, state.offset - step);
  loadRows();
});
document.getElementById("nextBtn").addEventListener("click", () => {
  const step = state.view === "pool" ? 60 : state.limit;
  if (state.offset + step < state.total) {
    state.offset += step;
    loadRows();
  }
});

loadSummary().then(loadRows);
