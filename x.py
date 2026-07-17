const state = {
  offset: 0,
  limit: 20,
  total: 0,
  totalKnown: false,
  hasNext: false,
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

function maxSimilarityThreshold() {
  const input = document.getElementById("maxSimilarityInput");
  if (!input) return "";
  const value = input.value.trim();
  if (value === "") return "";
  const number = Number(value);
  return Number.isFinite(number) ? String(number) : "";
}

function statText(value) {
  return value === null || value === undefined ? "fast" : value;
}

function formatStatValue(value) {
  if (value && typeof value === "object") {
    if (value.similarity !== undefined) {
      const left = value.left || {};
      const right = value.right || {};
      const leftKey = [left.bucket, left.shot_key, left.frame_idx].filter(x => x !== undefined && x !== null && x !== "").join("/");
      const rightKey = [right.bucket, right.shot_key, right.frame_idx].filter(x => x !== undefined && x !== null && x !== "").join("/");
      return `sim=${value.similarity} (${leftKey} <-> ${rightKey})`;
    }
    return JSON.stringify(value);
  }
  return value;
}

function displayBucket(value) {
  return String(value || "").replace("__", "_");
}

function metricList(meta) {
  if (!meta) return "";
  const keys = ["bucket", "bucket_source", "body_label", "body_part_bucket", "emotion", "shot_key", "shot_no", "frame_idx", "yaw", "pitch", "roll", "emotion_score"];
  const bodyPose = meta.body_pose || {};
  const base = keys
    .filter(key => meta[key] !== undefined && meta[key] !== null && meta[key] !== "")
    .map(key => `<span>${esc(key)}=${esc(key === "bucket" ? displayBucket(meta[key]) : meta[key])}</span>`);
  if (bodyPose.label) base.push(`<span>body_label=${esc(bodyPose.label)}</span>`);
  if (bodyPose.body_part) base.push(`<span>body_part=${esc(bodyPose.body_part)}</span>`);
  return base.join("");
}

function qualityStateClass(value) {
  if (value === false) return "bad";
  if (value === true) return "ok";
  return "neutral";
}

function qualitySummary(meta) {
  if (!meta) return "";
  const labels = [];
  if (typeof meta.face_quality_label === "boolean") {
    labels.push(`<span class="qualityPill ${qualityStateClass(meta.face_quality_label)}">face_quality=${meta.face_quality_label}</span>`);
  }
  if (typeof meta.full_quality_label === "boolean") {
    labels.push(`<span class="qualityPill ${qualityStateClass(meta.full_quality_label)}">full_quality=${meta.full_quality_label}</span>`);
  }
  return labels.length ? `<div class="qualityBar">${labels.join("")}</div>` : "";
}

function qualityDetailPills(quality) {
  if (!quality || typeof quality !== "object") return "";
  const labels = [];
  for (const key of ["mask_hole", "face_bbox_boundary", "face_mask_coverage"]) {
    const item = quality[key];
    if (!item || typeof item !== "object") continue;
    const passed = item.passed;
    const status = item.status || "";
    const bits = [key];
    if (status) bits.push(status);
    if (item.hole_count !== undefined) bits.push(`holes=${item.hole_count}`);
    if (item.threshold !== undefined) bits.push(`thr=${item.threshold}`);
    if (item.touches_boundary !== undefined) bits.push(`touch=${item.touches_boundary}`);
    if (item.mask_foreground_ratio !== undefined && item.mask_foreground_ratio !== null) bits.push(`fg=${Number(item.mask_foreground_ratio).toFixed(3)}`);
    if (item.mask_background_pixel_count !== undefined && item.mask_background_pixel_count !== null) bits.push(`bg_px=${item.mask_background_pixel_count}`);
    if (item.yaw !== undefined && item.yaw !== null) bits.push(`yaw=${Number(item.yaw).toFixed(1)}`);
    if (item.is_frontal !== undefined && item.is_frontal !== null) bits.push(`frontal=${item.is_frontal}`);
    labels.push(`<span class="qualityPill ${qualityStateClass(passed)}">${esc(bits.join(" | "))}</span>`);
  }
  return labels.length ? `<div class="qualityBar detail">${labels.join("")}</div>` : "";
}

function displayedQualityTypes(meta) {
  const text = [meta.path, meta.white_path, meta.bucket_source, meta.body_label, meta.body_part_bucket, meta.bucket]
    .map(value => String(value || "").toLowerCase())
    .join(" ");
  if (text.includes("full_") || text.includes("/full") || text.includes("body")) {
    return ["full_orig", "full_white"];
  }
  return ["face_orig", "face_white"];
}

function qualityForDisplayedImages(meta) {
  if (!meta) return "";
  const detailByType = meta.quality_detail_by_image_type || {};
  const types = displayedQualityTypes(meta);
  const blocks = [];
  for (const type of types) {
    const detail = detailByType[type] || {};
    const hasDetail = detail && (typeof detail.quality_label === "boolean" || detail.quality);
    if (!hasDetail) continue;
    blocks.push(`
      <div class="qualityImageBlock">
        <div class="qualityImageTitle">
          <span>${esc(type)}</span>
          ${typeof detail.quality_label === "boolean" ? `<span class="qualityPill ${qualityStateClass(detail.quality_label)}">quality=${detail.quality_label}</span>` : ""}
        </div>
        ${qualityDetailPills(detail.quality)}
      </div>
    `);
  }
  if (blocks.length) return `<div class="qualityImages">${blocks.join("")}</div>`;
  return `${qualitySummary(meta)}${qualityDetailPills(meta.quality)}`;
}

function pathRows(path, row) {
  if (!path) return "";
  const info = (row && row._path_info && row._path_info[path]) || null;
  const rows = [`<div class="path"><b>raw</b>${esc(path)}</div>`];
  if (info && info.abs && info.abs !== path) {
    rows.push(`<div class="path"><b>abs</b>${esc(info.abs)}</div>`);
  }
  if (info && info.rel && info.rel !== path) {
    rows.push(`<div class="path"><b>rel</b>${esc(info.rel)}</div>`);
  }
  if (info) {
    rows.push(`<div class="path ${info.exists ? "exists" : "missingPath"}"><b>exists</b>${esc(info.exists)}</div>`);
  }
  return rows.join("");
}

function imageTile(path, label, meta = {}, whitePath = null, row = null) {
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
    ? `<div class="thumbPair">${thumb(path, "original")}${thumb(whitePath, "white")}</div>`
    : thumb(path, "original");
  return `
    <div class="tile">
      ${imgs}
      <b>${esc(label)}</b>
      <div class="meta">${metricList(meta)}</div>
      ${qualityForDisplayedImages(meta)}
      ${pathRows(path, row)}
      ${whitePath ? pathRows(whitePath, row) : ""}
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

function targetVideoTile(path, row = null) {
  const url = mediaUrl(path);
  return `
    <div class="videoTile">
      <b>target_video</b>
      <div class="videoPlaceholder">
        <span>video not preloaded</span>
        ${path ? `<a href="${url}" target="_blank" rel="noopener">open video</a>` : ""}
      </div>
      ${pathRows(path, row)}
    </div>
  `;
}

function gallery(title, paths, metas, whitePaths, row = null) {
  const tiles = (paths || []).map((path, index) => {
    const meta = (metas || [])[index] || {};
    const white = (whitePaths || [])[index] || null;
    const label = meta.bucket ? displayBucket(meta.bucket) : (meta.emotion || `${title}_${index + 1}`);
    return imageTile(path, label, meta, white, row);
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
    .map(([key, value]) => `<span>${esc(key)}=${esc(formatStatValue(value))}</span>`)
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
        ${imageTile(row.first_frame, "first_frame", {}, null, row)}
        ${targetVideoTile(row.target_video, row)}
      </section>
      ${gallery("angle_ref", row.angle_ref, row.angle_ref_meta, row.angle_ref_white, row)}
      ${gallery("emo_ref", row.emo_ref, row.emo_ref_meta, row.emo_ref_white, row)}
      ${gallery("body_pose_ref", row.body_pose_ref, row.body_pose_ref_meta, row.body_pose_ref_white, row)}
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
      ${qualityDetailPills(row.quality)}
      ${typeof row.quality_label === "boolean" ? `<div class="qualityBar"><span class="qualityPill ${qualityStateClass(row.quality_label)}">quality=${row.quality_label}</span></div>` : ""}
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
  if (summary.pairs_jsonl) {
    document.getElementById("pairsPath").textContent = summary.pairs_jsonl;
  }
  document.getElementById("summary").innerHTML = `
    <div><b>File</b><span>${esc(summary.pairs_jsonl || "")}</span></div>
    <div><b>Total</b><span>${esc(statText(summary.total))}</span></div>
    <div><b>Persons</b><span>${esc(statText(summary.person_count))}</span></div>
    <div><b>Angle refs</b><span>${esc(statText(summary.angle_refs))}</span></div>
    <div><b>Emo refs</b><span>${esc(statText(summary.emo_refs))}</span></div>
    <div><b>Body refs</b><span>${esc(statText(summary.body_pose_refs))}</span></div>
  `;

  const select = document.getElementById("personSelect");
  select.innerHTML = `<option value="">All persons</option>` + (summary.persons || [])
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
  const threshold = maxSimilarityThreshold();
  if (threshold !== "") {
    params.set("max_similarity_threshold", threshold);
  }
  const res = await fetch(`/api/pairs?${params.toString()}`);
  const data = await res.json();
  state.total = data.total;
  state.totalKnown = Boolean(data.total_known);
  state.hasNext = Boolean(data.has_next);
  const end = data.offset + data.rows.length;
  const rangeText = data.rows.length
    ? `${data.offset + 1}-${end} / ${state.totalKnown ? data.total : "?"}`
    : `${data.offset} / ${state.totalKnown ? data.total : "?"}`;
  document.getElementById("content").innerHTML = data.rows
    .map((row, index) => rowCard(row, index))
    .join("") || `<div class="emptyPage">No rows match the current filter.</div>`;
  document.getElementById("pageInfo").textContent = threshold !== ""
    ? `${rangeText} | max similarity < ${threshold}`
    : rangeText;
  document.getElementById("prevBtn").disabled = state.offset <= 0;
  document.getElementById("nextBtn").disabled = !state.hasNext;
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
  document.getElementById("pageInfo").textContent = state.total > 0
    ? `${state.offset + 1}-${Math.min(state.offset + 60, state.total)} / ${state.total}`
    : `0 / 0`;
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
  document.getElementById("maxSimilarityControl").style.display = state.view === "pairs" ? "" : "none";
  await loadSummary();
  await loadRows();
});
document.getElementById("personSelect").addEventListener("change", resetAndLoad);
document.getElementById("flagSelect").addEventListener("change", resetAndLoad);
document.getElementById("poolImageTypeSelect").addEventListener("change", resetAndLoad);
document.getElementById("maxSimilarityInput").addEventListener("change", resetAndLoad);
document.getElementById("maxSimilarityInput").addEventListener("keydown", event => {
  if (event.key === "Enter") resetAndLoad();
});
document.getElementById("prevBtn").addEventListener("click", () => {
  const step = state.view === "pool" ? 60 : state.limit;
  state.offset = Math.max(0, state.offset - step);
  loadRows();
});
document.getElementById("nextBtn").addEventListener("click", () => {
  const step = state.view === "pool" ? 60 : state.limit;
  if (state.view === "pool") {
    if (state.offset + step < state.total) {
      state.offset += step;
      loadRows();
    }
    return;
  }
  if (state.hasNext) {
    state.offset += step;
    loadRows();
  }
});

loadSummary().then(() => {
  loadRows();
});
