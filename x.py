<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Pairs Viewer</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Training Pairs Viewer</h1>
      <p id="pairsPath">{{ pairs_jsonl }}</p>
    </div>
    <div class="toolbar">
      <select id="viewSelect">
        <option value="pairs">Training pairs</option>
        <option value="pool">Filted pool</option>
      </select>
      <select id="personSelect"></select>
      <select id="flagSelect" style="display:none">
        <option value="all">All flags</option>
        <option value="both_true">Both true</option>
        <option value="any_false">Any false</option>
        <option value="angle_true">Angle true</option>
        <option value="angle_false">Angle false</option>
        <option value="emo_true">Emo true</option>
        <option value="emo_false">Emo false</option>
      </select>
      <select id="poolImageTypeSelect" style="display:none">
        <option value="face_orig">Face Orig</option>
        <option value="face_white">Face White</option>
        <option value="full_orig">Full Orig</option>
        <option value="full_white">Full White</option>
      </select>
      <label id="maxSimilarityControl" class="thresholdControl">
        <span>Max similarity &lt;</span>
        <input id="maxSimilarityInput" type="number" min="-1" max="1" step="0.01" placeholder="off">
      </label>
      <button id="prevBtn">Prev</button>
      <button id="nextBtn">Next</button>
      <span id="pageInfo" class="pageInfo"></span>
    </div>
  </header>

  <section id="summary" class="summary"></section>
  <main id="content" class="content"></main>

  <div id="lightbox" class="lightbox" onclick="closeLightbox()">
    <img id="lightboxImg" src="" alt="">
  </div>

  <script src="{{ url_for('static', filename='app.js') }}"></script>
</body>
</html>
