'use strict';

const state = { frame: null, images: {}, running: false };

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const CAPTIONS = {
  rgb: 'Raw left color camera image (KITTI image_2).',
  lidar_projection: 'Velodyne point cloud projected onto the image, colored by depth (calibration check).',
  detections_2d: 'YOLOv8 2D detections, remapped from COCO to KITTI classes.',
  detections_3d: 'Fused oriented 3D boxes from LiDAR points inside each detection frustum.',
};

const STEP_ORDER = ['rgb', 'lidar_projection', 'detections_2d', 'fusion', 'detections_3d'];

/* ---------- frame loading ---------- */
async function loadFrames() {
  const res = await fetch('/api/frames');
  const data = await res.json();
  const picker = $('#framePicker');
  picker.innerHTML = '';
  if (!data.frames.length) {
    picker.innerHTML = '<span class="muted">No frames found. Run <code>python scripts/download_sample.py</code></span>';
    return;
  }
  data.frames.forEach((f, i) => {
    const chip = document.createElement('button');
    chip.className = 'frame-chip';
    chip.innerHTML = `Frame ${String(f.idx).padStart(6, '0')}` +
      (f.has_labels ? '<span class="gt-dot" title="ground-truth labels available"></span>' : '');
    chip.onclick = () => selectFrame(f.idx, chip);
    picker.appendChild(chip);
    if (i === 0) selectFrame(f.idx, chip);
  });
  // Optional auto-run (handy for demos / screenshots): open with #autorun.
  // #autorun-instant additionally disables fade animations for clean captures.
  if (location.hash.includes('autorun')) {
    if (location.hash.includes('instant')) document.body.classList.add('instant');
    setTimeout(run, 300);
  }
}

function selectFrame(idx, chip) {
  state.frame = idx;
  $$('.frame-chip').forEach((c) => c.classList.remove('selected'));
  chip.classList.add('selected');
  $('#runBtn').disabled = false;
}

/* ---------- count-up animation ---------- */
function countUp(el, target) {
  const start = parseInt(el.dataset.val || '0', 10) || 0;
  const dur = 750, t0 = performance.now();
  function frame(t) {
    const p = Math.min((t - t0) / dur, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = Math.round(start + (target - start) * eased).toLocaleString();
    if (p < 1) requestAnimationFrame(frame);
    else el.dataset.val = target;
  }
  requestAnimationFrame(frame);
}

/* ---------- stepper choreography ---------- */
function resetSteps() {
  $$('.step').forEach((s) => s.classList.remove('active', 'done'));
  $$('.flow').forEach((f) => f.classList.remove('active'));
}
function runStepAnimation() {
  resetSteps();
  const steps = $$('.step'), flows = $$('.flow');
  let i = 0;
  steps[0].classList.add('active');
  return setInterval(() => {
    if (i < steps.length - 1) {
      steps[i].classList.remove('active');
      steps[i].classList.add('done');
      if (flows[i]) flows[i].classList.add('active');
      i++;
      steps[i].classList.add('active');
    }
  }, 420);
}
function finishSteps() {
  $$('.step').forEach((s) => { s.classList.remove('active'); s.classList.add('done'); });
  $$('.flow').forEach((f) => f.classList.remove('active'));
}

/* ---------- main run ---------- */
async function run() {
  if (state.running || state.frame === null) return;
  state.running = true;
  const btn = $('#runBtn');
  btn.classList.add('loading'); btn.disabled = true;
  $('#scanline').classList.add('run');
  $('#placeholder').style.display = 'none';
  const stepTimer = runStepAnimation();

  try {
    const res = await fetch(`/api/run/${state.frame}`);
    const data = await res.json();
    state.images = data.images;

    finishSteps();
    showImage('detections_3d');
    $$('.tab').forEach((t) => t.classList.toggle('active', t.dataset.img === 'detections_3d'));

    // BEV
    const bev = $('#bevImg');
    bev.onload = () => bev.classList.add('show', 'swap');
    bev.src = data.images.bev;
    if (bev.complete && bev.naturalWidth) bev.onload();
    $('#bevPlaceholder').style.display = 'none';

    // Stats
    countUp($('#statLidar'), data.n_lidar);
    countUp($('#stat2d'), data.n_2d);
    countUp($('#stat3d'), data.n_3d);
    const miou = data.metrics ? data.metrics.overall.mean_iou : null;
    $('#statMiou').textContent = miou !== null ? miou.toFixed(2) : '—';

    renderTable(data.detections);
    renderMetrics(data.metrics);
    $('#stats').classList.remove('flash'); void $('#stats').offsetWidth; $('#stats').classList.add('flash');
  } catch (e) {
    alert('Run failed: ' + e);
    console.error(e);
  } finally {
    clearInterval(stepTimer);
    $('#scanline').classList.remove('run');
    btn.classList.remove('loading'); btn.disabled = false;
    state.running = false;
  }
}

function showImage(key) {
  const img = $('#mainImg');
  if (!state.images[key]) return;
  img.classList.remove('show', 'swap');
  img.onload = () => { img.classList.add('show'); void img.offsetWidth; img.classList.add('swap'); };
  img.src = state.images[key];
  if (img.complete && img.naturalWidth) img.onload();
  $('#caption').textContent = CAPTIONS[key] || '';
}

/* ---------- tables + metrics ---------- */
function renderTable(dets) {
  const tb = $('#detTable tbody');
  if (!dets.length) {
    tb.innerHTML = '<tr><td colspan="6" class="muted center">No 3D boxes produced for this frame</td></tr>';
    return;
  }
  tb.innerHTML = dets.map((d) => {
    const [h, w, l] = d.dimensions_hwl, [x, y, z] = d.location;
    return `<tr>
      <td><span class="cls-pill cls-${d.kitti_class}">${d.kitti_class}</span></td>
      <td>${d.score.toFixed(2)}</td>
      <td>${d.depth.toFixed(1)} m</td>
      <td>${h.toFixed(2)}×${w.toFixed(2)}×${l.toFixed(2)}</td>
      <td>(${x.toFixed(1)}, ${y.toFixed(1)}, ${z.toFixed(1)})</td>
      <td>${d.n_points}</td>
    </tr>`;
  }).join('');
}

function metricBar(label, value) {
  return `<div class="metric-row">
    <div class="metric-top"><span>${label}</span><b>${(value * 100).toFixed(0)}%</b></div>
    <div class="bar"><span data-w="${(value * 100).toFixed(0)}"></span></div>
  </div>`;
}

function renderMetrics(metrics) {
  const box = $('#metricsBox');
  if (!metrics) {
    box.innerHTML = '<p class="muted center">No ground-truth labels for this frame.</p>';
    return;
  }
  const o = metrics.overall;
  let html = metricBar('Precision', o.precision) + metricBar('Recall', o.recall) + metricBar('Mean BEV IoU', o.mean_iou);
  const classes = Object.entries(metrics.per_class).filter(([, s]) => s.n_gt > 0 || s.n_pred > 0);
  if (classes.length) {
    html += '<div class="metric-classes">' + classes.map(([c, s]) =>
      `<div class="mc"><span>${c}</span><span>P ${s.precision} · R ${s.recall} · IoU ${s.mean_iou} · ${s.tp}/${s.n_gt} GT</span></div>`
    ).join('') + '</div>';
  }
  html += `<p class="muted small" style="margin-top:14px">Matched at IoU ≥ ${metrics.iou_thresh} in the bird's-eye plane.</p>`;
  box.innerHTML = html;
  // Animate bars after layout.
  requestAnimationFrame(() => $$('.bar > span').forEach((s) => { s.style.width = s.dataset.w + '%'; }));
}

/* ---------- wiring ---------- */
$('#runBtn').addEventListener('click', run);
$$('.tab').forEach((t) => t.addEventListener('click', () => {
  $$('.tab').forEach((x) => x.classList.remove('active'));
  t.classList.add('active');
  showImage(t.dataset.img);
}));
loadFrames();
