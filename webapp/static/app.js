import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const heightInput = document.getElementById("height-mm");
const methodSelect = document.getElementById("method-select");
const useDepthRow = document.getElementById("use-depth-row");
const useDepthCheckbox = document.getElementById("use-depth");
const bustModeRow = document.getElementById("bust-mode-row");
const bustModeCheckbox = document.getElementById("bust-mode");
const cropBaseField = document.getElementById("crop-base-field");
const cropPreview = document.getElementById("crop-preview");
const cropPreviewImg = document.getElementById("crop-preview-img");
const cropShade = document.getElementById("crop-shade");
const cropLine = document.getElementById("crop-line");
const generateBtn = document.getElementById("generate-btn");
const resetBtn = document.getElementById("reset-btn");
const statusBox = document.getElementById("status-box");
const messagesEl = document.getElementById("messages");
const modeBanner = document.getElementById("mode-banner");
const emptyState = document.getElementById("empty-state");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingMsg = document.getElementById("loading-msg");
const statsBar = document.getElementById("stats-bar");
const downloadBar = document.getElementById("download-bar");
const trellisStatusEl = document.getElementById("trellis-status");
const descriptionField = document.getElementById("description-field");
const descriptionInput = document.getElementById("description-input");
const descriptionHint = document.getElementById("description-hint");
const tabBtns = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");
const emptyResultHint = document.getElementById("empty-result-hint");

let selectedFiles = [];
let trellisStatusLoaded = false;

function setActiveTab(name) {
  tabBtns.forEach((btn) => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });
}

tabBtns.forEach((btn) => btn.addEventListener("click", () => setActiveTab(btn.dataset.tab)));

// The Android app's WebView loads this same URL (see android-app/capacitor.config.ts's
// server.url), so a deploy here is already "live" there with no separate app update --
// but the WebView (like any browser) can still hold a cached copy of index.html/app.js/
// style.css for a while. This button forces a real network re-fetch of the page past
// that cache, landing on whatever is currently deployed.
document.getElementById("refresh-app-btn").addEventListener("click", () => {
  window.location.href = window.location.pathname + "?_r=" + Date.now();
});

// Neither reconstruction method actually consumes this text as a model
// input right now -- confirmed directly, not assumed: TRELLIS.2's own
// Space API (inspected via gradio_client.Client(...).view_api()) exposes
// no text parameter at all on its image_to_3d endpoint, and this
// project's own capsule/silhouette pipelines are purely vision-based, no
// text-conditioning hook exists anywhere in them either. So: captured and
// stored with the job for your own reference and for future use, but
// honestly not yet steering the render -- said plainly here instead of
// implying it does something it doesn't.
descriptionHint.textContent =
  "Not used by the 3D model yet (no text input exists on either reconstruction path right now) -- saved with your job for reference.";

function updateDescriptionVisibility() {
  descriptionField.style.display = selectedFiles.length > 0 ? "" : "none";
}

async function loadTrellisStatus() {
  trellisStatusEl.style.display = "block";
  trellisStatusEl.textContent = "Checking TRELLIS.2 availability…";
  try {
    const res = await fetch("/api/trellis-status");
    const data = await res.json();
    const parts = [];
    if (!data.token_configured) {
      parts.push(`<span class="quota-warn">No server token configured — likely to hit rate limits fast.</span>`);
    } else {
      parts.push(`Server token: <b>${data.token_preview}</b>`);
      if (data.quota.available) {
        const mins = (data.quota.remaining_seconds / 60).toFixed(1);
        const cls = data.quota.remaining_seconds > 60 ? "quota-ok" : "quota-warn";
        parts.push(`<span class="${cls}">${mins} min ZeroGPU quota left</span>`);
      } else {
        parts.push(`real quota unknown (token lacks billing-read permission)`);
      }
    }
    parts.push(
      `this session: ${data.stats.attempts} tried, ${data.stats.successes} succeeded, ${data.stats.fallbacks} fell back`
    );
    trellisStatusEl.innerHTML = parts.join(" · ");
    trellisStatusLoaded = true;
  } catch (err) {
    trellisStatusEl.textContent = "Couldn't check TRELLIS.2 status (network error).";
  }
}

function updateModeBanner() {
  if (selectedFiles.length === 0) {
    modeBanner.textContent = "Add a photo to begin.";
  } else if (methodSelect.value === "trellis") {
    modeBanner.textContent =
      selectedFiles.length === 1
        ? "TRELLIS.2 mode: sent to an external MIT-licensed model, confirmed good quality on human photos — animal photos are refused server-side (confirmed unreliable)."
        : "TRELLIS.2 takes exactly one photo — remove the extra ones, or switch back to the built-from-scratch method for multi-photo mode.";
  } else if (methodSelect.value === "silhouette_relief") {
    modeBanner.textContent =
      selectedFiles.length === 1
        ? "Silhouette relief mode: the mesh's outline follows the photo's own 2D cutout directly, extruded into 3D — no anatomical guess beyond what the cutout itself shows."
        : "Silhouette relief takes exactly one photo — remove the extra ones, or switch back to the capsule method for multi-photo mode.";
  } else if (selectedFiles.length === 1) {
    modeBanner.textContent =
      "Single-photo mode: a stylized geometric guess, with real face detail (nose/chin/eyes) when a face is detected.";
  } else {
    modeBanner.textContent = `Multi-photo mode (${selectedFiles.length} photos): visual-hull reconstruction from your photos' actual silhouettes — shoot them in turntable order, front first.`;
  }
}

function updateMethodUI() {
  useDepthRow.style.display = methodSelect.value === "trellis" ? "none" : "";
  bustModeRow.style.display = methodSelect.value === "silhouette_relief" ? "" : "none";
  updateModeBanner();
  updateCropTool();
  if (methodSelect.value === "trellis") {
    loadTrellisStatus();
  } else {
    trellisStatusEl.style.display = "none";
  }
}

// ---------------- custom base/crop tool ----------------
// 1.0 = the full photo, nothing cropped away -- the default, matching
// the backend's own crop_frac=None meaning "use the whole photo".
let cropFrac = 1.0;
let cropDragging = false;

function setCropLine(frac) {
  cropFrac = Math.min(1, Math.max(0.05, frac));
  const pct = (cropFrac * 100).toFixed(2) + "%";
  cropLine.style.top = pct;
  cropShade.style.top = pct;
}

function updateCropTool() {
  const show = methodSelect.value === "silhouette_relief" && bustModeCheckbox.checked && selectedFiles.length === 1;
  cropBaseField.style.display = show ? "" : "none";
  if (show) {
    cropPreviewImg.src = URL.createObjectURL(selectedFiles[0]);
    setCropLine(cropFrac);
  }
}
bustModeCheckbox.addEventListener("change", updateCropTool);

function cropFracFromPointer(e) {
  const rect = cropPreview.getBoundingClientRect();
  return (e.clientY - rect.top) / rect.height;
}
cropPreview.addEventListener("pointerdown", (e) => {
  cropDragging = true;
  cropPreview.setPointerCapture(e.pointerId);
  setCropLine(cropFracFromPointer(e));
});
cropPreview.addEventListener("pointermove", (e) => {
  if (cropDragging) setCropLine(cropFracFromPointer(e));
});
cropPreview.addEventListener("pointerup", () => {
  cropDragging = false;
});
methodSelect.addEventListener("change", updateMethodUI);

function renderFileList() {
  fileList.innerHTML = "";
  selectedFiles.forEach((file, i) => {
    const chip = document.createElement("div");
    chip.className = "file-chip";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = file.name;
    const x = document.createElement("span");
    x.className = "x";
    x.textContent = "✕";
    x.onclick = (e) => {
      e.stopPropagation();
      selectedFiles.splice(i, 1);
      renderFileList();
      updateModeBanner();
    };
    chip.appendChild(img);
    chip.appendChild(name);
    chip.appendChild(x);
    fileList.appendChild(chip);
  });
  updateModeBanner();
  updateDescriptionVisibility();
  updateCropTool();
}

function addFiles(fileListObj) {
  for (const f of fileListObj) {
    if (f.type.startsWith("image/")) selectedFiles.push(f);
  }
  renderFileList();
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => addFiles(e.target.files));

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag");
  })
);
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files) addFiles(e.dataTransfer.files);
});

function setStatus(kind, text) {
  statusBox.className = `status-box show ${kind}`;
  statusBox.textContent = text;
  emptyResultHint.style.display = "none";
  // Any status (success or error) means there's something to look at —
  // jump to the Result tab instead of leaving it set on a tab the user
  // isn't looking at (the exact failure mode the old flat single-panel
  // layout had: a generate error could land below the fold with no
  // indication anything happened).
  setActiveTab("result");
}
function clearStatus() {
  statusBox.className = "status-box";
  statusBox.textContent = "";
}

function renderMessages(messages) {
  messagesEl.innerHTML = "";
  (messages || []).forEach((m) => {
    const div = document.createElement("div");
    const isWarn = /warning|note/i.test(m);
    div.className = "message" + (isWarn ? " warn" : "");
    div.textContent = m;
    messagesEl.appendChild(div);
  });
}

// ---------------- three.js viewer ----------------

let scene, camera, renderer, controls, currentModel, groundPlane;
// The GLB download link (see loadModel's OBJ-based preview below) — a
// general-purpose colored-mesh export other tools can consume, not tied
// to any particular viewer or display feature.
let currentGlbUrl = null;

function initViewer() {
  const canvas = document.getElementById("viewer-canvas");
  const stage = document.getElementById("stage");

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
  camera.position.set(140, 160, 220);

  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 60, 0);
  // Inverts drag-to-pan direction (two-finger drag on touch, right-click
  // drag on desktop) -- directly requested. OrbitControls multiplies its
  // pan delta by this each frame, so negating it flips the direction
  // without touching anything else (zoom/rotate speeds are separate
  // properties, untouched here).
  controls.panSpeed = -1;

  const hemi = new THREE.HemisphereLight(0xaecbff, 0x1a1a1f, 1.1);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.6);
  dir.position.set(120, 220, 140);
  scene.add(dir);
  const fill = new THREE.DirectionalLight(0x6ea8ff, 0.35);
  fill.position.set(-150, 80, -100);
  scene.add(fill);

  const groundGeo = new THREE.PlaneGeometry(2000, 2000);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x12141b,
    roughness: 1,
    metalness: 0,
  });
  groundPlane = new THREE.Mesh(groundGeo, groundMat);
  groundPlane.rotation.x = -Math.PI / 2;
  groundPlane.position.y = 0;
  scene.add(groundPlane);

  const grid = new THREE.GridHelper(1000, 50, 0x2a3140, 0x1c212c);
  grid.position.y = 0.01;
  scene.add(grid);

  function resize() {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  window.addEventListener("resize", resize);
  resize();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

// A plain min/max bounding box (the previous implementation here) frames
// the camera around EVERY vertex, including a single stray one -- a bad
// detection, a mis-synthesized leg/hair extension, anything far from the
// real figure -- which balloons the box and leaves the actual model tiny
// and pushed off to one side of an otherwise near-empty viewer. Reported
// directly (a real screenshot: the model sitting small in a corner, most
// of the viewer dark and empty) but not reproducible against this
// project's own test photos, so this can't be fixed by chasing one
// specific cause -- using the 1st-99th percentile of vertex positions per
// axis instead of their raw extremes makes framing robust to that whole
// class of outlier regardless of what produced it, at the cost of (at
// most) a sliver of the true extremes going very slightly out of frame,
// which Reset view already exists to recover from.
function _robustRange(values) {
  if (values.length === 0) return [0, 1];
  const sorted = [...values].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.01)];
  const hi = sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.99) - 1)];
  return lo < hi ? [lo, hi] : [sorted[0], sorted[sorted.length - 1]];
}

function frameCamera(object3d) {
  object3d.updateMatrixWorld(true);
  const xs = [], ys = [], zs = [];
  const v = new THREE.Vector3();
  object3d.traverse((child) => {
    if (!child.isMesh || !child.geometry || !child.geometry.attributes.position) return;
    const pos = child.geometry.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(child.matrixWorld);
      xs.push(v.x);
      ys.push(v.y);
      zs.push(v.z);
    }
  });

  const [xlo, xhi] = _robustRange(xs);
  const [ylo, yhi] = _robustRange(ys);
  const [zlo, zhi] = _robustRange(zs);
  const center = new THREE.Vector3((xlo + xhi) / 2, (ylo + yhi) / 2, (zlo + zhi) / 2);
  const size = new THREE.Vector3(xhi - xlo, yhi - ylo, zhi - zlo);
  let radius = Math.max(size.x, size.y, size.z) * 0.72;
  // A degenerate mesh (too few vertices, all coincident) could leave
  // radius at 0 or NaN -- never let that reach camera.position below.
  if (!isFinite(radius) || radius <= 0) radius = 100;

  controls.target.copy(center);
  const dir = new THREE.Vector3(0.55, 0.42, 0.9).normalize();
  camera.position.copy(center).add(dir.multiplyScalar(radius * 2.6));
  camera.near = radius / 100;
  camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

function loadModel(objUrl) {
  if (currentModel) {
    scene.remove(currentModel);
    currentModel.traverse((c) => {
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    });
    currentModel = null;
  }
  const loader = new OBJLoader();
  loader.load(objUrl, (obj) => {
    obj.traverse((child) => {
      if (child.isMesh) {
        // OBJLoader already parses the extended "v x y z r g b" vertex-color
        // lines the pipeline now writes (real colors baked from the source
        // photo — see procedural_body._bake_photo_colors) into
        // geometry.attributes.color; vertexColors just has to be turned on
        // to actually use them instead of a flat guessed color. Falls back
        // to the old flat tint automatically when a mesh has no baked
        // colors (geometry.attributes.color is then unset).
        const hasVertexColors = !!child.geometry.attributes.color;
        child.material = new THREE.MeshStandardMaterial({
          color: hasVertexColors ? 0xffffff : 0x9fb3d9,
          vertexColors: hasVertexColors,
          roughness: 0.55,
          metalness: 0.05,
        });
        child.castShadow = false;
      }
    });

    // Sit the model on the ground plane: translate so its own minimum Y
    // touches y=0, matching the print-bed framing the preview is for.
    const box = new THREE.Box3().setFromObject(obj);
    obj.position.y -= box.min.y;

    scene.add(obj);
    currentModel = obj;
    frameCamera(obj);
    emptyState.style.display = "none";
  });
}

document.getElementById("reset-view-btn").addEventListener("click", () => {
  if (currentModel) frameCamera(currentModel);
});

// ---------------- form submit ----------------
//
// /api/generate itself now returns almost immediately (just a job_id --
// see webapp/app.py's own comment on why), and the actual generation
// keeps running server-side in its own thread regardless of whether this
// tab/app is even open. The client's job here is just to poll for it to
// finish instead of holding one connection open the whole time, which
// means backgrounding the app (or Android killing it outright) no longer
// loses the result -- reopening the app finds the saved job_id in
// localStorage and picks the poll back up, or finds it already done.

const LAST_JOB_KEY = "figuro_pending_job_id";
let pollTimer = null;

function stopPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  localStorage.removeItem(LAST_JOB_KEY);
}

function finishedGenerating() {
  generateBtn.disabled = false;
  loadingOverlay.classList.remove("show");
}

function showResult(data) {
  if (data.method_used === "capsule (fallback)") {
    setStatus("ok", "Model generated — TRELLIS.2 was unavailable, so this used the built-in method instead.");
  } else {
    setStatus("ok", "Model generated.");
  }
  renderMessages(data.messages);
  loadModel(data.obj_url);
  currentGlbUrl = data.glb_url;
  if (methodSelect.value === "trellis") loadTrellisStatus();

  statsBar.style.display = "flex";
  statsBar.innerHTML = `
    <span><b>${data.stats.faces.toLocaleString()}</b> faces</span>
    <span><b>${data.stats.height_mm.toFixed(1)}mm</b> tall</span>
    <span><b>${data.stats.watertight ? "watertight" : "needs repair"}</b></span>
  `;
  downloadBar.innerHTML = `
    <a href="${data.stl_url}" download>Download STL</a>
    <a class="secondary" href="${data.obj_url}" download>Download OBJ</a>
    <a class="secondary" href="${data.glb_url}" download>Download GLB</a>
  `;
}

async function pollJob(jobId) {
  let res;
  try {
    res = await fetch(`/api/jobs/${jobId}/status`);
  } catch (err) {
    // likely just coming back from background/offline -- keep trying
    // rather than giving up on one dropped request.
    pollTimer = setTimeout(() => pollJob(jobId), 3000);
    return;
  }
  if (res.status === 404) {
    stopPolling();
    finishedGenerating();
    setStatus("error", "That job wasn't found — it may have expired (jobs are kept for 2 hours).");
    return;
  }
  const data = await res.json();
  if (data.status === "running") {
    pollTimer = setTimeout(() => pollJob(jobId), 2000);
    return;
  }
  stopPolling();
  finishedGenerating();
  if (data.status === "error") {
    setStatus("error", data.detail || "Something went wrong.");
    return;
  }
  showResult(data);
}

// A throttled background tab/app can sit on a stale setTimeout for a
// while before it next fires -- re-poll right away the moment the page
// is actually visible again instead of waiting that out.
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState !== "visible") return;
  const jobId = localStorage.getItem(LAST_JOB_KEY);
  if (!jobId) return;
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  pollJob(jobId);
});

async function generate() {
  if (selectedFiles.length === 0) {
    setStatus("error", "Add at least one photo first.");
    return;
  }
  const method = methodSelect.value;
  if ((method === "trellis" || method === "silhouette_relief") && selectedFiles.length !== 1) {
    setStatus("error", `${method === "trellis" ? "TRELLIS.2" : "Silhouette relief"} takes exactly one photo — remove the extras or switch methods.`);
    return;
  }
  clearStatus();
  messagesEl.innerHTML = "";
  generateBtn.disabled = true;
  loadingOverlay.classList.add("show");
  loadingMsg.textContent =
    method === "trellis"
      ? "Sending to TRELLIS.2 (external model, this can take a minute)…"
      : method === "silhouette_relief"
      ? "Extracting the photo's own outline and extruding it into 3D…"
      : selectedFiles.length === 1
      ? "Detecting pose and face, building the figure…"
      : "Extracting silhouettes and carving the visual hull…";
  downloadBar.innerHTML = "";
  statsBar.style.display = "none";

  const form = new FormData();
  selectedFiles.forEach((f) => form.append("images", f));
  form.append("height_mm", heightInput.value || "150");
  form.append("use_depth", useDepthCheckbox.checked ? "true" : "false");
  form.append("method", method);
  form.append("description", descriptionInput.value || "");
  form.append("bust_mode", bustModeCheckbox.checked ? "true" : "false");
  if (cropBaseField.style.display !== "none" && cropFrac < 0.999) {
    form.append("crop_frac", String(cropFrac));
  }

  try {
    const res = await fetch("/api/generate", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      setStatus("error", data.detail || "Something went wrong.");
      finishedGenerating();
      return;
    }
    localStorage.setItem(LAST_JOB_KEY, data.job_id);
    pollJob(data.job_id);
  } catch (err) {
    setStatus("error", "Network error: " + err.message);
    finishedGenerating();
  }
}

generateBtn.addEventListener("click", generate);
resetBtn.addEventListener("click", () => {
  selectedFiles = [];
  descriptionInput.value = "";
  cropFrac = 1.0;
  stopPolling();
  renderFileList();
  clearStatus();
  messagesEl.innerHTML = "";
  emptyResultHint.style.display = "";
  statsBar.style.display = "none";
  downloadBar.innerHTML = "";
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }
  currentGlbUrl = null;
  emptyState.style.display = "flex";
  setActiveTab("photos");
});

initViewer();
renderFileList();

// A direct, unambiguous way to confirm a given page load is actually on
// a given deploy -- see /api/version in webapp/app.py -- instead of
// having to guess from whether something visibly changed.
fetch("/api/version")
  .then((r) => r.json())
  .then((v) => {
    document.getElementById("build-tag").textContent = v.commit ? `· build ${v.commit}` : "";
  })
  .catch(() => {});

// Resume a job that was still running when this page/app was last closed
// or backgrounded hard enough to be killed -- the job itself kept running
// server-side the whole time (see _run_job in webapp/app.py), so there is
// still a real result to pick back up, not just a lost request. Placed
// after initViewer/renderFileList so everything pollJob's success path
// touches (the three.js scene, the file list) is already set up.
(function resumePendingJob() {
  const jobId = localStorage.getItem(LAST_JOB_KEY);
  if (!jobId) return;
  generateBtn.disabled = true;
  loadingOverlay.classList.add("show");
  loadingMsg.textContent = "Picking back up a model that was already generating…";
  setActiveTab("result");
  pollJob(jobId);
})();
