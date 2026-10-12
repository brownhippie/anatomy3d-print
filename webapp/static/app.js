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
const settingsGearBtn = document.getElementById("settings-gear-btn");
const settingsDrawer = document.getElementById("settings-drawer");
const settingsDrawerBackdrop = document.getElementById("settings-drawer-backdrop");
const settingsDrawerClose = document.getElementById("settings-drawer-close");

let selectedFiles = [];
let trellisStatusLoaded = false;

const mainEl = document.querySelector("main");

function setActiveTab(name) {
  tabBtns.forEach((btn) => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });
  // On a phone, "Viewer" takes over the full screen (the panel collapses to
  // just its tab bar) instead of the 3D model living in a sliver below an
  // upload/settings form -- see the main.viewer-active rules in style.css.
  mainEl.classList.toggle("viewer-active", name === "viewer");
  // #stage is `display:none` on a phone until this tab is active (that's
  // the whole fix above), which means its own size was 0x0 for however
  // long it sat hidden -- including at page load, when initViewer()'s own
  // first resize() ran. Nothing re-measures it just because CSS display
  // flips, so without this the canvas stays stuck at that stale 0x0 size
  // (confirmed directly: the model loads fine but the canvas paints
  // nothing, a blank Viewer tab) -- piggyback on the resize listener
  // initViewer() already wires up, now that #stage actually has a size.
  if (name === "viewer") window.dispatchEvent(new Event("resize"));
  if (name === "lab") labOnShow();
}

tabBtns.forEach((btn) => btn.addEventListener("click", () => setActiveTab(btn.dataset.tab)));

// Settings (method, height, bust mode, crop tool) live in a gear-icon drawer
// instead of a tab, so the Photos/Result tabs stay short and the 3D viewer
// below them keeps most of the screen on a phone instead of being squeezed
// out by an inline settings form.
function openSettingsDrawer() { settingsDrawer.classList.add("open"); }
function closeSettingsDrawer() { settingsDrawer.classList.remove("open"); }
settingsGearBtn.addEventListener("click", openSettingsDrawer);
settingsDrawerBackdrop.addEventListener("click", closeSettingsDrawer);
settingsDrawerClose.addEventListener("click", closeSettingsDrawer);

// The Android app's WebView loads this same URL (see android-app/capacitor.config.ts's
// server.url), so a deploy here is already "live" there with no separate app update --
// but the WebView (like any browser) can still hold a cached copy of index.html/app.js/
// style.css for a while. This button forces a real network re-fetch of the page past
// that cache, landing on whatever is currently deployed.
document.getElementById("refresh-app-btn").addEventListener("click", () => {
  window.location.href = window.location.pathname + "?_r=" + Date.now();
});

// Inside the actual Android app's WebView (Capacitor injects window.Capacitor
// into every page it loads, remote URL or not -- see android-app/capacitor.config.ts),
// a "Get the Android app" link makes no sense: you're already in it. Swap it
// for what someone in that position actually wants -- confirmation they're on
// the latest deploy -- by relabeling the existing Refresh button (it already
// does exactly that, see above) instead of adding a second, separate control.
// In an ordinary mobile/desktop browser, window.Capacitor stays undefined and
// both buttons stay as they are.
//
// window.Capacitor isn't guaranteed to exist the instant this (deferred
// module) script runs -- confirmed directly as a real bug, not a guess:
// reported as the header visibly flickering between "Get the Android app"
// and "Check for updates" rather than settling on one. A single check-once
// could miss the bridge still initializing and wrongly keep the browser UI
// for that whole page load, then catch it on the next (e.g. after tapping
// the refresh/update button itself) -- flipping between the two across
// reloads, not within one. Retries briefly instead of checking once.
(function detectNativeApp(attemptsLeft = 20) {
  if (window.Capacitor && window.Capacitor.isNativePlatform && window.Capacitor.isNativePlatform()) {
    document.getElementById("get-app-link").style.display = "none";
    document.getElementById("refresh-app-btn").textContent = "⟲ Check for updates";
  } else if (attemptsLeft > 0) {
    setTimeout(() => detectNativeApp(attemptsLeft - 1), 50);
  }
})();

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
  // A generated model is what someone actually wants to see -- jump
  // straight to the Viewer tab for it. An error has no model to show, so
  // it still jumps to Result (the exact failure mode the old flat
  // single-panel layout had: a generate error could land below the fold
  // with no indication anything happened).
  setActiveTab(kind === "ok" ? "viewer" : "result");
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
// The object's own center/radius (set by frameCamera below), used to keep
// pan from carrying the view's focus away from the model -- see the
// controls "change" listener in initViewer.
const modelCenter = new THREE.Vector3();
let modelRadius = 100;
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
  // One-finger drag orbits the model; that's the drag gesture someone on
  // a phone actually uses, and it was going the unintuitive way -- flip
  // it (rotateSpeed, not panSpeed: the earlier panSpeed=-1 fix inverted
  // a gesture -- two-finger drag/right-click pan -- that one-finger
  // orbiting never actually used).
  controls.rotateSpeed = -1;
  // Two-finger pinch keeps OrbitControls' default DOLLY_PAN touch gesture
  // -- finger-distance change zooms, the two fingers' average movement
  // pans -- directly requested as "pinch for movement" alongside
  // one-finger rotate. (An earlier attempt disabled panning outright
  // because an *inverted* panSpeed was making any incidental pan inside
  // a pinch feel like drift; panSpeed is left at its default direction
  // here; if panning itself turns out backwards, that's the one line to
  // flip, not a reason to disable it again.)
  //
  // Pan still shouldn't be able to carry the view arbitrarily far from
  // the model, though -- zoom dollies toward/away from controls.target,
  // so a target that's drifted away from the object makes zoom feel
  // like it's no longer centered on it -- directly requested ("keep
  // everything focused towards the object, even zoom"). Clamp target to
  // a radius around the model's own center on every control change
  // (pan, rotate, or dolly all fire this) instead of only on Reset view.
  controls.addEventListener("change", () => {
    const maxDrift = modelRadius * 0.6;
    const offset = controls.target.clone().sub(modelCenter);
    if (offset.length() > maxDrift) {
      controls.target.copy(modelCenter).add(offset.setLength(maxDrift));
    }
  });

  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  // Studio lighting that travels with the camera, so whichever side you
  // orbit to is lit the same way instead of falling into shadow. Positions
  // are in camera space; each light aims at a point straight ahead.
  scene.add(camera);
  const aimAhead = new THREE.Object3D();
  aimAhead.position.set(0, 0, -1);
  camera.add(aimAhead);
  const addCameraLight = (color, intensity, x, y, z) => {
    const light = new THREE.DirectionalLight(color, intensity);
    light.position.set(x, y, z);
    light.target = aimAhead;
    camera.add(light);
  };
  addCameraLight(0xfff0e0, 1.9, -0.7, 0.9, 0.3); // key: upper left, warm
  addCameraLight(0xffe9d8, 0.9, 0.9, 0.1, 0.2); // fill: right, keeps shadows soft
  addCameraLight(0xffffff, 1.2, 0.6, 0.8, -3); // rim: from behind, outlines the figure
  scene.add(new THREE.HemisphereLight(0xfff6ee, 0x6b5a4c, 1.0));

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

  modelCenter.copy(center);
  modelRadius = radius;
  // Close enough to inspect a face, never inside the model; far enough to
  // see it whole, never so far it's lost in the scene.
  controls.minDistance = radius * 0.45;
  controls.maxDistance = radius * 8;
  controls.target.copy(center);
  const dir = new THREE.Vector3(0.55, 0.42, 0.9).normalize();
  camera.position.copy(center).add(dir.multiplyScalar(radius * 2.6));
  camera.near = radius / 100;
  camera.far = radius * 100;
  camera.updateProjectionMatrix();
  controls.update();
}

// Soft sand clay: a warm, light, mostly matte finish so form reads from
// gentle light and shadow without any photo colour.
const FIGURE_MATERIAL = new THREE.MeshPhysicalMaterial({
  color: 0xf0d6bd,
  roughness: 0.62,
  metalness: 0,
  clearcoat: 0.12,
  clearcoatRoughness: 0.6,
  sheen: 0.4,
  sheenColor: new THREE.Color(0xfff0e0),
  sheenRoughness: 0.8,
});

// The exported OBJs carry no normals, so OBJLoader shades every triangle
// flat, which turns each small step in the mesh into a visible facet.
// Average each corner's normal over the faces that meet at that point,
// skipping faces that bend more than creaseDeg away, so curved surfaces
// shade smoothly while real edges (the base disk's rim) stay sharp.
function smoothShade(geometry, creaseDeg = 50) {
  const pos = geometry.attributes.position;
  const corners = pos.count;
  const faces = corners / 3;
  // fn: unit face normals (for the crease test); fw: the same normals
  // scaled by face area (for the average). Grid-extracted meshes are full
  // of thin sliver triangles laid out in rows; giving each an equal vote
  // skews the normals row by row and shades as horizontal bands.
  const fn = new Float32Array(faces * 3);
  const fw = new Float32Array(faces * 3);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let f = 0; f < faces; f++) {
    a.fromBufferAttribute(pos, 3 * f);
    b.fromBufferAttribute(pos, 3 * f + 1).sub(a);
    c.fromBufferAttribute(pos, 3 * f + 2).sub(a);
    b.cross(c);
    fw[3 * f] = b.x; fw[3 * f + 1] = b.y; fw[3 * f + 2] = b.z;
    b.normalize();
    fn[3 * f] = b.x; fn[3 * f + 1] = b.y; fn[3 * f + 2] = b.z;
  }
  const byPoint = new Map();
  const keys = new Array(corners);
  for (let i = 0; i < corners; i++) {
    const k = `${Math.round(pos.getX(i) * 1000)},${Math.round(pos.getY(i) * 1000)},${Math.round(pos.getZ(i) * 1000)}`;
    keys[i] = k;
    const list = byPoint.get(k);
    if (list) list.push(Math.floor(i / 3)); else byPoint.set(k, [Math.floor(i / 3)]);
  }
  const cosCrease = Math.cos((creaseDeg * Math.PI) / 180);
  const normals = new Float32Array(corners * 3);
  for (let i = 0; i < corners; i++) {
    const f = Math.floor(i / 3);
    const fx = fn[3 * f], fy = fn[3 * f + 1], fz = fn[3 * f + 2];
    let sx = 0, sy = 0, sz = 0;
    for (const g of byPoint.get(keys[i])) {
      if (fx * fn[3 * g] + fy * fn[3 * g + 1] + fz * fn[3 * g + 2] >= cosCrease) {
        sx += fw[3 * g]; sy += fw[3 * g + 1]; sz += fw[3 * g + 2];
      }
    }
    const len = Math.hypot(sx, sy, sz) || 1;
    normals[3 * i] = sx / len; normals[3 * i + 1] = sy / len; normals[3 * i + 2] = sz / len;
  }
  geometry.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
}

// Viewer finishes (display only, instant, no rebuild). "photo" uses the
// photo colours baked into the mesh when it has them (capsule builds), and
// falls back to clay when it doesn't (outline builds carry no colour).
const FINISH_MATERIALS = {
  clay: FIGURE_MATERIAL,
  ivory: new THREE.MeshPhysicalMaterial({ color: 0xe6dccb, roughness: 0.5, metalness: 0, clearcoat: 0.35, clearcoatRoughness: 0.4 }),
  grey: new THREE.MeshStandardMaterial({ color: 0xa9b0b9, roughness: 0.7, metalness: 0 }),
  photo: new THREE.MeshStandardMaterial({ color: 0xffffff, vertexColors: true, roughness: 0.6, metalness: 0 }),
};
const _materialCache = {};
const viewerLook = { finish: "clay", flat: false };
try {
  const saved = JSON.parse(localStorage.getItem("figuro_finish") || "null");
  if (saved && FINISH_MATERIALS[saved.finish]) Object.assign(viewerLook, saved);
} catch (e) { /* storage unavailable: keep defaults */ }

function lookMaterial(finish, flat) {
  const key = `${finish}:${flat}`;
  if (!_materialCache[key]) {
    const m = FINISH_MATERIALS[finish].clone();
    m.flatShading = flat;
    _materialCache[key] = m;
  }
  return _materialCache[key];
}

function applyLook() {
  if (!currentModel) return;
  currentModel.traverse((child) => {
    if (!child.isMesh) return;
    const colors = child.userData.photoColors;
    const usePhoto = viewerLook.finish === "photo" && colors;
    if (usePhoto) child.geometry.setAttribute("color", colors);
    else if (child.geometry.attributes.color) child.geometry.deleteAttribute("color");
    child.material = lookMaterial(usePhoto ? "photo" : (viewerLook.finish === "photo" ? "clay" : viewerLook.finish), viewerLook.flat);
  });
}

function loadModel(objUrl, onLoaded) {
  if (currentModel) {
    scene.remove(currentModel);
    currentModel.traverse((c) => {
      if (c.geometry) c.geometry.dispose();
    });
    currentModel = null;
  }
  const loader = new OBJLoader();
  loader.load(objUrl, (obj) => {
    obj.traverse((child) => {
      if (child.isMesh) {
        if (child.geometry.attributes.color) child.userData.photoColors = child.geometry.attributes.color;
        smoothShade(child.geometry);
      }
    });

    // Sit the model on the ground plane: translate so its own minimum Y
    // touches y=0, matching the print-bed framing the preview is for.
    const box = new THREE.Box3().setFromObject(obj);
    obj.position.y -= box.min.y;

    scene.add(obj);
    currentModel = obj;
    applyLook();
    frameCamera(obj);
    emptyState.style.display = "none";
    if (onLoaded) onLoaded();
  });
}

const finishSelect = document.getElementById("finish-select");
const flatToggle = document.getElementById("flat-toggle");
finishSelect.value = viewerLook.finish;
flatToggle.setAttribute("aria-pressed", String(viewerLook.flat));
function saveLook() {
  try { localStorage.setItem("figuro_finish", JSON.stringify(viewerLook)); } catch (e) { /* ignore */ }
  applyLook();
}
finishSelect.addEventListener("change", () => { viewerLook.finish = finishSelect.value; saveLook(); });
flatToggle.addEventListener("click", () => {
  viewerLook.flat = !viewerLook.flat;
  flatToggle.setAttribute("aria-pressed", String(viewerLook.flat));
  saveLook();
});

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
  } else if (data.method_used === "silhouette_relief (bust)") {
    setStatus("ok", "Model generated as a bust — no hips in the photo, so no legs were added.");
  } else if (data.method_used === "silhouette_relief (outline)") {
    setStatus("ok", "Model generated from the photo's outline — no usable human pose was found (an animal, or a person too hidden to read).");
  } else {
    setStatus("ok", "Model generated.");
  }
  renderMessages(data.messages);
  labBar.hidden = true;
  loadModel(data.preview_url || data.obj_url);
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
    // stage/progress are written by the pipeline's own progress_cb calls
    // (see report_progress in webapp/app.py's _run_job) -- a real
    // countdown of where generation actually is, not a guessed timer.
    // Absent on the very first poll or two, before the background thread
    // has reached its first checkpoint, so fall back to the generic
    // message rather than showing a blank/stale percentage.
    if (typeof data.progress === "number") {
      const pct = Math.round(data.progress * 100);
      loadingMsg.textContent = `${data.stage || "Working"}… ${pct}%`;
    }
    pollTimer = setTimeout(() => pollJob(jobId), 1200);
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

// ---------------- Combination Lab ----------------
//
// One photo -> the server builds every combination on its own (see
// webapp/lab_api.py), scores each, and this tab shows them as they land.
// Tapping one loads it in the viewer with a rating bar; ratings and notes
// are stored on the server so "Copy my results" (or the export link) gives
// Claude everything in one go.

const LAB_RUN_KEY = "figuro_lab_run";
const labBar = document.getElementById("lab-bar");
const labGrid = document.getElementById("lab-grid");
const labMore = document.getElementById("lab-more");
const labSort = document.getElementById("lab-sort");
const labFamily = document.getElementById("lab-family");
const labOnly = document.getElementById("lab-only");
const labRunSelect = document.getElementById("lab-run-select");
const labState = { space: null, runId: null, items: [], offset: 0, count: 0, current: -1, timer: null };
try { labState.runId = localStorage.getItem(LAB_RUN_KEY); } catch (e) { /* storage unavailable */ }

const fmt = (n) => Number(n).toLocaleString();

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function labLoadSpace() {
  if (labState.space) return;
  const s = await (await fetch("/api/lab/space")).json();
  labState.space = s;
  const perView = s.finishes.length * s.shading.length;
  document.getElementById("lab-count").innerHTML = `
    <div class="lab-count-big">${fmt(s.total_builds)}</div>
    <div>build combinations — ${s.families.map((f) => `${escapeHtml(f.name)} ${fmt(f.count)}`).join(" + ")}</div>
    <div class="lab-count-sub">× ${perView} viewer finishes (${s.finishes.length} colours × ${s.shading.length} shadings) = ${fmt(s.total_views)} ways to look at them</div>`;
  document.getElementById("lab-dims").innerHTML = s.families.map((f) => `
    <h4>${escapeHtml(f.name)} — ${fmt(f.count)}</h4>
    <ul>${f.dims.map((d) => `<li><b>${escapeHtml(d.label)}</b> (${d.options.length}): ${d.options.map(escapeHtml).join(" · ")}</li>`).join("")}</ul>
    <div class="hint">${f.dims.map((d) => d.options.length).join(" × ")} = ${fmt(f.count)}</div>`).join("")
    + `<h4>Historical versions included</h4><ul>${s.templates.map((t) => `<li>${escapeHtml(t.name)}</li>`).join("")}</ul>`;
  document.getElementById("lab-score-hint").textContent = `How the auto-score works: ${s.score_formula}.`;
}

async function labRefreshRuns() {
  const data = await (await fetch("/api/lab/runs")).json();
  const runs = data.runs;
  document.getElementById("lab-runs-field").hidden = runs.length === 0;
  labRunSelect.innerHTML = runs.map((r) => `<option value="${r.run_id}">${escapeHtml(r.photo_name)} — ${fmt(r.done)}/${fmt(r.total)} (${r.status})</option>`).join("");
  if (!labState.runId || !runs.some((r) => r.run_id === labState.runId)) labState.runId = runs.length ? runs[0].run_id : null;
  if (labState.runId) labRunSelect.value = labState.runId;
  return data;
}

function labShowRun(run) {
  document.getElementById("lab-run").hidden = !run;
  document.getElementById("lab-controls").hidden = !run;
  if (!run) return;
  document.getElementById("lab-run-photo").src = run.photo_url;
  document.getElementById("lab-run-name").textContent = run.photo_name;
  const status = { running: "Running", paused: "Paused", done: "Finished", error: "Stopped by an error" }[run.status] || run.status;
  document.getElementById("lab-run-status").textContent = run.capsule_unavailable
    ? `${status} · capsule builder can't use this photo, outline only`
    : status;
  document.getElementById("lab-progress-bar").style.width = `${run.total ? (100 * run.done) / run.total : 0}%`;
  document.getElementById("lab-run-stats").textContent =
    `${fmt(run.done)} of ${fmt(run.total)} done · ${fmt(run.ok)} built · ${fmt(run.failed)} failed · ${fmt(run.rated)} rated`
    + (run.eta_hours != null && run.status === "running" ? ` · about ${run.eta_hours} h left` : "");
  const pause = document.getElementById("lab-pause");
  pause.hidden = run.status === "done";
  pause.textContent = run.status === "running" ? "Pause" : "Resume";
}

async function labLoadItems(reset) {
  if (!labState.runId) { labGrid.innerHTML = ""; labShowRun(null); return; }
  if (reset) { labState.items = []; labState.offset = 0; }
  const q = new URLSearchParams({ sort: labSort.value, family: labFamily.value, only: labOnly.value, offset: labState.offset, limit: 40 });
  const res = await fetch(`/api/lab/runs/${labState.runId}?${q}`);
  if (!res.ok) { labGrid.innerHTML = ""; return; }
  const data = await res.json();
  labShowRun(data.run);
  labState.count = data.count;
  labState.items = labState.items.concat(data.items);
  labState.offset = labState.items.length;
  labRenderGrid();
}

function labCard(item, i) {
  const stars = item.rating ? "★".repeat(item.rating.stars) : "";
  const pic = item.thumb_url ? `<img loading="lazy" src="${item.thumb_url}" alt="" />` : `<div class="lab-fail">failed</div>`;
  const label = item.template || (item.changed.length ? item.changed.slice(0, 2).join(" · ") + (item.changed.length > 2 ? ` +${item.changed.length - 2}` : "") : "current settings");
  const metric = item.status === "ok"
    ? `score ${item.score}${item.face_found ? ` · face ${item.face_err_pct}%` : ""}`
    : escapeHtml((item.error || "").slice(0, 60));
  return `<button type="button" class="lab-card${i === labState.current ? " current" : ""}" data-i="${i}">
    ${pic}
    <div class="lab-card-n">#${fmt(item.n)} <span>${escapeHtml(item.family.split(" ")[0])}</span>${stars ? `<span class="lab-card-stars">${stars}</span>` : ""}</div>
    <div class="lab-card-label">${escapeHtml(label)}</div>
    <div class="lab-card-metric">${metric}</div>
  </button>`;
}

function labRenderGrid() {
  labGrid.innerHTML = labState.items.length
    ? labState.items.map(labCard).join("")
    : `<div class="empty-result-hint">Nothing here yet — results appear as the server finishes them.</div>`;
  labMore.hidden = labState.items.length >= labState.count;
}

labGrid.addEventListener("click", (e) => {
  const card = e.target.closest(".lab-card");
  if (card) labOpen(Number(card.dataset.i));
});
labMore.addEventListener("click", () => labLoadItems(false));
[labSort, labFamily, labOnly].forEach((el) => el.addEventListener("change", () => labLoadItems(true)));
labRunSelect.addEventListener("change", () => {
  labState.runId = labRunSelect.value;
  try { localStorage.setItem(LAB_RUN_KEY, labState.runId); } catch (e) { /* ignore */ }
  labLoadItems(true);
});

function labPaintStars(n) {
  document.querySelectorAll("#lab-stars button").forEach((b) => b.classList.toggle("on", Number(b.dataset.star) <= n));
}

async function labOpen(i) {
  if (i >= labState.items.length && labState.items.length < labState.count) await labLoadItems(false);
  const item = labState.items[i];
  if (!item) return;
  labState.current = i;
  labBar.hidden = false;
  document.getElementById("lab-bar-title").textContent = `#${fmt(item.n)} of ${fmt(labState.space ? labState.space.total_builds : 0)} · ${item.family}`;
  document.getElementById("lab-bar-sub").textContent = item.status === "ok"
    ? `score ${item.score} · shape ${item.shape_iou}% · spikes ${item.spike_pct}% · wall ${item.min_wall_mm}mm${item.face_found === null ? "" : item.face_found ? ` · face ${item.face_err_pct}%` : " · no face found"}`
    : `failed: ${item.error || ""}`;
  document.getElementById("lab-info-list").innerHTML = (item.template ? `<li><b>${escapeHtml(item.template)}</b></li>` : "")
    + item.options.map((o) => `<li>${escapeHtml(o)}</li>`).join("");
  document.getElementById("lab-note").value = item.rating ? item.rating.note : "";
  labPaintStars(item.rating ? item.rating.stars : 0);
  statsBar.style.display = "none";
  downloadBar.innerHTML = item.status === "ok"
    ? `<a href="/api/lab/runs/${labState.runId}/stl/${item.code}" download>Download STL</a>`
    : "";
  setActiveTab("viewer");
  if (item.model_url) loadModel(item.model_url);
}

async function labRate(stars, note) {
  const item = labState.items[labState.current];
  if (!item) return;
  const res = await fetch(`/api/lab/runs/${labState.runId}/rate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: item.code, stars, note }),
  });
  if (res.ok) {
    item.rating = stars > 0 || note ? { stars, note } : null;
    labPaintStars(stars);
  }
}

document.getElementById("lab-stars").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-star]");
  if (!b) return;
  const item = labState.items[labState.current];
  const stars = Number(b.dataset.star);
  const same = item && item.rating && item.rating.stars === stars;
  labRate(same ? 0 : stars, document.getElementById("lab-note").value);
});
document.getElementById("lab-note-save").addEventListener("click", () => {
  const item = labState.items[labState.current];
  labRate(item && item.rating ? item.rating.stars : 0, document.getElementById("lab-note").value);
});
document.getElementById("lab-info-btn").addEventListener("click", () => {
  const info = document.getElementById("lab-info");
  info.hidden = !info.hidden;
});
document.getElementById("lab-prev").addEventListener("click", () => labState.current > 0 && labOpen(labState.current - 1));
document.getElementById("lab-next").addEventListener("click", () => labOpen(labState.current + 1));

document.getElementById("lab-start").addEventListener("click", async () => {
  const hint = document.getElementById("lab-start-hint");
  if (!selectedFiles.length) { hint.textContent = "Pick a photo in the Photos tab first."; setActiveTab("photos"); return; }
  const form = new FormData();
  form.append("image", selectedFiles[0]);
  hint.textContent = "Starting…";
  const res = await fetch("/api/lab/runs", { method: "POST", body: form });
  if (!res.ok) { hint.textContent = "Couldn't start the run — try again."; return; }
  labState.runId = (await res.json()).run_id;
  try { localStorage.setItem(LAB_RUN_KEY, labState.runId); } catch (e) { /* ignore */ }
  hint.textContent = "Running. The first results take about a minute (the photo is analysed once first).";
  await labRefreshRuns();
  labLoadItems(true);
});

document.getElementById("lab-pause").addEventListener("click", async (e) => {
  const action = e.target.textContent === "Pause" ? "pause" : "resume";
  const run = await (await fetch(`/api/lab/runs/${labState.runId}/${action}`, { method: "POST" })).json();
  labShowRun(run);
});

document.getElementById("lab-copy").addEventListener("click", async () => {
  const hint = document.getElementById("lab-copy-hint");
  const exportUrl = `${location.origin}/api/lab/runs/${labState.runId}/export`;
  const data = await (await fetch(exportUrl)).json();
  const lines = [
    `Figuro Lab results — run ${data.run.run_id} (${data.run.photo_name})`,
    `${data.run.done}/${data.run.total} combinations done, ${data.run.ok} built, ${data.ratings.length} rated`,
    `Full data: ${exportUrl}`,
    "",
    ...data.ratings.map((r) => `#${r.n} ${r.code} ${"★".repeat(r.stars)}${r.note ? ` — ${r.note}` : ""}`),
  ];
  const text = lines.join("\n");
  try {
    await navigator.clipboard.writeText(text);
    hint.textContent = "Copied — paste it to Claude.";
  } catch (err) {
    hint.textContent = `Couldn't copy automatically. Send Claude this link: ${exportUrl}`;
  }
});

async function labOnShow() {
  try {
    await labLoadSpace();
    await labRefreshRuns();
    await labLoadItems(true);
  } catch (e) { /* offline: the tab just stays as it was */ }
  clearInterval(labState.timer);
  labState.timer = setInterval(async () => {
    if (!document.getElementById("tab-lab").classList.contains("active") || document.visibilityState !== "visible") return;
    if (!labState.runId) return;
    const res = await fetch(`/api/lab/runs/${labState.runId}?limit=1`);
    if (res.ok) labShowRun((await res.json()).run);
  }, 8000);
}

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
  // The loading spinner itself lives inside .stage (the Viewer tab's
  // content), so that's the tab that needs to be active to see it.
  setActiveTab("viewer");
  pollJob(jobId);
})();
