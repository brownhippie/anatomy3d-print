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
  updateModeBanner();
  if (methodSelect.value === "trellis") {
    loadTrellisStatus();
  } else {
    trellisStatusEl.style.display = "none";
  }
}
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

function frameCamera(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z) * 0.72;

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

  try {
    const res = await fetch("/api/generate", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      setStatus("error", data.detail || "Something went wrong.");
      renderMessages(data.messages);
      return;
    }
    if (data.method_used === "capsule (fallback)") {
      setStatus("ok", "Model generated — TRELLIS.2 was unavailable, so this used the built-in method instead.");
    } else {
      setStatus("ok", "Model generated.");
    }
    renderMessages(data.messages);
    loadModel(data.obj_url);
    currentGlbUrl = data.glb_url;
    if (method === "trellis") loadTrellisStatus();

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
  } catch (err) {
    setStatus("error", "Network error: " + err.message);
  } finally {
    generateBtn.disabled = false;
    loadingOverlay.classList.remove("show");
  }
}

generateBtn.addEventListener("click", generate);
resetBtn.addEventListener("click", () => {
  selectedFiles = [];
  descriptionInput.value = "";
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
