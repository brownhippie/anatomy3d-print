import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const fileList = document.getElementById("file-list");
const heightInput = document.getElementById("height-mm");
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

let selectedFiles = [];

function updateModeBanner() {
  if (selectedFiles.length === 0) {
    modeBanner.textContent = "Add a photo to begin.";
  } else if (selectedFiles.length === 1) {
    modeBanner.textContent =
      "Single-photo mode: a stylized geometric guess, with real face detail (nose/chin/eyes) when a face is detected.";
  } else {
    modeBanner.textContent = `Multi-photo mode (${selectedFiles.length} photos): visual-hull reconstruction from your photos' actual silhouettes — shoot them in turntable order, front first.`;
  }
}

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
        child.material = new THREE.MeshStandardMaterial({
          color: 0x9fb3d9,
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
  clearStatus();
  messagesEl.innerHTML = "";
  generateBtn.disabled = true;
  loadingOverlay.classList.add("show");
  loadingMsg.textContent =
    selectedFiles.length === 1
      ? "Detecting pose and face, building the figure…"
      : "Extracting silhouettes and carving the visual hull…";
  downloadBar.innerHTML = "";
  statsBar.style.display = "none";

  const form = new FormData();
  selectedFiles.forEach((f) => form.append("images", f));
  form.append("height_mm", heightInput.value || "150");
  form.append("use_depth", useDepthCheckbox.checked ? "true" : "false");

  try {
    const res = await fetch("/api/generate", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      setStatus("error", data.detail || "Something went wrong.");
      renderMessages(data.messages);
      return;
    }
    setStatus("ok", "Model generated.");
    renderMessages(data.messages);
    loadModel(data.obj_url);

    statsBar.style.display = "flex";
    statsBar.innerHTML = `
      <span><b>${data.stats.faces.toLocaleString()}</b> faces</span>
      <span><b>${data.stats.height_mm.toFixed(1)}mm</b> tall</span>
      <span><b>${data.stats.watertight ? "watertight" : "needs repair"}</b></span>
    `;
    downloadBar.innerHTML = `
      <a href="${data.stl_url}" download>Download STL</a>
      <a class="secondary" href="${data.obj_url}" download>Download OBJ</a>
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
  renderFileList();
  clearStatus();
  messagesEl.innerHTML = "";
  statsBar.style.display = "none";
  downloadBar.innerHTML = "";
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }
  emptyState.style.display = "flex";
});

initViewer();
renderFileList();
