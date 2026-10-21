import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// geometry: layers run left to right along x. Each layer is the residual stream passing through a token-mixer slab
// (linear or full attention) and then the MLP, drawn as a vertical sheet of its neurons.
const SHEET = 4.2, GAP = 1.9, SHEET_Y0 = 0.7, ATTN_DX = -0.62, MLP_DX = 0.18;
const PAL = {
  bg: 0x000000,
  idle: new THREE.Color(0.036, 0.046, 0.052), swept: new THREE.Color(0.2, 0.24, 0.26),
  lo: new THREE.Color(0.42, 0.30, 0.10), hi: new THREE.Color(1.0, 0.71, 0.28), named: new THREE.Color(1, 1, 1),
  link: new THREE.Color(0.22, 0.78, 1.0),
  wire: 0x1c2a30, wireData: 0x3a5a66, slabEdge: 0x3c4a52, slabEdgeFull: 0x6a8592, head: 0x2f4a56, headFull: 0x5f8fa3,
};

function glowPoints(sizePx, maxScale = 3) {
  return new THREE.RawShaderMaterial({
    uniforms: { size: { value: sizePx * Math.min(devicePixelRatio, 2) }, ref: { value: 1 }, maxScale: { value: maxScale },
      focusX: { value: 0 }, fadeW: { value: 1 }, fadeAmt: { value: 0 }, camSide: { value: 1 } },
    vertexShader: `precision highp float;
      uniform mat4 modelViewMatrix, projectionMatrix; uniform float size, ref, maxScale, focusX, fadeW, fadeAmt, camSide;
      attribute vec3 position; attribute vec3 color; varying vec3 vC;
      void main() { float d = (position.x - focusX) / fadeW;
        vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * mv;
        // up close nothing overlaps, so each point brightens with proximity; neighbours never fade below a quarter
        float near = clamp(sqrt(ref / -mv.z), 1.0, 3.5);
        // layers between the camera and the focus go fully transparent as you zoom in (x-ray), the rest keep a floor
        float front = step(0.45, (position.x - focusX) * camSide / ${GAP.toFixed(3)});
        float keep = mix(mix(1.0, exp(-d * d), fadeAmt * 0.75), 1.0 - fadeAmt, front);
        vC = color * keep * near;
        gl_PointSize = size * clamp(sqrt(ref / -mv.z), 0.6, maxScale); }`,
    fragmentShader: `precision highp float; varying vec3 vC;
      void main() { float r = length(gl_PointCoord - 0.5); if (r > 0.5) discard;
        float edge = 1.0 - smoothstep(0.38, 0.5, r); gl_FragColor = vec4(vC * edge, 1.0); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

export function createMap(canvas, labelsDiv, arch, cb) {
  const L = arch.n_layers, N = arch.inter;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(PAL.bg);
  const scene = new THREE.Scene();
  const layerX = l => l * GAP;
  const X0 = ATTN_DX - 1.6, X1 = (L - 1) * GAP + MLP_DX + 1.4;

  const insets = { left: 330, right: 0 };
  let fitPerPx = null;
  const camera = new THREE.PerspectiveCamera(32, 1, 0.02, 2000);
  let homeDist = 60;
  const home = { target: new THREE.Vector3((L - 1) * GAP / 2, SHEET_Y0 + SHEET / 2 - 0.4, 0), dir: new THREE.Vector3(0.62, 0.4, 1).normalize() };
  camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(home.target);
  controls.enableDamping = false;
  controls.zoomToCursor = true;
  controls.minDistance = 1.6; controls.maxDistance = 400;
  controls.zoomSpeed = 0.6;
  controls.update();

  const gridSide = Math.ceil(Math.sqrt(N));
  let layoutCells = null;  // Int16Array [L * N] of grid cells, or null for neuron-index order
  const neuronPos = (layer, n) => {
    const c = layoutCells ? layoutCells[layer * N + n] : n;
    const i = c % gridSide, j = Math.floor(c / gridSide);
    return new THREE.Vector3(layerX(layer) + MLP_DX, SHEET_Y0 + (j / (gridSide - 1)) * SHEET, (i / (gridSide - 1) - 0.5) * SHEET);
  };

  // every neuron of every layer, one draw call
  const pos = new Float32Array(L * N * 3), col = new Float32Array(L * N * 3);
  for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
    const p = neuronPos(l, n), k = (l * N + n) * 3;
    pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
    PAL.idle.toArray(col, k);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  const pointMat = glowPoints(2.0, 5);
  const points = new THREE.Points(geo, pointMat);
  scene.add(points);

  // structure
  const wires = [];
  const seg = (a, b) => wires.push(a.x, a.y, a.z, b.x, b.y, b.z);
  const V = (x, y, z) => new THREE.Vector3(x, y, z);
  const sheetCorners = x => { const h = SHEET / 2; return [V(x, SHEET_Y0, -h), V(x, SHEET_Y0, h), V(x, SHEET_Y0 + SHEET, h), V(x, SHEET_Y0 + SHEET, -h)]; };
  seg(V(X0, 0, 0), V(X1, 0, 0));
  const slabs = [], headPos = { full: [], lin: [] };
  for (let l = 0; l < L; l++) {
    const x = layerX(l), mx = x + MLP_DX, ax = x + ATTN_DX, cs = sheetCorners(mx);
    for (let i = 0; i < 4; i++) seg(cs[i], cs[(i + 1) % 4]);
    seg(V(ax, 0, 0), V(ax, SHEET_Y0, 0)); seg(V(mx, 0, 0), V(mx, SHEET_Y0, 0));
    const full = arch.layer_types[l] === "full_attention", t = full ? 0.3 : 0.1;
    const slab = new THREE.Mesh(new THREE.BoxGeometry(t, SHEET * 0.92, SHEET * 0.92), new THREE.MeshBasicMaterial({
      color: full ? 0x8fa6b2 : 0x6f8590, transparent: true, opacity: full ? 0.075 : 0.045, depthWrite: false }));
    slab.position.set(ax, SHEET_Y0 + SHEET / 2, 0);
    slab.userData = { layer: l, full };
    scene.add(slab); slabs.push(slab);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(slab.geometry),
      new THREE.LineBasicMaterial({ color: full ? PAL.slabEdgeFull : PAL.slabEdge, transparent: true }));
    edges.position.copy(slab.position); scene.add(edges);
    slab.userData.edges = edges; slab.userData.baseOpacity = slab.material.opacity;
    const heads = full ? arch.attn_heads : arch.linear_v_heads;
    for (let h = 0; h < heads; h++) {
      const z = ((h + 0.5) / heads - 0.5) * SHEET * 0.82;
      (full ? headPos.full : headPos.lin).push(ax + t / 2 + 0.003, SHEET_Y0 + 0.35, z, ax + t / 2 + 0.003, SHEET_Y0 + SHEET - 0.35, z);
    }
  }
  const fadeLines = [];  // whole-scene line work that dims as the camera closes in on one layer
  for (const [key, color] of [["lin", PAL.head], ["full", PAL.headFull]]) {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(headPos[key], 3));
    const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }); m.userData.base = 0.55;
    scene.add(new THREE.LineSegments(g, m)); fadeLines.push(m);
  }
  const endSlabs = [];
  for (const x of [X0, X1]) {
    const m = new THREE.LineBasicMaterial({ color: PAL.slabEdgeFull, transparent: true }); m.userData.base = 1;
    const slab = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(0.14, 1.4, 2.2)), m);
    slab.position.set(x, 0, 0); scene.add(slab); endSlabs.push(m);
  }
  const wg = new THREE.BufferGeometry(); wg.setAttribute("position", new THREE.Float32BufferAttribute(wires, 3));
  const wireMat = new THREE.LineBasicMaterial({ color: PAL.wire, transparent: true }); wireMat.userData.base = 1;
  scene.add(new THREE.LineSegments(wg, wireMat)); fadeLines.push(wireMat, ...endSlabs);
  const dataOutline = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: PAL.wireData }));
  dataOutline.frustumCulled = false;
  scene.add(dataOutline);
  const bounds = [];
  for (const x of [X0, X1]) for (const y of [-0.8, SHEET_Y0 + SHEET]) for (const z of [-SHEET / 2, SHEET / 2]) bounds.push(V(x, y, z));

  // highlights and connections (their geometry changes every update, so culling is off)
  const hiGeo = new THREE.BufferGeometry();
  const hiMat = glowPoints(5, 2);
  const hiPts = new THREE.Points(hiGeo, hiMat);
  hiPts.frustumCulled = false; hiPts.renderOrder = 3;
  scene.add(hiPts);
  const conGeo = new THREE.BufferGeometry();
  const conLines = new THREE.LineSegments(conGeo, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  conLines.frustumCulled = false; conLines.renderOrder = 2;
  scene.add(conLines);
  let selected = null, hovered = null, connections = [], hoverCon = null;
  const markGeo = new THREE.BufferGeometry();
  const markPts = new THREE.Points(markGeo, glowPoints(4, 2));
  markPts.frustumCulled = false; markPts.renderOrder = 1;
  scene.add(markPts);

  function refreshHighlights() {
    const p = [], c = [];
    const add = (l, n, color) => { const v = neuronPos(l, n); p.push(v.x, v.y, v.z); c.push(color.r, color.g, color.b); };
    const src = hoverCon ? hoverCon.src : selected, shown = hoverCon ? hoverCon.list : connections;
    const zmax = Math.max(...shown.map(k => k.z), 1);
    for (const k of shown) add(k.layer, k.n, PAL.link.clone().multiplyScalar(0.35 + 0.65 * k.z / zmax));
    if (hovered) add(hovered.layer, hovered.n, PAL.named.clone().multiplyScalar(0.7));
    if (selected) add(selected.layer, selected.n, PAL.named);
    hiGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    hiGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
    const lp = [], lc = [];
    if (src) {
      const a = neuronPos(src.layer, src.n);
      for (const k of shown) {
        const b = neuronPos(k.layer, k.n), s = 0.2 + 0.8 * k.z / zmax, cl = PAL.link.clone().multiplyScalar(s);
        lp.push(a.x, a.y, a.z, b.x, b.y, b.z); lc.push(cl.r, cl.g, cl.b, cl.r, cl.g, cl.b);
      }
    }
    conGeo.setAttribute("position", new THREE.Float32BufferAttribute(lp, 3));
    conGeo.setAttribute("color", new THREE.Float32BufferAttribute(lc, 3));
    render();
  }

  // layer labels, projected each frame
  const labels = [];
  for (let l = 0; l < L; l++) {
    const s = document.createElement("span"); s.textContent = `L${l}`; labelsDiv.appendChild(s); labels.push(s);
    s.onclick = () => enterLayer(l);
  }
  const dataLayers = new Set();
  function placeLabels() {
    const w = canvas.clientWidth, h = canvas.clientHeight, v = new THREE.Vector3();
    for (let l = 0; l < L; l++) {
      v.set(layerX(l), -0.55, SHEET / 2).project(camera);
      const s = labels[l];
      s.style.left = `${(v.x + 1) / 2 * w}px`; s.style.top = `${(1 - v.y) / 2 * h}px`;
      s.className = dataLayers.has(l) ? "data" : "";
    }
  }

  let pending = false;
  function render() {
    if (pending) return; pending = true;
    requestAnimationFrame(() => {
      pending = false;
      for (const m of [pointMat, hiMat, markPts.material]) m.uniforms.ref.value = homeDist;
      applyFade();
      renderer.render(scene, camera); placeLabels();
    });
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    const narrow = w <= 760, li = narrow ? 0 : insets.left, ri = narrow ? 0 : insets.right;
    const availW = Math.max(200, w - li - ri);
    camera.aspect = w / h;
    if (fitPerPx === null) {
      // distance at which the model's bounding sphere fits the visible gap, then place the camera there once
      const center = home.target, r = Math.max(...bounds.map(b => b.distanceTo(center)));
      const vHalf = THREE.MathUtils.degToRad(camera.fov / 2), hHalf = Math.atan(Math.tan(vHalf) * availW / h);
      homeDist = r / Math.sin(Math.min(vHalf, hHalf)) * 0.82;
      camera.position.copy(home.target).addScaledVector(home.dir, homeDist);
      controls.update();
      fitPerPx = homeDist;
    }
    // the orbit target sits centered in the visible gap between list and panel
    camera.setViewOffset(w, h, (ri - li) / 2, 0, w, h);
    camera.updateProjectionMatrix(); render();
  }
  new ResizeObserver(resize).observe(canvas);
  controls.addEventListener("change", render);

  // picking
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.05;
  const mouse = new THREE.Vector2();
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    mouse.set((ev.clientX - r.left) / r.width * 2 - 1, -(ev.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(mouse, camera);
    // hits come sorted by depth; on a sheet seen at an angle several neighbours fall within the threshold, so take
    // the frontmost layer's hits and pick the point closest to the ray
    const hits = ray.intersectObject(points);
    if (hits.length) {
      // the frontmost layer that is actually visible at the current fade, then the point closest to the ray in it
      const layers = [...new Set(hits.map(h => Math.floor(h.index / N)))];
      const front = layers.find(l => layerVisibility(l) > 0.5) ?? layers.reduce((a, b) => layerVisibility(b) > layerVisibility(a) ? b : a);
      let best = null;
      for (const h of hits) if (Math.floor(h.index / N) === front && (!best || h.distanceToRay < best.distanceToRay)) best = h;
      return { type: "neuron", layer: front, n: best.index % N };
    }
    const b = ray.intersectObjects(slabs)[0];
    if (b) return { type: "block", ...b.object.userData };
    return null;
  }
  let moveQueued = null, downAt = null;
  canvas.addEventListener("pointermove", ev => {
    if (moveQueued) { moveQueued = ev; return; }
    moveQueued = ev;
    requestAnimationFrame(() => {
      const e = moveQueued; moveQueued = null;
      const p = pick(e);
      const h = p && p.type === "neuron" ? { layer: p.layer, n: p.n } : null;
      if (JSON.stringify(h) !== JSON.stringify(hovered)) { hovered = h; refreshHighlights(); }
      cb.onHover(p, e.clientX, e.clientY);
    });
  });
  canvas.addEventListener("pointerleave", () => { hovered = null; refreshHighlights(); cb.onHover(null); });
  canvas.addEventListener("pointerdown", ev => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener("pointerup", ev => {
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4) return;
    const p = pick(ev);
    if (p && p.type === "neuron") cb.onPick(p.layer, p.n);
    else if (p && p.type === "block" && layerMode === null) enterLayer(p.layer);
  });

  // the closer the camera is to the orbit target, the more the layers away from the target's x fade out
  const fade = { fx: 0, fw: 1, amt: 0, side: 1 };
  const layerVisibility = l => {
    const dx = layerX(l) + MLP_DX - fade.fx, d = dx / fade.fw;
    if (dx * fade.side / GAP > 0.45) return 1 - fade.amt;
    return 1 - 0.75 * fade.amt * (1 - Math.exp(-d * d));
  };
  let layerMode = null, saved = null;
  function enterLayer(l) {
    if (layerMode === null) saved = { pos: camera.position.clone(), tgt: controls.target.clone() };
    layerMode = l;
    const c = new THREE.Vector3(layerX(l) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12;
    controls.target.copy(c); camera.position.copy(c).add(new THREE.Vector3(-dist, 0, 0));
    controls.enableRotate = false;
    controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(l);
  }
  function exitLayer() {
    if (layerMode === null) return;
    layerMode = null;
    controls.enableRotate = true;
    controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
    camera.position.copy(saved.pos); controls.target.copy(saved.tgt);
    controls.update(); render(); cb.onLayerMode && cb.onLayerMode(null);
  }

  function applyFade() {
    if (layerMode !== null) {
      const fx = layerX(layerMode) + MLP_DX;
      pointMat.uniforms.focusX.value = fx; pointMat.uniforms.fadeW.value = 0.05; pointMat.uniforms.fadeAmt.value = 1;
      fade.fx = fx; fade.fw = 0.05; fade.amt = 1;
      for (const slab of slabs) { slab.material.opacity = 0; slab.userData.edges.material.opacity = 0; }
      for (let l = 0; l < L; l++) labels[l].style.opacity = 0;
      for (const m of fadeLines) m.opacity = 0;
      dataOutline.visible = false;
      return;
    }
    dataOutline.visible = true;
    // continuous in zoom: eased on log-distance from the overview (no fade) to a sheet filling the view (full fade)
    const dist = camera.position.distanceTo(controls.target);
    const far = Math.log(homeDist * 0.7), near = Math.log(homeDist * 0.07);
    const amt = THREE.MathUtils.smoothstep(THREE.MathUtils.clamp((far - Math.log(dist)) / (far - near), 0, 1), 0, 1);
    const fx = controls.target.x, fw = GAP * (0.85 + 2.5 * (1 - amt));
    const side = Math.sign(camera.position.x - fx) || 1;
    pointMat.uniforms.focusX.value = fx; pointMat.uniforms.fadeW.value = fw; pointMat.uniforms.fadeAmt.value = amt;
    pointMat.uniforms.camSide.value = side;
    fade.fx = fx; fade.fw = fw; fade.amt = amt; fade.side = side;
    for (const slab of slabs) {
      const d = (slab.position.x - fx) / fw, f = 1 - amt * (1 - Math.exp(-d * d));
      slab.material.opacity = slab.userData.baseOpacity * f; slab.userData.edges.material.opacity = f;
    }
    for (let l = 0; l < L; l++) {
      const d = (layerX(l) + MLP_DX - fx) / fw;
      labels[l].style.opacity = 1 - amt * (1 - Math.exp(-d * d));
    }
    for (const m of fadeLines) m.opacity = m.userData.base * (1 - 0.9 * amt);
  }

  function flyTo(target, dist) {
    const from = controls.target.clone(), to = target.clone();
    const dir = camera.position.clone().sub(controls.target), d0 = dir.length();
    dir.normalize();
    const t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 450), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, to, e);
      camera.position.copy(controls.target).addScaledVector(dir, d0 + (dist - d0) * e);
      controls.update(); render();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }
  const curDist = () => camera.position.distanceTo(controls.target);

  // face a layer's neuron sheet head-on, close enough that the neighbours fade away
  function viewLayer(layer) {
    const c = new THREE.Vector3(layerX(layer) + MLP_DX, SHEET_Y0 + SHEET / 2, 0);
    const dist = (SHEET / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
    const from = controls.target.clone(), p0 = camera.position.clone();
    const p1 = c.clone().add(new THREE.Vector3(dist, 0.0001, 0)), t0 = performance.now();
    (function step(t) {
      const k = Math.min(1, (t - t0) / 600), e = k * k * (3 - 2 * k);
      controls.target.lerpVectors(from, c, e); camera.position.lerpVectors(p0, p1, e);
      controls.update(); render();
      if (k < 1) requestAnimationFrame(step);
    })(t0);
  }

  return {
    neuronPos,
    setUnits(layer, units) {
      dataLayers.add(layer);
      const zs = units.filter(u => u.kind === "neuron").map(u => u.z);
      const rnd = units.filter(u => u.kind === "random").map(u => u.z);
      const floor = rnd.length ? Math.max(...rnd) : Math.min(...zs), top = Math.max(...zs, floor + 1e-6);
      for (const u of units) {
        if (u.kind !== "neuron") continue;
        let c = PAL.swept;
        if (u.name) c = PAL.named;
        else if (u.z > floor) c = PAL.lo.clone().lerp(PAL.hi, Math.min(1, (u.z - floor) / (top - floor)));
        c.toArray(col, (layer * N + u.n) * 3);
      }
      geo.attributes.color.needsUpdate = true;
      const dl = [];
      for (const l of dataLayers) {
        const cs = sheetCorners(layerX(l) + MLP_DX);
        for (let i = 0; i < 4; i++) dl.push(...cs[i].toArray(), ...cs[(i + 1) % 4].toArray());
      }
      dataOutline.geometry.setAttribute("position", new THREE.Float32BufferAttribute(dl, 3));
      render();
    },
    // fly: bring the neuron into view without ever zooming out from where the user is
    select(layer, n, fly = true) {
      selected = layer == null ? null : { layer, n };
      connections = [];
      refreshHighlights();

    },
    setConnections(con) {
      connections = [...con.down, ...con.up];
      refreshHighlights();
    },
    setHoverConnections(src, con) {
      hoverCon = src ? { src, list: [...con.down, ...con.up] } : null;
      refreshHighlights();
    },
    focusLayer() {},
    setLayout(cells) {
      layoutCells = cells;
      for (let l = 0; l < L; l++) for (let n = 0; n < N; n++) {
        const p = neuronPos(l, n), k = (l * N + n) * 3;
        pos[k] = p.x; pos[k + 1] = p.y; pos[k + 2] = p.z;
      }
      geo.attributes.position.needsUpdate = true;
      geo.computeBoundingSphere();
      refreshHighlights();
    },
    enterLayer, exitLayer,
    get layerMode() { return layerMode; },
    home() { flyTo(home.target, homeDist); },
    setMarks(marks) {
      const p = [], c = [], up = new THREE.Color(1.0, 0.71, 0.28), down = new THREE.Color(1.0, 0.31, 0.6);
      for (const m of marks) {
        const v = neuronPos(m.layer, m.n), k = (m.sign === "-" ? down : up).clone().multiplyScalar(m.w ?? 1);
        p.push(v.x, v.y, v.z); c.push(k.r, k.g, k.b);
      }
      markGeo.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
      markGeo.setAttribute("color", new THREE.Float32BufferAttribute(c, 3));
      render();
    },
    setInsets(left, right) { insets.left = left; insets.right = right; resize(); },
    hover(layer, n) { hovered = layer == null ? null : { layer, n }; refreshHighlights(); },
    debug() { return { cam: camera.position.toArray().map(v => +v.toFixed(2)), tgt: controls.target.toArray().map(v => +v.toFixed(2)), selected, hovered, n: connections.length, hover: hoverCon ? hoverCon.list.length : 0, dist: curDist(), homeDist, w: canvas.clientWidth, h: canvas.clientHeight, insets }; },
    render,
  };
}
