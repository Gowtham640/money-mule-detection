"use strict";

/*
 * LiveGraph3D — incremental Three.js renderer for the transaction graph.
 *
 * It only draws what the app tells it (nodes = backend accounts, edges =
 * backend window edges, travellers = backend transactions). It owns no data.
 *
 * Cost model: nodes and glows are two InstancedMeshes (2 draw calls), all
 * edges share one LineSegments buffer updated in place, travellers are one
 * more InstancedMesh, and raycasting happens only after the pointer moves.
 * Rendering pauses while the canvas is scrolled out of view.
 */
(function () {
  const COLORS = {
    normal: new THREE.Color(0x2f7dff),
    idle: new THREE.Color(0x163a7a),
    early: new THREE.Color(0xffc34d),
    fraud: new THREE.Color(0xff3d63),
    banned: new THREE.Color(0x3d4b47),
    fresh: new THREE.Color(0x22ff99),
    edge: new THREE.Color(0x2bd9b0),
    edgeAttack: new THREE.Color(0xff3d63),
    edgeEarly: new THREE.Color(0xffb020),
  };
  const SCALE = { normal: 1.0, idle: 0.72, early: 1.55, fraud: 2.0, banned: 0.55, fresh: 1.45 };

  function hashUnit(seed) {
    let hash = 2166136261;
    const source = String(seed);
    for (let i = 0; i < source.length; i += 1) {
      hash ^= source.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    // murmur3 finaliser: FNV alone barely changes the high bits for ids that
    // differ only in their last characters, which collapsed the layout.
    hash ^= hash >>> 16;
    hash = Math.imul(hash, 0x85ebca6b);
    hash ^= hash >>> 13;
    hash = Math.imul(hash, 0xc2b2ae35);
    hash ^= hash >>> 16;
    return (hash >>> 0) / 4294967295;
  }

  class LiveGraph3D {
    constructor(container, options = {}) {
      this.container = container;
      this.layout = options.layout || "sphere";
      this.nodeRadius = options.nodeRadius || 4.6;
      this.handlers = {};
      this.ids = [];
      this.index = new Map();
      this.state = [];
      this.capacity = 0;
      this.edges = new Map();
      this.edgeCapacity = 0;
      this.edgeColorsDirty = true;
      this.travellers = [];
      this.maxTravellers = options.maxTravellers || 80;
      this.mouse = new THREE.Vector2(2, 2);
      this.pointerMoved = false;
      this.hoverIndex = -1;
      this.visible = true;
      this.downAt = null;
      this._matrix = new THREE.Matrix4();
      this._color = new THREE.Color();

      this.scene = new THREE.Scene();
      this.scene.fog = new THREE.FogExp2(0x06150f, options.fog ?? 0.00018);
      this.camera = new THREE.PerspectiveCamera(60, 1, 1, 20000);
      const distance = options.cameraDistance || 1750;
      this.camera.position.set(0, 220, distance);
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.setClearColor(0x000000, 0);
      container.innerHTML = "";
      container.appendChild(this.renderer.domElement);
      this.renderer.domElement.style.cursor = "grab";
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.06;
      this.controls.minDistance = 80;
      this.controls.maxDistance = 9000;
      this.controls.autoRotate = Boolean(options.autoRotate);
      this.controls.autoRotateSpeed = 0.25;
      this.root = new THREE.Group();
      this.scene.add(this.root);
      this.raycaster = new THREE.Raycaster();

      if (options.background !== false) this._initBackground();
      this._initNodes(256);
      this._initEdges(512);
      this._initTravellers();
      this._bindEvents();
      this.resize();
      this._loop = this._loop.bind(this);
      requestAnimationFrame(this._loop);
    }

    // ------------------------------------------------------------ setup
    _initBackground() {
      const n = 1600;
      const positions = new Float32Array(n * 3);
      const colors = new Float32Array(n * 3);
      for (let i = 0; i < n; i += 1) {
        const radius = 1600 + hashUnit(`r${i}`) * 3400;
        const theta = hashUnit(`t${i}`) * Math.PI * 2;
        const phi = Math.acos(hashUnit(`p${i}`) * 2 - 1);
        positions[i * 3] = Math.sin(phi) * Math.cos(theta) * radius;
        positions[i * 3 + 1] = Math.cos(phi) * radius * 0.68;
        positions[i * 3 + 2] = Math.sin(phi) * Math.sin(theta) * radius;
        const pick = hashUnit(`c${i}`);
        const c = pick < 0.5 ? [0, 0.8, 0.8] : pick < 0.82 ? [0, 0.28, 1] : [1, 0, 0.66];
        colors.set(c, i * 3);
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      this.scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        size: 1.1, vertexColors: true, transparent: true, opacity: 0.26, depthWrite: false,
      })));
      const grid = new THREE.GridHelper(4400, 70, 0x0f5a3d, 0x0a3525);
      grid.position.y = -760;
      grid.material.opacity = 0.18;
      grid.material.transparent = true;
      this.scene.add(grid);
    }

    _initNodes(capacity) {
      if (this.coreMesh) {
        this.root.remove(this.coreMesh);
        this.root.remove(this.glowMesh);
        this.coreMesh.dispose?.();
        this.glowMesh.dispose?.();
      }
      const previous = this.capacity;
      this.capacity = capacity;
      const coreGeometry = new THREE.SphereGeometry(this.nodeRadius, 14, 10);
      const glowGeometry = new THREE.SphereGeometry(this.nodeRadius * 2.1, 12, 8);
      this.coreMesh = new THREE.InstancedMesh(coreGeometry, new THREE.MeshBasicMaterial({ color: 0xffffff }), capacity);
      this.glowMesh = new THREE.InstancedMesh(glowGeometry, new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.2, depthWrite: false, blending: THREE.AdditiveBlending,
      }), capacity);
      for (const mesh of [this.coreMesh, this.glowMesh]) {
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        // instanceColor must exist before the first render or the shader is
        // compiled without per-instance colour.
        for (let i = 0; i < capacity; i += 1) mesh.setColorAt(i, COLORS.idle);
        mesh.count = this.ids.length;
        mesh.frustumCulled = false;
        this.root.add(mesh);
      }
      if (!this.base || previous < capacity) {
        const grow = (old) => {
          const next = new Float32Array(capacity * 3);
          if (old) next.set(old);
          return next;
        };
        this.base = grow(this.base);
        this.current = grow(this.current);
      }
      for (let i = 0; i < this.ids.length; i += 1) this._paintNode(i);
    }

    _initEdges(capacity) {
      if (this.edgeLines) {
        this.root.remove(this.edgeLines);
        this.edgeLines.geometry.dispose();
      }
      this.edgeCapacity = capacity;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(THREE.DynamicDrawUsage));
      geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(THREE.DynamicDrawUsage));
      geometry.setDrawRange(0, 0);
      this.edgeLines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending,
      }));
      this.edgeLines.frustumCulled = false;
      this.root.add(this.edgeLines);
      this.edgeColorsDirty = true;
    }

    _initTravellers() {
      const geometry = new THREE.SphereGeometry(2.2, 8, 6);
      this.travellerMesh = new THREE.InstancedMesh(geometry, new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending,
      }), this.maxTravellers);
      this.travellerMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.travellerMesh.count = 0;
      this.travellerMesh.frustumCulled = false;
      for (let i = 0; i < this.maxTravellers; i += 1) this.travellerMesh.setColorAt(i, COLORS.edge);
      this.root.add(this.travellerMesh);
    }

    _bindEvents() {
      const el = this.renderer.domElement;
      el.addEventListener("pointermove", (event) => {
        const rect = el.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
        this.pointerClient = { x: event.clientX - rect.left, y: event.clientY - rect.top };
        this.pointerMoved = true;
      }, { passive: true });
      el.addEventListener("pointerleave", () => {
        this.mouse.set(2, 2);
        this.pointerMoved = true;
      }, { passive: true });
      el.addEventListener("pointerdown", (event) => {
        this.downAt = { x: event.clientX, y: event.clientY };
        el.style.cursor = "grabbing";
      }, { passive: true });
      el.addEventListener("pointerup", (event) => {
        el.style.cursor = this.hoverIndex >= 0 ? "pointer" : "grab";
        if (!this.downAt) return;
        const moved = Math.hypot(event.clientX - this.downAt.x, event.clientY - this.downAt.y);
        this.downAt = null;
        if (moved < 5 && this.hoverIndex >= 0) this._emit("click", this.ids[this.hoverIndex]);
      }, { passive: true });
      if (window.ResizeObserver) {
        new ResizeObserver(() => this.resize()).observe(this.container);
      } else {
        window.addEventListener("resize", () => this.resize(), { passive: true });
      }
      if (window.IntersectionObserver) {
        new IntersectionObserver((entries) => {
          this.visible = entries.some((entry) => entry.isIntersecting);
        }).observe(this.container);
      }
    }

    on(name, handler) {
      (this.handlers[name] = this.handlers[name] || []).push(handler);
    }

    _emit(name, payload) {
      (this.handlers[name] || []).forEach((handler) => handler(payload));
    }

    resize() {
      const width = Math.max(this.container.clientWidth, 1);
      const height = Math.max(this.container.clientHeight, 1);
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    }

    // ------------------------------------------------------------- nodes
    _spherePosition(id) {
      const theta = hashUnit(`${id}|t`) * Math.PI * 2;
      const phi = Math.acos(hashUnit(`${id}|p`) * 2 - 1);
      const radius = 170 + 760 * Math.cbrt(hashUnit(`${id}|r`));
      return [
        Math.sin(phi) * Math.cos(theta) * radius,
        Math.cos(phi) * radius * 0.78,
        Math.sin(phi) * Math.sin(theta) * radius,
      ];
    }

    hasNode(id) {
      return this.index.has(String(id));
    }

    upsertNode(id, props = {}) {
      const key = String(id);
      let i = this.index.get(key);
      if (i === undefined) {
        i = this.ids.length;
        if (i >= this.capacity) this._initNodes(this.capacity * 2);
        this.ids.push(key);
        this.index.set(key, i);
        this.state.push({ status: "normal", active: false, freshUntil: 0, phase: hashUnit(`${key}|ph`) * Math.PI * 2 });
        const p = this.layout === "sphere" ? this._spherePosition(key) : [0, 0, 0];
        this.base.set(p, i * 3);
        this.current.set(p, i * 3);
        this.coreMesh.count = this.ids.length;
        this.glowMesh.count = this.ids.length;
        if (this.layout === "cluster") this._clusterLayout();
      }
      const s = this.state[i];
      let changed = false;
      if (props.status !== undefined && props.status !== s.status) {
        s.status = props.status;
        changed = true;
      }
      if (props.active !== undefined && props.active !== s.active) {
        s.active = props.active;
        changed = true;
      }
      if (props.fresh) {
        s.freshUntil = performance.now() + 20000;
        changed = true;
      }
      if (props.label !== undefined) s.label = props.label;
      if (changed || props.force) {
        this._paintNode(i);
        this.edgeColorsDirty = true;
      }
    }

    _visualKey(s) {
      if (s.status === "fraud" || s.status === "early" || s.status === "banned") return s.status;
      if (s.freshUntil > performance.now()) return "fresh";
      return s.active ? "normal" : "idle";
    }

    _paintNode(i) {
      const key = this._visualKey(this.state[i]);
      this.state[i].paintedKey = key;
      this.coreMesh.setColorAt(i, COLORS[key]);
      this.glowMesh.setColorAt(i, COLORS[key]);
      if (this.coreMesh.instanceColor) this.coreMesh.instanceColor.needsUpdate = true;
      if (this.glowMesh.instanceColor) this.glowMesh.instanceColor.needsUpdate = true;
    }

    clear() {
      this.ids = [];
      this.index.clear();
      this.state = [];
      this.edges.clear();
      this.travellers = [];
      this.coreMesh.count = 0;
      this.glowMesh.count = 0;
      this.edgeLines.geometry.setDrawRange(0, 0);
      this.travellerMesh.count = 0;
    }

    _clusterLayout() {
      // Hub (highest degree) in the centre when the pattern has one,
      // everyone else on a ring in order of appearance.
      const degree = new Map(this.ids.map((id) => [id, 0]));
      this.edges.forEach((edge) => {
        degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
        degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
      });
      const ranked = [...degree.entries()].sort((a, b) => b[1] - a[1]);
      const hub = ranked.length > 3 && ranked[0][1] >= 3 && ranked[0][1] > (ranked[1]?.[1] || 0) ? ranked[0][0] : null;
      const ring = this.ids.filter((id) => id !== hub);
      const radius = Math.max(120, ring.length * 26);
      ring.forEach((id, n) => {
        const angle = (n / Math.max(ring.length, 1)) * Math.PI * 2 - Math.PI / 2;
        this.base.set([Math.cos(angle) * radius, Math.sin(angle) * radius, 0], this.index.get(id) * 3);
      });
      if (hub) this.base.set([0, 0, 0], this.index.get(hub) * 3);
    }

    // ------------------------------------------------------------- edges
    upsertEdge(edge) {
      const existing = this.edges.get(edge.id);
      if (existing) {
        existing.count = edge.count;
        existing.attack = edge.attack_count > 0;
      } else {
        if (!this.index.has(edge.source) || !this.index.has(edge.target)) return;
        this.edges.set(edge.id, { source: edge.source, target: edge.target, count: edge.count, attack: edge.attack_count > 0 });
        if (this.layout === "cluster") this._clusterLayout();
      }
      this.edgeColorsDirty = true;
    }

    removeEdge(id) {
      if (this.edges.delete(id)) this.edgeColorsDirty = true;
    }

    clearEdges() {
      this.edges.clear();
      this.edgeColorsDirty = true;
    }

    _edgeColor(edge, out) {
      const a = this.state[this.index.get(edge.source)]?.status;
      const b = this.state[this.index.get(edge.target)]?.status;
      if (edge.attack || a === "fraud" || b === "fraud") return out.copy(COLORS.edgeAttack).multiplyScalar(0.95);
      if (a === "early" || b === "early") return out.copy(COLORS.edgeEarly).multiplyScalar(0.6);
      return out.copy(COLORS.edge).multiplyScalar(Math.min(0.22 + edge.count * 0.08, 0.5));
    }

    _writeEdges() {
      if (this.edges.size > this.edgeCapacity) this._initEdges(Math.max(this.edgeCapacity * 2, this.edges.size));
      const positions = this.edgeLines.geometry.attributes.position.array;
      const colors = this.edgeLines.geometry.attributes.color.array;
      let n = 0;
      this.edges.forEach((edge) => {
        const i = this.index.get(edge.source);
        const j = this.index.get(edge.target);
        if (i === undefined || j === undefined) return;
        positions.set(this.current.subarray(i * 3, i * 3 + 3), n * 6);
        positions.set(this.current.subarray(j * 3, j * 3 + 3), n * 6 + 3);
        if (this.edgeColorsDirty) {
          this._edgeColor(edge, this._color);
          colors[n * 6] = colors[n * 6 + 3] = this._color.r;
          colors[n * 6 + 1] = colors[n * 6 + 4] = this._color.g;
          colors[n * 6 + 2] = colors[n * 6 + 5] = this._color.b;
        }
        n += 1;
      });
      this.edgeLines.geometry.setDrawRange(0, n * 2);
      this.edgeLines.geometry.attributes.position.needsUpdate = true;
      if (this.edgeColorsDirty) this.edgeLines.geometry.attributes.color.needsUpdate = true;
      this.edgeColorsDirty = false;
    }

    // -------------------------------------------------------- travellers
    pulse(source, target, kind = "normal", duration = 900) {
      const i = this.index.get(String(source));
      const j = this.index.get(String(target));
      if (i === undefined || j === undefined) return;
      if (this.travellers.length >= this.maxTravellers) this.travellers.shift();
      const color = kind === "attack" ? COLORS.fraud : kind === "risky" ? COLORS.early : COLORS.edge;
      this.travellers.push({ i, j, started: performance.now(), duration, color });
    }

    // ------------------------------------------------------------ camera
    focus(ids, durationMs = 900) {
      const points = (ids || [])
        .map((id) => this.index.get(String(id)))
        .filter((i) => i !== undefined)
        .map((i) => new THREE.Vector3().fromArray(this.base, i * 3));
      if (!points.length) return;
      const box = new THREE.Box3().setFromPoints(points);
      const center = box.getCenter(new THREE.Vector3());
      const size = Math.max(box.getSize(new THREE.Vector3()).length(), 160);
      const distance = THREE.MathUtils.clamp(size * 1.6, 260, 2600);
      const dir = this.camera.position.clone().sub(this.controls.target).normalize();
      const startTarget = this.controls.target.clone();
      const startCamera = this.camera.position.clone();
      const endCamera = center.clone().add(dir.multiplyScalar(distance));
      const started = performance.now();
      const step = () => {
        const t = Math.min((performance.now() - started) / durationMs, 1);
        const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
        this.controls.target.lerpVectors(startTarget, center, e);
        this.camera.position.lerpVectors(startCamera, endCamera, e);
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    reset(durationMs = 900) {
      this.focus(this.ids, durationMs);
    }

    // -------------------------------------------------------------- loop
    _loop(now) {
      requestAnimationFrame(this._loop);
      if (!this.visible || document.hidden) return;
      this.controls.update();
      const t = now * 0.001;
      const lerp = this.layout === "cluster" ? 0.12 : 1;
      for (let i = 0; i < this.ids.length; i += 1) {
        const s = this.state[i];
        const o = i * 3;
        const drift = this.layout === "sphere" ? 7 : 3;
        const tx = this.base[o] + Math.sin(t * 0.6 + s.phase) * drift;
        const ty = this.base[o + 1] + Math.cos(t * 0.55 + s.phase) * drift;
        const tz = this.base[o + 2] + Math.sin(t * 0.5 + s.phase * 1.3) * drift;
        this.current[o] += (tx - this.current[o]) * lerp;
        this.current[o + 1] += (ty - this.current[o + 1]) * lerp;
        this.current[o + 2] += (tz - this.current[o + 2]) * lerp;
        const key = this._visualKey(s);
        let scale = SCALE[key] || 1;
        if (key === "fraud") scale *= 1 + Math.sin(t * 4 + s.phase) * 0.12;
        if (i === this.hoverIndex) scale *= 1.35;
        this._matrix.makeScale(scale, scale, scale);
        this._matrix.setPosition(this.current[o], this.current[o + 1], this.current[o + 2]);
        this.coreMesh.setMatrixAt(i, this._matrix);
        this.glowMesh.setMatrixAt(i, this._matrix);
        if (key !== s.paintedKey) this._paintNode(i);
      }
      this.coreMesh.instanceMatrix.needsUpdate = true;
      this.glowMesh.instanceMatrix.needsUpdate = true;
      this._writeEdges();

      let live = 0;
      const nowMs = performance.now();
      this.travellers = this.travellers.filter((tr) => nowMs - tr.started < tr.duration);
      for (const tr of this.travellers) {
        const p = (nowMs - tr.started) / tr.duration;
        const a = tr.i * 3;
        const b = tr.j * 3;
        this._matrix.makeScale(1, 1, 1);
        this._matrix.setPosition(
          this.current[a] + (this.current[b] - this.current[a]) * p,
          this.current[a + 1] + (this.current[b + 1] - this.current[a + 1]) * p,
          this.current[a + 2] + (this.current[b + 2] - this.current[a + 2]) * p,
        );
        this.travellerMesh.setMatrixAt(live, this._matrix);
        this.travellerMesh.setColorAt(live, tr.color);
        live += 1;
      }
      this.travellerMesh.count = live;
      this.travellerMesh.instanceMatrix.needsUpdate = true;
      if (this.travellerMesh.instanceColor) this.travellerMesh.instanceColor.needsUpdate = true;

      if (this.pointerMoved) {
        this.pointerMoved = false;
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const hit = this.mouse.x <= 1 ? this.raycaster.intersectObject(this.coreMesh)[0] : null;
        const next = hit ? hit.instanceId : -1;
        if (next !== this.hoverIndex) {
          this.hoverIndex = next;
          this.renderer.domElement.style.cursor = next >= 0 ? "pointer" : "grab";
        }
        this._emit("hover", next >= 0 ? { id: this.ids[next], ...this.pointerClient } : null);
      }
      this.renderer.render(this.scene, this.camera);
    }
  }

  window.LiveGraph3D = LiveGraph3D;
})();
