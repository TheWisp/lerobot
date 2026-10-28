/* Aligned RGB-D scene inspection for prepared FLASH demos. Geometry is display-only. */
import * as THREE from '/static/vendor/three.module.js';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const observedColor = 0xe2f3f5;
const groupColor = 0xe5b661;
const gripperColor = 0xf2a578;
const visionColor = 0xaadce9;

function setPose(group, baseFromObject, baseFromCamera) {
    const world = new THREE.Matrix4().set(...baseFromObject.flat());
    const camera = new THREE.Matrix4().set(...baseFromCamera.flat()).invert();
    const toView = new THREE.Matrix4().makeScale(1, -1, -1);
    group.matrix.copy(toView.multiply(camera).multiply(world));
    group.matrixAutoUpdate = false;
}

function makeReference(data) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.positions_object_m, 3));
    const colors = new Float32Array(data.colors_rgb.length * 3);
    data.colors_rgb.forEach((rgb, index) => rgb.forEach((channel, i) => { colors[index * 3 + i] = channel / 255; }));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(data.indices);
    const group = new THREE.Group();
    group.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.92, depthTest: false,
    })));
    group.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        color: observedColor, size: 2, sizeAttenuation: false, transparent: true, opacity: 0.55, depthTest: false,
    })));
    return {group, geometry};
}

function makeDashedReference(geometry, color) {
    const group = new THREE.Group();
    const edges = new THREE.WireframeGeometry(geometry);
    group.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
        color, transparent: true, opacity: 0.35, depthTest: false,
    })));
    return group;
}

function makeContext(depth, rgb, camera) {
    const {width, height, intrinsics: k} = camera;
    if (depth.length !== width * height || rgb.length !== width * height * 4)
        throw new Error('RGB and depth dimensions differ');
    const step = 4; // Only a display sampling budget; it does not affect tracking.
    const cols = Math.ceil(width / step), rows = Math.ceil(height / step);
    const positions = new Float32Array(cols * rows * 3), colors = new Float32Array(cols * rows * 3);
    const z = new Float32Array(cols * rows), triangles = [];
    const fx = k[0][0], fy = k[1][1], cx = k[0][2], cy = k[1][2];
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
        const u = col * step, v = row * step, id = row * cols + col, pixel = v * width + u;
        const depthM = depth[pixel] / 1000;
        z[id] = depthM;
        positions.set([(u - cx) * depthM / fx, -(v - cy) * depthM / fy, -depthM], id * 3);
        const gray = (0.299 * rgb[pixel * 4] + 0.587 * rgb[pixel * 4 + 1] +
            0.114 * rgb[pixel * 4 + 2]) / 255;
        // Keep observed geometry readable without competing with the tracked patch.
        for (let channel = 0; channel < 3; channel++)
            colors[id * 3 + channel] = 0.02 + 0.12 * gray;
    }
    const add = (a, b, c) => {
        const min = Math.min(z[a], z[b], z[c]), max = Math.max(z[a], z[b], z[c]);
        const footprint = step * (z[a] + z[b] + z[c]) / 3 / Math.min(fx, fy);
        if (min > 0 && max - min <= 3 * footprint) triangles.push(a, b, c);
    };
    for (let row = 0; row < rows - 1; row++) for (let col = 0; col < cols - 1; col++) {
        const a = row * cols + col, b = a + 1, c = a + cols, d = c + 1;
        add(a, c, b); add(b, c, d);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(triangles);
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide}));
}

function makePoseAxes(length) {
    const group = new THREE.Group();
    group.add(new THREE.AxesHelper(length));
    return group;
}

function disposeEvidence(item, references) {
    item.traverse(child => {
        // A cloned surface shares the saved reference buffer across frames.
        if (child.geometry && !Object.values(references).some(ref => ref.geometry === child.geometry))
            child.geometry.dispose();
        if (child.material) {
            for (const material of Array.isArray(child.material) ? child.material : [child.material])
                material.dispose();
        }
    });
}

export function createFlashSceneView(canvas, surfaces) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x172232);
    const camera = new THREE.PerspectiveCamera(43, 1, 0.01, 4);
    camera.position.set(0, 0, 0);
    camera.lookAt(0, 0, -0.45);
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 0, -0.45);
    controls.update();
    let context = null;
    let frameCamera = null;
    let latestSnapshot = null;
    let focusedRole = null;
    let lastFocusCenter = null;
    const evidence = new THREE.Group();
    scene.add(evidence);
    const references = Object.fromEntries(Object.entries(surfaces).map(([role, data]) => [role, makeReference(data)]));

    function draw() {
        const width = canvas.clientWidth, height = canvas.clientHeight;
        if (!width || !height) return;
        if (canvas.width !== Math.round(width * renderer.getPixelRatio()) ||
            canvas.height !== Math.round(height * renderer.getPixelRatio())) {
            renderer.setSize(width, height, false);
        }
        camera.aspect = width / height;
        if (frameCamera) {
            const k = frameCamera.intrinsics;
            const vertical = frameCamera.height / (2 * k[1][1]);
            const horizontal = frameCamera.width / (2 * k[0][0] * camera.aspect);
            // Fit the whole recorded camera view inside a differently shaped pane.
            camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.max(vertical, horizontal)));
        }
        camera.updateProjectionMatrix();
        renderer.render(scene, camera);
    }

    function reset() {
        focusedRole = null;
        lastFocusCenter = null;
        controls.target.set(0, 0, -0.45);
        camera.position.set(0, 0, 0);
        controls.update();
        draw();
    }

    function update(depth, rgba, snapshot, options = {}) {
        frameCamera = snapshot.camera;
        latestSnapshot = snapshot;
        if (context) {
            scene.remove(context);
            context.geometry.dispose();
            context.material.dispose();
        }
        context = makeContext(depth, rgba, snapshot.camera);
        scene.add(context);
        while (evidence.children.length) {
            const item = evidence.children[0];
            evidence.remove(item);
            disposeEvidence(item, references);
        }
        const baseFromCamera = snapshot.camera.base_from_camera;
        const held = snapshot.held_prediction;
        for (const [role, ref] of Object.entries(references)) {
            const used = held?.role === role ? held : snapshot.accepted?.[role];
            if (options.showSurface !== false && used?.pose) {
                const surface = ref.group.clone();
                // A clone shares the buffers, but owns its material for this frame.
                surface.children.forEach(child => { child.material = child.material.clone(); });
                setPose(surface, used.pose, baseFromCamera);
                surface.userData = {kind: held?.role === role ? 'held' : 'accepted', role, enabled: true};
                evidence.add(surface);
                const coords = snapshot.centers_object_m?.[role];
                if (coords && options.showAxes !== false) {
                    const axes = makePoseAxes(0.025);
                    axes.position.fromArray(coords);
                    surface.add(axes);
                }
            }
            const extracted = snapshot.layers?.[role]?.demo_extraction;
            if (extracted?.pose) {
                const groupBased = extracted.estimator === 'environment_group_motion';
                const ghost = makeDashedReference(ref.geometry, groupBased ? groupColor : visionColor);
                setPose(ghost, extracted.pose, baseFromCamera);
                ghost.userData = {kind: 'layer', estimator: 'demo_extraction', role,
                    enabled: groupBased && !!options.showGroup};
                ghost.visible = ghost.userData.enabled;
                evidence.add(ghost);
            }
            if (held?.role === role && held.pose) {
                const ghost = makeDashedReference(ref.geometry, gripperColor);
                setPose(ghost, held.pose, baseFromCamera);
                ghost.userData = {kind: 'held', role, enabled: !!options.showGripper};
                ghost.visible = ghost.userData.enabled;
                evidence.add(ghost);
            }
        }
        if (options.showGripper && snapshot.accepted?.__gripper__?.pose) {
            const tool = makePoseAxes(0.04);
            setPose(tool, snapshot.accepted.__gripper__.pose, baseFromCamera);
            tool.userData = {kind: 'accepted', role: '__gripper__', enabled: true};
            evidence.add(tool);
        }
        if (focusedRole) {
            const center = objectCenter(focusedRole);
            if (center && lastFocusCenter) {
                const delta = center.clone().sub(lastFocusCenter);
                camera.position.add(delta);
                controls.target.add(delta);
                controls.update();
            }
            lastFocusCenter = center;
        }
        draw();
    }

    function highlight(source) {
        const selected = ['layer', 'accepted', 'held'].includes(source?.kind) ? source : null;
        for (const item of evidence.children) {
            const match = selected && item.userData.kind === selected.kind &&
                item.userData.role === selected.role &&
                (selected.kind !== 'layer' || item.userData.estimator === selected.estimator);
            item.visible = !!item.userData.enabled || !!match;
            item.traverse(child => {
                if (!child.material) return;
                for (const material of Array.isArray(child.material) ? child.material : [child.material]) {
                    if (material.userData.baseOpacity === undefined) material.userData.baseOpacity = material.opacity;
                    material.transparent = true;
                    material.opacity = material.userData.baseOpacity * (selected && !match ? 0.14 : 1);
                }
            });
        }
        draw();
    }

    function objectCenter(role) {
        const local = latestSnapshot?.centers_object_m?.[role];
        const pose = latestSnapshot?.held_prediction?.role === role
            ? latestSnapshot.held_prediction : latestSnapshot?.accepted?.[role];
        if (!local || !pose?.pose || !latestSnapshot.camera?.base_from_camera) return null;
        const point = new THREE.Vector3(...local);
        const world = new THREE.Matrix4().set(...pose.pose.flat());
        const cameraFromWorld = new THREE.Matrix4().set(...latestSnapshot.camera.base_from_camera.flat()).invert();
        return point.applyMatrix4(cameraFromWorld.multiply(world)).multiply(new THREE.Vector3(1, -1, -1));
    }

    function focus(role) {
        const center = objectCenter(role);
        if (!center || !references[role]) return false;
        focusedRole = role;
        lastFocusCenter = center.clone();
        references[role].geometry.computeBoundingSphere();
        const radius = references[role].geometry.boundingSphere.radius;
        const distance = 2 * radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2));
        controls.target.copy(center);
        camera.position.copy(center).add(new THREE.Vector3(0.15, 0.12, 1).normalize().multiplyScalar(distance));
        controls.update();
        draw();
        return true;
    }

    controls.addEventListener('change', draw);
    const resize = new ResizeObserver(draw);
    resize.observe(canvas);
    canvas.addEventListener('webglcontextlost', event => event.preventDefault());
    function dispose() {
        resize.disconnect();
        controls.dispose();
        while (evidence.children.length) {
            const item = evidence.children[0];
            evidence.remove(item);
            disposeEvidence(item, references);
        }
        if (context) { context.geometry.dispose(); context.material.dispose(); }
        for (const ref of Object.values(references)) {
            ref.geometry.dispose();
            ref.group.children.forEach(child => child.material.dispose());
        }
        renderer.dispose();
    }
    return {update, reset, focus, highlight, draw, dispose};
}
