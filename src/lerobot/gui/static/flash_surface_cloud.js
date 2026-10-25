/* The FLASH surface view uses the same Three.js OrbitControls as the URDF view. */
import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

const COLORS = {observed: 0x42e4ec, reference: 0xffffff, axes: [0xff5555, 0x58e280, 0x699bff]};

function cameraPoint(point) {
    return new THREE.Vector3(point[0], -point[1], -point[2]);
}

function makePoints(points, color, size) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(points.length * 3);
    points.forEach((point, index) => cameraPoint(point).toArray(positions, index * 3));
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    return new THREE.Points(geometry, new THREE.PointsMaterial({color, size, sizeAttenuation: false}));
}

function makeLabel(letter) {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 48px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#0a1320';
    ctx.lineWidth = 7;
    ctx.strokeText(letter, 32, 34);
    ctx.fillText(letter, 32, 34);
    return new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(canvas), depthTest: false}));
}

function clearGroup(group) {
    for (const child of [...group.children]) {
        group.remove(child);
        child.geometry?.dispose();
        if (child.material?.map) child.material.map.dispose();
        child.material?.dispose();
    }
}

export function createFlashSurfaceCloud(canvas) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a1320);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.0001, 100);
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const controls = new OrbitControls(camera, renderer.domElement);
    const cloud = new THREE.Group();
    scene.add(cloud);
    let identity = null;
    let framing = null;
    let lastCenter = null;
    let lastFollow = false;
    const viewKey = () => identity ? `flashSurfaceView:${identity}` : null;

    function saveView() {
        if (!framing || !viewKey()) return;
        try {
            sessionStorage.setItem(viewKey(), JSON.stringify({
                p: camera.position.clone().sub(framing.center).toArray(),
                t: controls.target.clone().sub(framing.center).toArray(),
            }));
        } catch (_) { /* The viewer still works when storage is unavailable. */ }
    }

    function restoreView() {
        let saved;
        try { saved = JSON.parse(sessionStorage.getItem(viewKey()) || 'null'); }
        catch (_) { return; }
        if (!Array.isArray(saved?.p) || !Array.isArray(saved?.t)) return;
        camera.position.fromArray(saved.p).add(framing.center);
        controls.target.fromArray(saved.t).add(framing.center);
        controls.update();
    }

    function draw() {
        const width = canvas.clientWidth, height = canvas.clientHeight;
        if (!width || !height) return;
        if (canvas.width !== Math.round(width * renderer.getPixelRatio()) ||
            canvas.height !== Math.round(height * renderer.getPixelRatio())) {
            renderer.setSize(width, height, false);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        }
        renderer.render(scene, camera);
    }

    function reset(clearSaved = true) {
        if (!framing) return;
        if (clearSaved && viewKey()) {
            try { sessionStorage.removeItem(viewKey()); } catch (_) { /* Storage is optional. */ }
        }
        const {center, radius} = framing;
        controls.target.copy(center);
        // Fit the reference surface's bounding sphere in the vertical field of view.
        const distance = Math.max(radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)), 0.03);
        camera.position.copy(center).add(new THREE.Vector3(0.45, 0.35, 1).normalize().multiplyScalar(distance));
        camera.near = Math.max(distance / 1000, 0.0001);
        camera.far = Math.max(distance * 100, 10);
        camera.updateProjectionMatrix();
        controls.update();
        draw();
    }

    function update(estimate, options) {
        clearGroup(cloud);
        if (!estimate?.reference_m?.length || !estimate.camera_from_reference) {
            draw();
            return;
        }
        const pose = estimate.camera_from_reference;
        const local = point => [0, 1, 2].map(i =>
            [0, 1, 2].reduce((sum, j) => sum + pose[j][i] * (point[j] - pose[j][3]), 0));
        const inView = point => options.follow ? point : local(point);
        const centerLocal = options.centerLocal;
        const center = cameraPoint(options.follow ? [0, 1, 2].map(i =>
            pose[i][3] + [0, 1, 2].reduce((sum, j) => sum + pose[i][j] * centerLocal[j], 0)) : centerLocal);
        const newIdentity = identity !== options.identity;
        framing = {center, radius: options.radius};
        if (newIdentity) {
            identity = options.identity;
            reset(false);
            restoreView();
        } else if (lastCenter && (options.follow || lastFollow)) {
            const delta = center.clone().sub(lastCenter);
            camera.position.add(delta);
            controls.target.add(delta);
            controls.update();
        }
        lastCenter = center.clone();
        lastFollow = options.follow;

        if (options.showObserved && estimate.observed_m?.length)
            cloud.add(makePoints(estimate.observed_m.map(inView), COLORS.observed, 3));
        if (options.showReference)
            cloud.add(makePoints(estimate.reference_m.map(inView), COLORS.reference, 2));
        if (options.showAxes && estimate.anchors_m?.length && estimate.axes_m?.length) {
            const anchor = cameraPoint(inView(estimate.anchors_m[0]));
            estimate.axes_m.slice(0, 3).forEach((end, i) => {
                const geometry = new THREE.BufferGeometry().setFromPoints([anchor, cameraPoint(inView(end))]);
                cloud.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({color: COLORS.axes[i]})));
            });
            estimate.anchors_m.slice(0, 3).forEach((point, i) => {
                const marker = makePoints([inView(point)], COLORS.axes[i], 8);
                cloud.add(marker);
                const label = makeLabel('ABC'[i]);
                label.position.copy(cameraPoint(inView(point)));
                label.scale.setScalar(Math.max(options.radius * 0.22, 0.004));
                cloud.add(label);
            });
        }
        draw();
    }

    controls.addEventListener('change', draw);
    controls.addEventListener('end', saveView);
    new ResizeObserver(draw).observe(canvas);
    canvas.addEventListener('webglcontextlost', event => event.preventDefault());
    return {update, reset, camera, controls, get identity() { return identity; }};
}
