/* Aligned RGB-D scene inspection for prepared FLASH demos. Geometry is display-only. */
import * as THREE from '/static/vendor/three.module.js';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

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
        vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.22, depthTest: false,
    })));
    group.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        color: 0x34d399, size: 3, sizeAttenuation: false, transparent: true, opacity: 0.9, depthTest: false,
    })));
    return {group, geometry};
}

function makeObservedSurface(runs, depth, rgba, camera, options) {
    const {width, height, intrinsics: k} = camera;
    const pixelCount = width * height;
    const lookup = new Int32Array(pixelCount);
    lookup.fill(-1);
    const positions = [], colors = [], pixels = [], pointPositions = [];
    const fx = k[0][0], fy = k[1][1], cx = k[0][2], cy = k[1][2];
    for (const [start, count] of runs) {
        for (let pixel = start; pixel < start + count; pixel++) {
            if (pixel < 0 || pixel >= pixelCount) throw new Error('Object mask exceeds camera frame');
            const z = depth[pixel] / 1000;
            if (!z) continue;
            const u = pixel % width, v = Math.floor(pixel / width);
            lookup[pixel] = pixels.length;
            pixels.push(pixel);
            const x = (u - cx) * z / fx, y = -(v - cy) * z / fy;
            positions.push(x, y, -z);
            colors.push(rgba[pixel * 4] / 255, rgba[pixel * 4 + 1] / 255,
                rgba[pixel * 4 + 2] / 255);
            if (u % options.pointSpacingPx === 0 && v % options.pointSpacingPx === 0)
                pointPositions.push(x, y, -z);
        }
    }
    const faces = [];
    const add = (a, b, c) => {
        if (a < 0 || b < 0 || c < 0) return;
        const za = -positions[a * 3 + 2], zb = -positions[b * 3 + 2], zc = -positions[c * 3 + 2];
        const footprint = ((za + zb + zc) / 3) / Math.min(fx, fy);
        if (Math.max(za, zb, zc) - Math.min(za, zb, zc) <= 3 * footprint)
            faces.push(a, b, c);
    };
    for (const pixel of pixels) {
        const u = pixel % width, v = Math.floor(pixel / width), a = lookup[pixel];
        if (u >= width - 1 || v >= height - 1) continue;
        const b = lookup[pixel + 1], c = lookup[pixel + width], d = lookup[pixel + width + 1];
        add(a, c, b); add(b, c, d);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(faces);
    const group = new THREE.Group();
    const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        vertexColors: true, side: THREE.DoubleSide, depthTest: false,
    }));
    mesh.visible = options.showSurface !== false;
    group.add(mesh);
    const pointGeometry = new THREE.BufferGeometry();
    pointGeometry.setAttribute('position', new THREE.Float32BufferAttribute(pointPositions, 3));
    const points = new THREE.Points(pointGeometry, new THREE.PointsMaterial({
        color: 0x35e3ed, size: 4, sizeAttenuation: false,
        transparent: true, opacity: 0.88, depthTest: false,
    }));
    points.visible = !!options.showPoints;
    group.add(points);
    return group;
}

function makeGroupSupport(pointsBaseM, baseFromCamera, verified, measured) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(pointsBaseM.flat(), 3));
    const group = new THREE.Group();
    group.add(new THREE.Points(geometry, new THREE.PointsMaterial({
        color: verified ? groupColor : visionColor, size: 4, sizeAttenuation: false,
        transparent: true, opacity: measured ? 0.9 : 0.3, depthTest: false,
    })));
    setPose(group, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], baseFromCamera);
    return group;
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
    // Coarse single-view depth has holes and stretched silhouettes when orbited.
    // Keep it as faint spatial context; sharp geometry belongs to measured masks.
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.18,
        depthWrite: false,
    }));
}

function makePoseAxes(length) {
    const group = new THREE.Group();
    group.add(new THREE.AxesHelper(length));
    return group;
}

function makeTurnPreview(rule, pose, baseFromCamera, radius) {
    const pivot = new THREE.Vector3(...rule.pivot_reference_m);
    const axis = new THREE.Vector3(...rule.axis_reference).normalize();
    const group = new THREE.Group();
    const marker = new THREE.Mesh(
        new THREE.SphereGeometry(Math.max(0.003, radius * 0.11)),
        new THREE.MeshBasicMaterial({color: 0xfbbf24, depthTest: false}),
    );
    marker.position.copy(pivot);
    group.add(marker);
    group.add(new THREE.ArrowHelper(axis, pivot, radius * 1.2, 0xfbbf24, radius * 0.3, radius * 0.16));
    const tangent = new THREE.Vector3(1, 0, 0).addScaledVector(axis, -axis.x);
    if (tangent.lengthSq() < 0.01) tangent.set(0, 1, 0).addScaledVector(axis, -axis.y);
    tangent.normalize();
    const bitangent = new THREE.Vector3().crossVectors(axis, tangent).normalize();
    const vertices = [];
    const span = rule.max_angle_rad - rule.min_angle_rad;
    const steps = Math.max(12, Math.ceil(64 * span / (2 * Math.PI)));
    for (let step = 0; step <= steps; step++) {
        const angle = rule.min_angle_rad + span * step / steps;
        const point = pivot.clone().addScaledVector(tangent, radius * Math.cos(angle))
            .addScaledVector(bitangent, radius * Math.sin(angle));
        vertices.push(point.x, point.y, point.z);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    group.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({color: 0xfbbf24, depthTest: false})));
    setPose(group, pose, baseFromCamera);
    group.userData = {kind: 'turn_rule', role: rule.reference_role, enabled: true};
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
    scene.background = null;
    const camera = new THREE.PerspectiveCamera(43, 1, 0.01, 4);
    camera.position.set(0, 0, 0);
    camera.lookAt(0, 0, -0.45);
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 0, -0.45);
    controls.update();
    let context = null;
    let cameraAligned = true;
    let frameCamera = null;
    let latestSnapshot = null;
    let turnPreview = null;
    let turnGroup = null;
    let observedCenters = {};
    let focusedRole = null;
    let lastFocusCenter = null;
    const raycaster = new THREE.Raycaster();
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
            const scale = Math.min(width / frameCamera.width, height / frameCamera.height);
            const insetX = (width - frameCamera.width * scale) / 2;
            const insetY = (height - frameCamera.height * scale) / 2;
            const near = camera.near;
            // Match CSS object-fit: contain and the recorded principal point.
            // A centered-FOV camera shifts overlays whenever cx/cy are off-center.
            const left = -near * (k[0][2] + insetX / scale) / k[0][0];
            const right = near * ((width - insetX) / scale - k[0][2]) / k[0][0];
            const top = near * (k[1][2] + insetY / scale) / k[1][1];
            const bottom = near * (k[1][2] - (height - insetY) / scale) / k[1][1];
            camera.projectionMatrix.makePerspective(left, right, top, bottom, near, camera.far);
            camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
            camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan((top - bottom) / (2 * near)));
        } else {
            camera.updateProjectionMatrix();
        }
        renderer.setClearColor(0x172232, cameraAligned ? 0 : 1);
        renderer.render(scene, camera);
    }

    function setCameraAligned(value) {
        cameraAligned = value;
        canvas.parentElement.classList.toggle('flash-scene-orbit', !value);
    }

    function reset() {
        focusedRole = null;
        lastFocusCenter = null;
        controls.target.set(0, 0, -0.45);
        camera.position.set(0, 0, 0);
        controls.update();
        setCameraAligned(true);
        draw();
    }

    function update(depth, rgba, snapshot, options = {}, frameEvidence = null) {
        frameCamera = snapshot.camera;
        latestSnapshot = snapshot;
        if (context) {
            scene.remove(context);
            context.geometry.dispose();
            context.material.dispose();
            context = null;
        }
        if (options.showDepthScene) {
            context = makeContext(depth, rgba, snapshot.camera);
            scene.add(context);
        }
        while (evidence.children.length) {
            const item = evidence.children[0];
            evidence.remove(item);
            disposeEvidence(item, references);
        }
        turnGroup = null;
        const baseFromCamera = snapshot.camera.base_from_camera;
        const held = snapshot.held_prediction;
        observedCenters = {};
        for (const [role, ref] of Object.entries(references)) {
            const used = held?.role === role ? held : snapshot.accepted?.[role];
            const observedRuns = frameEvidence?.masks_rle?.[role];
            if ((options.showSurface !== false || options.showPoints) && observedRuns?.length) {
                const observed = makeObservedSurface(observedRuns, depth, rgba, snapshot.camera, options);
                if (observed.children[0].geometry.attributes.position.count) {
                    observed.children[0].geometry.computeBoundingSphere();
                    observedCenters[role] = observed.children[0].geometry.boundingSphere.center.clone();
                }
                observed.userData = {kind: 'observed', role, enabled: true};
                observed.children[0].userData.role = role;
                evidence.add(observed);
            }
            if ((options.showProjection || !frameEvidence?.available) && used?.pose) {
                const surface = ref.group.clone();
                // A clone shares the buffers, but owns its material for this frame.
                surface.children.forEach(child => { child.material = child.material.clone(); });
                if (held?.role === role) surface.children[1].material.color.setHex(gripperColor);
                setPose(surface, used.pose, baseFromCamera);
                surface.userData = {kind: held?.role === role ? 'held' : 'accepted', role, enabled: true};
                evidence.add(surface);
            }
            const coords = snapshot.centers_object_m?.[role];
            if (coords && used?.pose && options.showAxes !== false) {
                const axes = makePoseAxes(0.025);
                axes.children[0].position.fromArray(coords);
                setPose(axes, used.pose, baseFromCamera);
                axes.userData = {kind: held?.role === role ? 'held' : 'accepted', role, enabled: true};
                evidence.add(axes);
            }
            const support = frameEvidence?.groups?.[role];
            if (support?.base_xyz_m?.length && options.showGroup) {
                const points = makeGroupSupport(support.base_xyz_m, baseFromCamera,
                    support.verified, support.measured_this_frame !== false);
                points.userData = {kind: 'layer', estimator: 'demo_extraction', role, enabled: true};
                evidence.add(points);
            }
            const extracted = snapshot.layers?.[role]?.demo_extraction;
            if (extracted?.pose) {
                const groupBased = extracted.estimator === 'environment_group_motion';
                const ghost = makeDashedReference(ref.geometry, groupBased ? groupColor : visionColor);
                setPose(ghost, extracted.pose, baseFromCamera);
                ghost.userData = {kind: 'layer', estimator: 'demo_extraction', role,
                    enabled: groupBased && !!options.showGroup && !support?.base_xyz_m?.length};
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
        refreshTurnPreview(false);
        draw();
    }

    function refreshTurnPreview(shouldDraw = true) {
        if (turnGroup) {
            evidence.remove(turnGroup);
            disposeEvidence(turnGroup, references);
            turnGroup = null;
        }
        const role = turnPreview?.reference_role;
        const used = role && latestSnapshot?.accepted?.[role];
        if (used?.pose && latestSnapshot?.camera?.base_from_camera && references[role]) {
            references[role].geometry.computeBoundingSphere();
            const radius = Math.max(0.02, references[role].geometry.boundingSphere.radius * 0.8);
            turnGroup = makeTurnPreview(
                turnPreview, used.pose, latestSnapshot.camera.base_from_camera, radius
            );
            evidence.add(turnGroup);
        }
        if (shouldDraw) draw();
    }

    function setTurnPreview(rule) {
        turnPreview = rule;
        refreshTurnPreview();
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
        if (!local || !pose?.pose || !latestSnapshot.camera?.base_from_camera)
            return observedCenters[role]?.clone() || null;
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
        setCameraAligned(false);
        draw();
        return true;
    }

    function pickObserved(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        const pointer = new THREE.Vector2(
            2 * (clientX - rect.left) / rect.width - 1,
            1 - 2 * (clientY - rect.top) / rect.height,
        );
        raycaster.setFromCamera(pointer, camera);
        const meshes = evidence.children.filter(item => item.userData.kind === 'observed')
            .map(item => item.children[0]);
        const hit = raycaster.intersectObjects(meshes, false)[0];
        if (!hit) return null;
        const point = hit.point;
        const role = hit.object.userData.role;
        const cameraXYZ = [point.x, -point.y, -point.z];
        const result = {role, camera_xyz_m: cameraXYZ};
        const used = latestSnapshot?.accepted?.[role];
        const baseFromCamera = latestSnapshot?.camera?.base_from_camera;
        if (used?.pose && baseFromCamera && hit.face) {
            const worldFromCamera = new THREE.Matrix4().set(...baseFromCamera.flat());
            const objectFromWorld = new THREE.Matrix4().set(...used.pose.flat()).invert();
            const objectFromCamera = objectFromWorld.multiply(worldFromCamera);
            const localPoint = new THREE.Vector3(...cameraXYZ).applyMatrix4(objectFromCamera);
            const renderNormal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld);
            const localNormal = new THREE.Vector3(
                renderNormal.x, -renderNormal.y, -renderNormal.z,
            ).transformDirection(objectFromCamera);
            result.reference_xyz_m = localPoint.toArray();
            result.reference_normal = localNormal.toArray();
            result.reference_pose_evidence = used.evidence;
        }
        return result;
    }

    function projectCameraXYZ(cameraXYZ) {
        const point = new THREE.Vector3(cameraXYZ[0], -cameraXYZ[1], -cameraXYZ[2]).project(camera);
        const rect = canvas.getBoundingClientRect();
        return [rect.left + (point.x + 1) * rect.width / 2,
            rect.top + (1 - point.y) * rect.height / 2];
    }

    controls.addEventListener('change', () => {
        // OrbitControls fires `start` for a plain click. Preserve the aligned
        // RGB backdrop until the camera actually moves away from calibration.
        if (cameraAligned && (camera.position.lengthSq() > 1e-10 ||
            controls.target.distanceToSquared(new THREE.Vector3(0, 0, -0.45)) > 1e-10))
            setCameraAligned(false);
        draw();
    });
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
    return {update, reset, focus, pickObserved, projectCameraXYZ, highlight, setTurnPreview, draw, dispose};
}
