/* Shared RGB-D rendering, following PR #266: pixel grid + GPU unprojection.
 * Camera coordinates are x right, y down, z forward; Three's view is y up/-z.
 * Missing depth stays missing. This renderer neither repairs nor estimates shape.
 */
import * as THREE from '/static/vendor/three.module.js';
import {OrbitControls} from '/static/vendor/OrbitControls.js';

export function calibratedProjection(camera, frame, width, height) {
    const k = frame.intrinsics;
    const scale = Math.min(width / frame.width, height / frame.height);
    const x = (width - frame.width * scale) / 2, y = (height - frame.height * scale) / 2;
    const n = camera.near;
    const left = -n * (k[0][2] + x / scale) / k[0][0];
    const right = n * ((width - x) / scale - k[0][2]) / k[0][0];
    const top = n * (k[1][2] + y / scale) / k[1][1];
    const bottom = n * (k[1][2] - (height - y) / scale) / k[1][1];
    camera.aspect = width / height;
    camera.projectionMatrix.makePerspective(left, right, top, bottom, n, camera.far);
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan((top - bottom) / (2 * n)));
}

export function sceneControls(camera, canvas) {
    // Same bindings as the URDF viewer: left rotate, right pan, wheel zoom.
    const controls = new OrbitControls(camera, canvas);
    controls.enableDamping = false; // Demand rendering; no hidden animation loop.
    return controls;
}

export function createRgbdCloud({stride = 1, opacity = 1, brightness = 1} = {}) {
    const geometry = new THREE.BufferGeometry();
    const uniforms = {
        depthTex: {value: null}, rgbTex: {value: null}, size: {value: new THREE.Vector2()},
        K: {value: new THREE.Vector4()}, depthScale: {value: 1},
        pointSize: {value: 2}, opacity: {value: opacity}, brightness: {value: brightness},
    };
    const material = new THREE.ShaderMaterial({
        uniforms, transparent: opacity < 1, depthWrite: opacity === 1,
        vertexShader: `uniform sampler2D depthTex;
            uniform vec2 size; uniform vec4 K; uniform float depthScale; uniform float pointSize;
            varying vec2 vUv; varying float valid;
            void main() {
                vec2 pix = position.xy;
                vUv = (pix + 0.5) / size;
                float z = texture2D(depthTex, vUv).r * depthScale;
                valid = z > 0.0 ? 1.0 : 0.0;
                vec3 p = vec3((pix.x-K.z)*z/K.x, -(pix.y-K.w)*z/K.y, -z);
                gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
                gl_PointSize = pointSize;
            }`,
        fragmentShader: `uniform sampler2D rgbTex; uniform float opacity; uniform float brightness;
            varying vec2 vUv; varying float valid;
            void main() {
                if (valid < 0.5) discard;
                gl_FragColor = vec4(texture2D(rgbTex, vUv).rgb * brightness, opacity);
            }`,
    });
    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    let width = 0, height = 0;
    function update(depth, rgb, frame, scale = 1) {
        if (depth.length !== frame.width * frame.height)
            throw new Error('Depth dimensions differ from the calibrated image');
        if (rgb instanceof Uint8Array || rgb instanceof Uint8ClampedArray) {
            if (rgb.length !== frame.width * frame.height * 4) throw new Error('RGB and depth dimensions differ');
        } else if (rgb.width !== frame.width || rgb.height !== frame.height) {
            throw new Error('RGB and depth dimensions differ');
        }
        if (width !== frame.width || height !== frame.height) {
            width = frame.width; height = frame.height;
            const pixels = [];
            for (let v = 0; v < height; v += stride)
                for (let u = 0; u < width; u += stride) pixels.push(u, v, 0);
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(pixels, 3));
            uniforms.depthTex.value?.dispose();
            uniforms.depthTex.value = new THREE.DataTexture(new Float32Array(depth.length), width, height, THREE.RedFormat, THREE.FloatType);
            uniforms.depthTex.value.flipY = false;
        }
        uniforms.depthTex.value.image.data.set(depth);
        uniforms.depthTex.value.needsUpdate = true;
        const arrayImage = ArrayBuffer.isView(rgb);
        if (!uniforms.rgbTex.value || !!uniforms.rgbTex.value.isDataTexture !== arrayImage) {
            uniforms.rgbTex.value?.dispose();
            uniforms.rgbTex.value = arrayImage
                ? new THREE.DataTexture(rgb, width, height, THREE.RGBAFormat)
                : new THREE.Texture(rgb);
        } else uniforms.rgbTex.value.image = arrayImage ? {data: rgb, width, height} : rgb;
        uniforms.rgbTex.value.flipY = false;
        uniforms.rgbTex.value.minFilter = uniforms.rgbTex.value.magFilter = THREE.NearestFilter;
        uniforms.rgbTex.value.needsUpdate = true;
        uniforms.size.value.set(width, height);
        const k = frame.intrinsics;
        uniforms.K.value.set(k[0][0], k[1][1], k[0][2], k[1][2]);
        uniforms.depthScale.value = scale;
    }
    function dispose() {
        uniforms.depthTex.value?.dispose(); uniforms.rgbTex.value?.dispose();
        geometry.dispose(); material.dispose();
    }
    return {points, update, dispose, uniforms};
}

function disposeLayer(group) {
    group.traverse(item => { item.geometry?.dispose(); item.material?.dispose(); });
    group.clear();
}

export function createPoseAxes(points) {
    const [origin, ...ends] = points;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(ends.flatMap(end => [...origin, ...end]), 3));
    const colors = [0xff6969, 0x6fe08a, 0x75a8ff].map(value => new THREE.Color(value));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors.flatMap(color => [...color.toArray(), ...color.toArray()]), 3));
    const axes = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({vertexColors: true, depthTest: false}));
    axes.userData.axis = true;
    return axes;
}

export function createRgbdScene(canvas) {
    const renderer = new THREE.WebGLRenderer({canvas, antialias: true, preserveDrawingBuffer: true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x101923, 1);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100);
    const controls = sceneControls(camera, canvas);
    const cloud = createRgbdCloud();
    scene.add(cloud.points);
    const layers = new Map();
    let calibration = null, aligned = true, bounds = new THREE.Box3();
    let options = {scene: true, estimate: true, labelled: true, axes: true};
    function draw() {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w || !h) return;
        if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio()))
            renderer.setSize(w, h, false);
        if (calibration) calibratedProjection(camera, calibration, w, h);
        cloud.uniforms.pointSize.value = Math.max(1, Math.min(w / (calibration?.width || w), h / (calibration?.height || h))) * renderer.getPixelRatio() * 1.3;
        renderer.render(scene, camera);
    }
    function reset() {
        aligned = true;
        camera.position.set(0, 0, 0); camera.up.set(0, 1, 0);
        const center = bounds.isEmpty() ? new THREE.Vector3(0, 0, -1) : bounds.getCenter(new THREE.Vector3());
        controls.target.set(0, 0, Math.min(-0.01, center.z));
        controls.update(); draw();
    }
    function focus() {
        if (bounds.isEmpty()) return;
        aligned = false;
        const center = bounds.getCenter(new THREE.Vector3());
        const radius = Math.max(0.015, bounds.getSize(new THREE.Vector3()).length() / 2);
        let direction = camera.position.clone().sub(controls.target).normalize();
        if (!direction.lengthSq()) direction.set(0, 0, 1);
        controls.target.copy(center);
        camera.position.copy(center).addScaledVector(direction, radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25);
        controls.update(); draw();
    }
    function setOptions(next) {
        options = {...options, ...next};
        cloud.points.visible = options.scene;
        for (const [name, group] of layers) {
            group.visible = !!options[name];
            group.traverse(child => { if (child.userData.axis) child.visible = options.axes; });
        }
        draw();
    }
    function update(depth, rgb, frame, descriptions) {
        calibration = frame;
        cloud.update(depth, rgb, frame);
        for (const group of layers.values()) { scene.remove(group); disposeLayer(group); }
        layers.clear(); bounds = new THREE.Box3();
        const opticalToView = new THREE.Matrix4().makeScale(1, -1, -1);
        for (const description of descriptions) {
            if (!description.pose || !description.points.length) continue;
            const group = layers.get(description.layer) || new THREE.Group();
            if (!layers.has(description.layer)) { layers.set(description.layer, group); scene.add(group); }
            const object = new THREE.Group();
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(description.points.flat(), 3));
            const appearance = {color: description.color, transparent: true, opacity: 0.9,
                depthTest: false, depthWrite: false};
            let surface;
            if (description.links) {
                geometry.setIndex(description.links.flat());
                surface = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial(appearance));
            } else {
                surface = new THREE.Points(geometry, new THREE.PointsMaterial({...appearance,
                    size: description.pointSize ?? 2.0, sizeAttenuation: false}));
            }
            object.add(surface);
            geometry.computeBoundingSphere();
            if (description.axes !== false && description.axisPoints) {
                const axes = createPoseAxes(description.axisPoints);
                axes.visible = options.axes;
                object.add(axes);
            }
            object.matrix.copy(opticalToView).multiply(new THREE.Matrix4().set(...description.pose.flat()));
            object.matrixAutoUpdate = false;
            group.add(object); object.updateMatrixWorld(true);
            bounds.union(new THREE.Box3().setFromObject(object));
        }
        setOptions(options);
        if (aligned) reset(); else draw();
    }
    controls.addEventListener('start', () => { aligned = false; });
    controls.addEventListener('change', draw);
    const resize = new ResizeObserver(draw); resize.observe(canvas);
    reset();
    return {update, draw, reset, focus, setOptions,
        view: () => ({position: camera.position.toArray(), target: controls.target.toArray(), aligned}),
        dispose() {
            resize.disconnect(); controls.dispose(); cloud.dispose();
            for (const group of layers.values()) disposeLayer(group);
            renderer.dispose();
            // The player reuses this canvas for the next recording. Dispose
            // its resources, but keep the browser's canvas context usable.
        },
    };
}
