// The Run tab's live camera video: the page's side of the stream.
// Design: src/lerobot/gui/docs/live_camera_video.md
//
// One connection carries a video track per camera and one data channel with
// the cycle's state, action and robot pose. The page makes the offer and the
// server answers it, so everything the stream needs has to be in the offer:
// one receive-only stream per camera, H.264 first on each (our frames are
// H.264 and the answering side cannot change that), and the cycle channel.
// The answer names the cameras in the order of its streams, which is the
// only thing that says which tile is which.
//
// Age at the eye: a painted frame's timestamp is its capture time in the
// transport's clock plus a constant the library picks at random and never
// sends. The channel announces every cycle's capture time, so the constant
// that turns the most readings into times that were announced is the one;
// readings no constant explains are reported as an age we do not know.
(() => {
    const CLOCK = 90000;           // RTP's clock for video, ticks per second
    const WRAP = 2 ** 32;          // the timestamp is 32 bits
    const RECENT = 120;            // readings the age is taken over: a few seconds
    const ANNOUNCED = 600;         // capture times kept for the join: ~20 s at 30 fps
    const RELEARN_AFTER = 30;      // unexplained frames before the constant is searched again

    function rtpFromCapture(seconds) {
        return Math.round(seconds * CLOCK) % WRAP;
    }

    function createAgeTracker() {
        const announced = [];                 // capture times, newest last
        const announcedSet = new Set();       // their timestamps, for the join
        const perCamera = new Map();          // camera -> {origin, readings:[{wire, shownAt}]}

        function noteCapture(captureTs) {
            announced.push(captureTs);
            announcedSet.add(rtpFromCapture(captureTs));
            while (announced.length > ANNOUNCED) {
                announcedSet.delete(rtpFromCapture(announced.shift()));
            }
        }

        function learnOrigin(readings) {
            // Try each of the newest few readings against every announced
            // capture time and keep the constant that explains the most. A
            // wrong one explains almost none, because real capture times are
            // irregular at the clock's resolution — but a perfectly regular
            // cadence is explained equally well by a constant a whole period
            // out, and then there is no answer to give rather than one that
            // would read as an age.
            let best = null;
            let bestVotes = 0;
            let tied = false;
            const probes = readings.slice(-8);
            for (const probe of probes) {
                for (const stamp of announcedSet) {
                    const candidate = ((probe.wire - stamp) % WRAP + WRAP) % WRAP;
                    if (candidate === best) continue;
                    let votes = 0;
                    for (const r of readings) {
                        if (announcedSet.has(((r.wire - candidate) % WRAP + WRAP) % WRAP)) votes++;
                    }
                    if (votes > bestVotes) { best = candidate; bestVotes = votes; tied = false; }
                    else if (votes === bestVotes && candidate !== best) { tied = true; }
                }
            }
            if (tied) return null;
            return bestVotes >= Math.max(3, Math.floor(readings.length / 4)) ? best : null;
        }

        function explains(origin, wire) {
            return announcedSet.has(((wire - origin) % WRAP + WRAP) % WRAP);
        }

        function notePainted(camera, wireRtp, shownAtSeconds) {
            let entry = perCamera.get(camera);
            if (!entry) { entry = { origin: null, readings: [], misses: 0 }; perCamera.set(camera, entry); }
            entry.readings.push({ wire: wireRtp, shownAt: shownAtSeconds });
            if (entry.readings.length > RECENT) entry.readings.shift();
            // The constant holds for the life of a track, so the search runs
            // once. A re-offered track is a new constant, and it shows as
            // frames that the current one no longer explains: a run of those
            // is what asks for the search again, rather than a clock. The
            // check is one lookup; the search is every announced time
            // against every reading, and running it per frame would cost the
            // page more than the pictures do.
            if (entry.origin !== null) {
                entry.misses = explains(entry.origin, wireRtp) ? 0 : entry.misses + 1;
                if (entry.misses < RELEARN_AFTER) return;
            }
            const learned = learnOrigin(entry.readings);
            if (learned !== null) { entry.origin = learned; entry.misses = 0; }
        }

        function captureOf(wire, origin, near) {
            const span = WRAP / CLOCK;
            const base = near - (near % span);
            const ticks = ((wire - origin) % WRAP + WRAP) % WRAP;
            let best = null;
            for (const k of [-1, 0, 1]) {
                const t = base + ticks / CLOCK + k * span;
                if (best === null || Math.abs(t - near) < Math.abs(best - near)) best = t;
            }
            return best;
        }

        function medianAgeMs(camera) {
            const entry = perCamera.get(camera);
            if (!entry || entry.origin === null || !announced.length) return null;
            const near = announced[announced.length - 1];
            const ages = [];
            for (const r of entry.readings) {
                ages.push((r.shownAt - captureOf(r.wire, entry.origin, near)) * 1000);
            }
            if (!ages.length) return null;
            ages.sort((a, b) => a - b);
            return ages[Math.floor(ages.length / 2)];
        }

        return { noteCapture, notePainted, medianAgeMs };
    }

    function defaultVideoCodecs() {
        if (typeof RTCRtpSender === 'undefined' || !RTCRtpSender.getCapabilities) return [];
        const caps = RTCRtpSender.getCapabilities('video');
        return (caps && caps.codecs) || [];
    }

    function createClient(opts = {}) {
        const doFetch = opts.fetch || ((...a) => fetch(...a));
        const PeerConnection = opts.RTCPeerConnection
            || (typeof RTCPeerConnection !== 'undefined' ? RTCPeerConnection : null);
        const videoCodecs = opts.videoCodecs || defaultVideoCodecs;
        const onState = opts.onState || (() => {});
        const onTrack = opts.onTrack || (() => {});
        const onCycle = opts.onCycle || (() => {});
        const tracker = createAgeTracker();

        const client = {
            state: { name: 'idle', reason: '' },
            cameras: [],
            profile: null,
            // The newest track per camera, so a tile rebuilt while the
            // stream is up is given what is already arriving.
            tracks: {},
            open,
            attach,
            close,
            notePainted,
            ageMs: (camera) => tracker.medianAgeMs(camera),
        };
        let pc = null;
        let session = null;
        let transceivers = [];
        let closed = false;
        // One negotiation at a time. Opening early and attaching the cameras
        // are started from different places — the tab appearing, the run's
        // frames appearing — and a connection can only carry one offer at a
        // time: overlapping them leaves the second answering a description
        // the first has already replaced.
        let chain = Promise.resolve();

        function serialize(work) {
            const next = chain.then(work, work);
            chain = next.catch(() => {});
            return next;
        }

        function setState(name, reason = '') {
            // Opening and attaching both pass through connecting, and a
            // listener that redraws on every call would redraw for a state
            // that did not change.
            if (client.state.name === name && client.state.reason === reason) return;
            client.state = { name, reason };
            onState(client.state);
        }

        // Set while the bar's failure is the server's rather than the
        // connection's, so a recovery clears the one and not the other.
        let failedOnCameras = false;

        function noteFailing(failing) {
            const cameras = Object.keys(failing || {});
            if (!cameras.length) {
                if (failedOnCameras) { failedOnCameras = false; setState('connecting'); }
                return;
            }
            failedOnCameras = true;
            setState('failed', cameras.map((c) => `${c}: ${failing[c]}`).join('; '));
        }

        function notePainted(camera, wireRtp, shownAtSeconds) {
            tracker.notePainted(camera, wireRtp, shownAtSeconds);
            if (['connecting', 'idle'].includes(client.state.name)) setState('streaming');
        }

        function h264Codecs() {
            return videoCodecs().filter((c) => (c.mimeType || '').toLowerCase() === 'video/h264');
        }

        function addVideoStream(codecs) {
            const t = pc.addTransceiver('video', { direction: 'recvonly' });
            if (t.setCodecPreferences) t.setCodecPreferences(codecs);
            transceivers.push(t);
            return t;
        }

        async function negotiate(attaching) {
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            const response = await doFetch('/api/run/live-video/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                    session,
                    attach: attaching,
                }),
            });
            if (!response.ok) {
                let reason = `the server refused the offer (${response.status})`;
                try { reason = (await response.json()).detail || reason; } catch (_) { /* keep it */ }
                throw new Error(reason);
            }
            const answer = await response.json();
            session = answer.session || session;
            if (answer.cameras && answer.cameras.length) client.cameras = answer.cameras;
            if (answer.profile) client.profile = answer.profile;
            await pc.setRemoteDescription(answer);
            return answer;
        }

        // Open the connection before the run has frames, so the first
        // picture costs a keyframe and one way across the link rather than a
        // connection's whole setup. The offer carries one placeholder video
        // stream: every later stream shares the transport chosen for the
        // first one, so a connection opened with the channel alone could
        // only ever carry one camera.
        async function _open() {
            if (pc) return true;
            // WebRTC first: the codec list comes from an RTCRtpSender, so a
            // browser without WebRTC reports no H.264 either and would be
            // told the wrong thing about itself.
            if (!PeerConnection) {
                setState('failed', 'this browser has no WebRTC');
                return false;
            }
            const h264 = h264Codecs();
            if (!h264.length) {
                setState('failed', 'this browser cannot decode H.264, which is what this stream is');
                return false;
            }
            setState('connecting');
            // `closed` says this client was closed for good, not that a
            // connection once was: a new one clears it, or the first
            // failed open would silence every failure after it.
            closed = false;
            const connection = new PeerConnection({ iceServers: [] });
            pc = connection;
            const channel = pc.createDataChannel('cycles');
            channel.onmessage = (e) => {
                let message;
                try { message = JSON.parse(e.data); } catch (_) { return; }
                if (typeof message.capture_ts === 'number') tracker.noteCapture(message.capture_ts);
                noteFailing(message.failing);
                onCycle(message);
            };
            addVideoStream(h264);
            pc.ontrack = (e) => {
                const index = transceivers.indexOf(e.transceiver);
                const camera = client.cameras[index];
                if (!camera) return;
                client.tracks[camera] = e.track;
                onTrack(camera, e.track);
            };
            connection.onconnectionstatechange = () => {
                if (closed || pc !== connection) return;
                // A browser's "disconnected" is a link that has gone quiet,
                // and it comes back: on a link that drops for seconds at a
                // time, calling it a failure would leave the bar reading
                // failed for the rest of a run that recovered.
                if (connection.connectionState === 'disconnected') {
                    if (client.state.name === 'streaming') setState('connecting');
                    return;
                }
                if (['failed', 'closed'].includes(connection.connectionState)) {
                    setState('failed', `the connection ${connection.connectionState}`);
                }
            };
            try {
                await negotiate(false);
                // Open, and carrying nothing: the operator opened the tab
                // before launching. Saying "connecting" here would describe
                // a connection that is already up and read as a fault.
                if (!client.cameras.length) setState('idle');
                return true;
            } catch (e) {
                setState('failed', e && e.message ? e.message : String(e));
                close();
                return false;
            }
        }

        // The run's cameras exist: bind them to the connection already up.
        async function _attach(cameras) {
            if (!pc && !(await _open())) return;
            const carrying = client.cameras;
            const same = carrying.length === cameras.length && carrying.every((c, i) => c === cameras[i]);
            if (same && Object.keys(client.tracks).length) {
                return;  // already carrying these
            }
            // A connection's video streams only ever grow, and their order
            // is what names them. A run that restarts with a camera gone —
            // or with the same count under other names — cannot be carried
            // by this connection: the offer would describe more streams than
            // the run has, or label every tile with the wrong camera.
            if (carrying.length && !same) {
                close();
                if (!(await _open())) return;
            }
            const h264 = h264Codecs();
            while (transceivers.length < cameras.length) addVideoStream(h264);
            setState('connecting');
            try {
                await negotiate(true);
            } catch (e) {
                setState('failed', e && e.message ? e.message : String(e));
            }
        }

        function open() {
            return serialize(_open);
        }

        function attach(cameras) {
            return serialize(() => _attach(cameras));
        }

        function close() {
            closed = true;
            if (pc) { try { pc.close(); } catch (_) { /* already gone */ } }
            pc = null;
            session = null;
            transceivers = [];
            client.tracks = {};
        }

        return client;
    }

    window.LiveVideo = { createClient, createAgeTracker, rtpFromCapture, CLOCK, WRAP };
})();
