"""Record FLASH inputs with the standard recording lifecycle and frame loop.

Run with the ordinary ``lerobot-record`` arguments via
``python -m lerobot.policies.flash.record``. The GUI selects this entry point
for a profile with measured geometry or infrared recording enabled.
"""

from copy import deepcopy

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets.camera_calibration import attach_robot_geometry
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.stereo_recording import add_stereo_features, attach_stereo_calibration, capture_stereo
from lerobot.utils.constants import DEFAULT_FEATURES, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame


class _RecordingDataset:
    """Expose ordinary observations to the loop; append recording-only fields at write."""

    def __init__(self, dataset, session):
        self.dataset = dataset
        self.session = session
        self.features = {**session.base_features, **DEFAULT_FEATURES}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def add_frame(self, frame):
        # Processed RGB/depth already belong to frame. Only separately declared
        # recording fields are supplied here, never replacements for them.
        self.dataset.add_frame({**frame, **self.session.frame_extras})


class _RecordingRobot:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session.robot, name)

    def connect(self, *args, **kwargs):
        self.session.robot.connect(*args, **kwargs)
        for dataset in self.session.datasets:
            attach_stereo_calibration(dataset.meta, self.session.robot)

    def get_observation(self):
        observation = self.session.robot.get_observation()
        if not self.session.recording:
            return observation
        images, timing = capture_stereo(self.session.robot)
        # Refresh the ordinary RGB/depth inputs from the same owned snapshot as
        # infrared. The existing observation processor still runs afterwards.
        observation.update({key: value for key, value in images.items() if key in observation})
        self.session.frame_extras = {
            **build_dataset_frame(self.session.extra_features, images, OBS_STR),
            **timing,
        }
        return observation


class RecordingSession:
    """One recording's camera snapshots and dataset metadata, with no global hooks."""

    def __init__(self, robot, features):
        self.robot = robot
        self.base_features = deepcopy(features)
        self.features = add_stereo_features(deepcopy(features), robot)
        self.extra_features = {
            key: value for key, value in self.features.items() if key not in self.base_features
        }
        self.frame_extras = {}
        self.recording = False
        self.datasets = []
        self.recording_robot = _RecordingRobot(self)

    def _attach(self, dataset):
        try:
            # Validate before connecting hardware, including on resume.
            sanity_check_dataset_robot_compatibility(dataset, self.robot, dataset.fps, self.features)
            attach_robot_geometry(dataset.meta, self.robot)
        except Exception:
            dataset.finalize()
            raise
        self.datasets.append(dataset)
        return _RecordingDataset(dataset, self)

    def create(self, *args, **kwargs):
        return self._attach(LeRobotDataset.create(*args, **{**kwargs, "features": self.features}))

    def resume(self, *args, **kwargs):
        return self._attach(LeRobotDataset.resume(*args, **kwargs))

    def record_loop(self, *args, **kwargs):
        from lerobot.scripts.lerobot_record import record_loop

        self.recording = kwargs.get("dataset") is not None or kwargs.get("intervention_dataset") is not None
        try:
            return record_loop(*args, **kwargs)
        finally:
            self.recording = False
            self.frame_extras = {}


def prepare_recording(robot, features):
    session = RecordingSession(robot, features)
    return session.recording_robot, session, session.record_loop


def main():
    from lerobot.scripts.lerobot_record import record
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    record(prepare_recording=prepare_recording)


if __name__ == "__main__":
    main()
