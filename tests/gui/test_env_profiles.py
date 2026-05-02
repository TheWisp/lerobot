"""Tests for the Environment (sim) profile API and schema introspection.

Mirror of tests/gui/test_robot_profiles.py shape, but for the env module
that backs the GUI's Environment tab.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from lerobot.gui.api import env as env_module
from lerobot.gui.server import app

# ============================================================================
# Schema introspection
# ============================================================================


class TestEnvSchemas:
    """Tests for /api/env/schemas — driven by EnvConfig.get_known_choices()."""

    def test_returns_at_least_gym_manipulator(self):
        """gym_manipulator (gym-hil) must be discoverable; it's the only type
        wired through to live teleop/record/replay."""
        client = TestClient(app)
        r = client.get("/api/env/schemas")
        assert r.status_code == 200
        type_names = [s["type_name"] for s in r.json()]
        assert "gym_manipulator" in type_names

    def test_includes_other_upstream_env_types(self):
        """The full EnvConfig registry should surface, even though only
        gym_manipulator is wired through right now."""
        client = TestClient(app)
        type_names = [s["type_name"] for s in client.get("/api/env/schemas").json()]
        for expected in ("aloha", "pusht", "libero", "metaworld"):
            assert expected in type_names, f"missing {expected}"

    def test_gym_manipulator_fields_have_task_and_fps(self):
        client = TestClient(app)
        schemas = client.get("/api/env/schemas").json()
        gm = next(s for s in schemas if s["type_name"] == "gym_manipulator")
        names = {f["name"] for f in gm["fields"]}
        assert "task" in names
        assert "fps" in names


# ============================================================================
# Profile CRUD
# ============================================================================


@pytest.fixture
def tmp_env_profiles_dir(tmp_path, monkeypatch):
    """Redirect env profile storage to a tmp dir for isolation."""
    monkeypatch.setattr(env_module, "ENV_PROFILES_DIR", tmp_path)
    return tmp_path


class TestEnvProfileCRUD:
    """Tests for env profile CRUD via FastAPI TestClient."""

    def test_list_empty(self, tmp_env_profiles_dir):
        client = TestClient(app)
        r = client.get("/api/env/profiles")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_then_list(self, tmp_env_profiles_dir):
        client = TestClient(app)
        profile = {
            "type": "gym_manipulator",
            "name": "panda_keyboard",
            "fields": {"name": "gym_hil", "task": "PandaPickCubeKeyboard-v0", "fps": 10},
        }
        r = client.post("/api/env/profiles", json=profile)
        assert r.status_code == 200
        assert r.json()["status"] == "created"

        r = client.get("/api/env/profiles")
        assert r.status_code == 200
        names = {p["name"] for p in r.json()}
        assert "panda_keyboard" in names

    def test_create_duplicate_returns_409(self, tmp_env_profiles_dir):
        client = TestClient(app)
        profile = {"type": "gym_manipulator", "name": "p", "fields": {}}
        client.post("/api/env/profiles", json=profile)
        r = client.post("/api/env/profiles", json=profile)
        assert r.status_code == 409

    def test_get_round_trips(self, tmp_env_profiles_dir):
        client = TestClient(app)
        original = {
            "type": "gym_manipulator",
            "name": "round_trip",
            "fields": {"task": "PandaPickCubeBase-v0", "fps": 5, "device": "cpu"},
        }
        client.post("/api/env/profiles", json=original)
        r = client.get("/api/env/profiles/round_trip")
        assert r.status_code == 200
        # Storage may add empty defaults; assert the fields we set survive.
        body = r.json()
        assert body["type"] == "gym_manipulator"
        assert body["name"] == "round_trip"
        assert body["fields"] == original["fields"]

    def test_get_missing_returns_404(self, tmp_env_profiles_dir):
        client = TestClient(app)
        r = client.get("/api/env/profiles/nonexistent")
        assert r.status_code == 404

    def test_update_overwrites(self, tmp_env_profiles_dir):
        client = TestClient(app)
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "p",
                "fields": {"fps": 10},
            },
        )
        client.put(
            "/api/env/profiles/p",
            json={
                "type": "gym_manipulator",
                "name": "p",
                "fields": {"fps": 30},
            },
        )
        body = client.get("/api/env/profiles/p").json()
        assert body["fields"]["fps"] == 30

    def test_delete_then_get_404(self, tmp_env_profiles_dir):
        client = TestClient(app)
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "p",
                "fields": {},
            },
        )
        r = client.delete("/api/env/profiles/p")
        assert r.status_code == 200
        assert client.get("/api/env/profiles/p").status_code == 404

    def test_delete_missing_returns_404(self, tmp_env_profiles_dir):
        client = TestClient(app)
        r = client.delete("/api/env/profiles/nonexistent")
        assert r.status_code == 404

    def test_rename_succeeds(self, tmp_env_profiles_dir):
        client = TestClient(app)
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "old",
                "fields": {"fps": 7},
            },
        )
        r = client.post("/api/env/profiles/old/rename", json={"new_name": "newname"})
        assert r.status_code == 200
        assert client.get("/api/env/profiles/old").status_code == 404
        body = client.get("/api/env/profiles/newname").json()
        assert body["name"] == "newname"
        assert body["fields"]["fps"] == 7  # data preserved

    def test_rename_collision_returns_409(self, tmp_env_profiles_dir):
        client = TestClient(app)
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "a",
                "fields": {},
            },
        )
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "b",
                "fields": {},
            },
        )
        r = client.post("/api/env/profiles/a/rename", json={"new_name": "b"})
        assert r.status_code == 409

    def test_corrupted_json_in_list_skipped(self, tmp_env_profiles_dir):
        """A corrupted profile file should be logged but not break listing."""
        (tmp_env_profiles_dir / "broken.json").write_text("not valid json {")
        # Plus a valid one so we can confirm listing still works
        (tmp_env_profiles_dir / "ok.json").write_text(
            json.dumps(
                {
                    "type": "gym_manipulator",
                    "name": "ok",
                    "fields": {},
                }
            )
        )
        client = TestClient(app)
        r = client.get("/api/env/profiles")
        assert r.status_code == 200
        names = {p["name"] for p in r.json()}
        assert "ok" in names
        # Corrupt one is silently skipped (logged warning).


# ============================================================================
# Field introspection
# ============================================================================


class TestRegisteredTasks:
    """`/api/env/registered-tasks` enumerates gym tasks for a namespace,
    falling back to a hardcoded list when the package can't be imported."""

    def test_gym_hil_returns_registry_path(self):
        """gym_hil is installed in the project's lerobot env, so the live
        gym registry path must populate; source must be 'registry'."""
        client = TestClient(app)
        r = client.get("/api/env/registered-tasks?name=gym_hil")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "gym_hil"
        assert body["source"] == "registry"
        assert len(body["tasks"]) >= 10  # 10 known as of gym_hil 0.1.13
        # Suffix-only (no namespace prefix); the env-profile task field
        # binds to these directly because gym_manipulator prepends the
        # namespace when calling gym.make.
        assert all("/" not in t for t in body["tasks"])
        assert "PandaPickCubeKeyboard-v0" in body["tasks"]

    def test_default_name_is_gym_hil(self):
        client = TestClient(app)
        r = client.get("/api/env/registered-tasks")
        assert r.json()["name"] == "gym_hil"

    def test_unknown_package_returns_empty_fallback(self):
        """Unknown name is not in _FALLBACK_TASKS, so we should get an
        empty list with a warning. The frontend can surface the warning
        instead of showing an empty dropdown silently."""
        client = TestClient(app)
        r = client.get("/api/env/registered-tasks?name=definitely_not_a_real_pkg")
        assert r.status_code == 200
        body = r.json()
        assert body["source"] == "fallback"
        assert body["tasks"] == []
        assert "not installed" in body["warning"]

    def test_known_package_uninstalled_uses_fallback_list(self):
        """If gym_hil were uninstalled, we'd want the dropdown still
        populated with the captured-at-build-time list. Simulate by
        patching importlib.import_module to raise."""
        import importlib

        def fake_import(name):
            raise ImportError(f"simulated: {name} not installed")

        client = TestClient(app)
        with patch.object(importlib, "import_module", fake_import):
            r = client.get("/api/env/registered-tasks?name=gym_hil")
        body = r.json()
        assert body["source"] == "fallback"
        # gym_hil has a populated _FALLBACK_TASKS entry — degraded but usable.
        assert len(body["tasks"]) == 10
        assert "PandaPickCubeKeyboard-v0" in body["tasks"]
        assert "warning" in body


class TestIntrospectFields:
    """Tests for the dataclass field introspection helper."""

    def test_skips_blacklisted_fields(self):
        """processor / features / robot / teleop are nested configs we don't
        expose in the simple v1 form — they must not appear in introspection."""
        from lerobot.envs.configs import HILSerlRobotEnvConfig

        env_module._ensure_envs_loaded()
        fields = env_module._introspect_fields(HILSerlRobotEnvConfig)
        names = {f["name"] for f in fields}
        for skipped in ("processor", "features", "features_map", "robot", "teleop"):
            assert skipped not in names, f"{skipped} should be skipped"

    def test_field_metadata_shape(self):
        """Every field row must have name/type_str/required/default keys."""
        from lerobot.envs.configs import HILSerlRobotEnvConfig

        env_module._ensure_envs_loaded()
        fields = env_module._introspect_fields(HILSerlRobotEnvConfig)
        assert len(fields) > 0
        for f in fields:
            assert set(f.keys()) >= {"name", "type_str", "required", "default"}
