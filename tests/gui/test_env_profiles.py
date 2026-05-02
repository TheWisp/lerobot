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


class TestRoundTripIntegrity:
    """The backend must not drop, invent, or silently coerce profile fields
    on a CRUD round-trip. The frontend's dirty detection compares the
    saved-from-API state against the freshly-collected form state — any
    asymmetry here makes it look dirty when it shouldn't be."""

    def test_minimal_profile_unchanged_on_round_trip(self, tmp_env_profiles_dir):
        """A profile with only the absolute-minimum fields must come back
        with exactly the same fields. No defaults injected, no fields
        dropped."""
        client = TestClient(app)
        original = {
            "type": "gym_manipulator",
            "name": "minimal",
            "fields": {"name": "gym_hil", "task": "PandaPickCubeKeyboard-v0"},
        }
        client.post("/api/env/profiles", json=original)
        body = client.get("/api/env/profiles/minimal").json()
        # Strict equality on the user-controlled keys.
        assert body["type"] == original["type"]
        assert body["name"] == original["name"]
        assert body["fields"] == original["fields"], (
            f"fields changed on round-trip: sent {original['fields']}, got {body['fields']}"
        )

    def test_full_profile_unchanged_on_round_trip(self, tmp_env_profiles_dir):
        """All field types preserved exactly."""
        client = TestClient(app)
        original = {
            "type": "gym_manipulator",
            "name": "full",
            "fields": {
                "name": "gym_hil",
                "task": "PandaPickCubeGamepad-v0",
                "fps": 7,
                "device": "cpu",
                "max_parallel_tasks": 2,
                "disable_env_checker": False,
            },
        }
        client.post("/api/env/profiles", json=original)
        body = client.get("/api/env/profiles/full").json()
        assert body["fields"] == original["fields"]

    def test_update_then_read_returns_updated(self, tmp_env_profiles_dir):
        """PUT followed by GET must return the PUT body exactly. Without
        this guarantee the frontend can't reliably show 'no unsaved
        changes' after a successful save."""
        client = TestClient(app)
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "p",
                "fields": {"task": "PandaPickCubeBase-v0"},
            },
        )
        updated = {
            "type": "gym_manipulator",
            "name": "p",
            "fields": {"task": "PandaPickCubeKeyboard-v0", "fps": 15},
        }
        client.put("/api/env/profiles/p", json=updated)
        body = client.get("/api/env/profiles/p").json()
        assert body["fields"] == updated["fields"]

    def test_empty_fields_dict_preserved(self, tmp_env_profiles_dir):
        """An empty `fields: {}` must round-trip as empty, not be
        silently filled with schema defaults."""
        client = TestClient(app)
        client.post(
            "/api/env/profiles",
            json={
                "type": "gym_manipulator",
                "name": "empty",
                "fields": {},
            },
        )
        body = client.get("/api/env/profiles/empty").json()
        assert body["fields"] == {}


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


class TestFieldChoices:
    """Tests for the per-(type, field) enum-alike choices registry. Until
    upstream types these fields as `Literal[...]` we hand-curate them; the
    schema response surfaces a `choices` list when applicable."""

    def test_global_render_mode_appears_for_aloha(self):
        """render_mode is a global enum-alike, expect it on every env type
        that declares the field."""
        client = TestClient(app)
        schemas = client.get("/api/env/schemas").json()
        aloha = next(s for s in schemas if s["type_name"] == "aloha")
        rm = next(f for f in aloha["fields"] if f["name"] == "render_mode")
        assert "choices" in rm
        assert rm["choices"] == ["rgb_array", "human", "rgb_array_list"]

    def test_global_device_for_isaaclab_arena(self):
        """device is a real field on IsaaclabArenaEnv (one of the few env
        types that surface it inside EnvConfig). gym_manipulator's device
        lives one level up on GymManipulatorConfig and is NOT in this
        schema — it's added by the frontend."""
        client = TestClient(app)
        schemas = client.get("/api/env/schemas").json()
        ila = next(s for s in schemas if s["type_name"] == "isaaclab_arena")
        device = next(f for f in ila["fields"] if f["name"] == "device")
        assert device["choices"] == ["cuda", "cpu", "mps"]

    def test_per_type_obs_type_differs_between_envs(self):
        """obs_type's allowed values differ per env type — pusht has
        environment_state_agent_pos which the others don't."""
        client = TestClient(app)
        schemas = {s["type_name"]: s for s in client.get("/api/env/schemas").json()}
        aloha_obs = next(f for f in schemas["aloha"]["fields"] if f["name"] == "obs_type")
        pusht_obs = next(f for f in schemas["pusht"]["fields"] if f["name"] == "obs_type")
        libero_obs = next(f for f in schemas["libero"]["fields"] if f["name"] == "obs_type")

        assert aloha_obs["choices"] == ["pixels", "pixels_agent_pos"]
        assert pusht_obs["choices"] == ["pixels_agent_pos", "environment_state_agent_pos"]
        assert libero_obs["choices"] == ["pixels", "pixels_agent_pos"]

        # The per-type registry must take precedence over the global one
        # (which doesn't define obs_type — but the assertion documents
        # the precedence rule).
        assert pusht_obs["choices"] != aloha_obs["choices"]

    def test_libero_control_mode_choices(self):
        """libero's control_mode comments out 'or "absolute"' — captured."""
        client = TestClient(app)
        schemas = {s["type_name"]: s for s in client.get("/api/env/schemas").json()}
        cm = next(f for f in schemas["libero"]["fields"] if f["name"] == "control_mode")
        assert cm["choices"] == ["relative", "absolute"]

    def test_no_choices_field_for_freeform_strings(self):
        """Free-form fields like task and camera_name must NOT have a
        choices list — no premature enumeration. (`task` is dropdown'd
        via the live registry endpoint, not this static registry.)"""
        client = TestClient(app)
        schemas = {s["type_name"]: s for s in client.get("/api/env/schemas").json()}
        gm = next(f for f in schemas["gym_manipulator"]["fields"] if f["name"] == "task")
        assert "choices" not in gm

    def test_choices_for_helper_precedence(self):
        """Per-type entry beats global entry for the same field name."""
        # If we ever add obs_type to _GLOBAL_FIELD_CHOICES, the per-type
        # entries must still win. Sanity-check the helper directly so
        # the precedence rule has a unit guard, not just integration.
        with patch.dict(env_module._GLOBAL_FIELD_CHOICES, {"obs_type": ["GLOBAL"]}, clear=False):
            # Per-type entry beats global one
            assert env_module._choices_for("aloha", "obs_type") == ["pixels", "pixels_agent_pos"]
            # Type with no specific entry falls through to the (mocked) global
            assert env_module._choices_for("gym_manipulator", "obs_type") == ["GLOBAL"]

    def test_choices_for_unknown_returns_none(self):
        """Unknown (type, field) returns None — caller knows it's free-form."""
        assert env_module._choices_for("aloha", "task") is None
        assert env_module._choices_for("nonsense_type", "obs_type") is None
