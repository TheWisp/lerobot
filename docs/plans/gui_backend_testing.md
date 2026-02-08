# GUI Backend Testing Plan

## Overview

Testing strategy for the LeRobot dataset GUI backend (`src/lerobot/gui/`).

## Current Test Coverage

| Component | Location | Tests | Status |
|-----------|----------|-------|--------|
| `dataset_tools.py` | `tests/datasets/test_dataset_tools.py` | 63 | ✅ Complete |
| Trim regression | `tests/datasets/test_trim_regression.py` | 3 | ✅ Complete |
| Trim video content | `tests/datasets/test_trim_video_content.py` | 2 | ✅ Complete |
| Merge regression | `tests/datasets/test_merge_regression.py` | 1 | ✅ Complete |
| `frame_cache.py` | `tests/gui/test_frame_cache.py` | 11 | ✅ Complete |
| `state.py` | `tests/gui/test_state.py` | 15 | ✅ Complete |
| `api/datasets.py` | - | 0 | ⏳ Pending |
| `api/edits.py` | - | 0 | ⏳ Pending |
| `api/playback.py` | - | 0 | ⏳ Pending (Low Priority) |

---

## Completed Tests

### FrameCache (`tests/gui/test_frame_cache.py`)
- ✅ `test_invalidate_dataset_removes_correct_entries`
- ✅ `test_invalidate_dataset_updates_current_bytes`
- ✅ `test_invalidate_nonexistent_dataset`
- ✅ `test_lru_eviction_removes_oldest_entries`
- ✅ `test_lru_eviction_multiple_entries`
- ✅ `test_access_moves_to_end_of_lru`
- ✅ `test_put_and_get`
- ✅ `test_cache_miss_returns_none`
- ✅ `test_stats_accuracy`
- ✅ `test_clear`
- ✅ `test_get_or_decode_caches_result`

### AppState (`tests/gui/test_state.py`)
- ✅ `test_get_edits_for_dataset_filters_correctly`
- ✅ `test_get_edits_for_dataset_empty_result`
- ✅ `test_get_edits_for_dataset_preserves_order`
- ✅ `test_trim_replacement_logic`
- ✅ `test_trim_replacement_different_episode_untouched`
- ✅ `test_add_edit`
- ✅ `test_remove_edit_valid_index`
- ✅ `test_remove_edit_invalid_index`
- ✅ `test_clear_edits_all`
- ✅ `test_clear_edits_by_dataset`
- ✅ `test_is_episode_deleted_true`
- ✅ `test_is_episode_deleted_false`
- ✅ `test_is_episode_deleted_trim_not_counted`
- ✅ `test_get_episode_trim_exists`
- ✅ `test_get_episode_trim_not_found`

---

## Remaining Tests (Future Work)

### API Tests (Medium Priority)
Requires FastAPI TestClient and mock dataset fixtures.

**Dataset API (`tests/gui/test_api_datasets.py`)**
- `test_open_dataset_already_open_returns_existing`
- `test_list_datasets`
- `test_close_dataset`
- `test_list_episodes`
- `test_get_frame_cache_control_headers`

**Edits API (`tests/gui/test_api_edits.py`)**
- `test_mark_episode_deleted`
- `test_set_episode_trim`
- `test_discard_edits`

### WebSocket Tests (Low Priority)
Complex async testing, may defer.

**Playback API (`tests/gui/test_api_playback.py`)**
- `test_playback_commands`

---

## Running Tests

```bash
# Run all GUI tests
pytest tests/gui/ -v

# Run with coverage
pytest tests/gui/ --cov=src/lerobot/gui --cov-report=term-missing
```
