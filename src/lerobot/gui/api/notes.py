"""Notes API: free-text notes on datasets, training runs, and checkpoints."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from lerobot.gui import notes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/notes", tags=["notes"])

# Notes are small local files, but reading one still touches the filesystem —
# which can be a stalled NFS mount or a spun-down disk holding a dataset source.
# Its own single-thread pool, per gui-async-hygiene: a slow stat here must not
# take a slot from video decode or camera teardown. Serialised on purpose, so a
# tree render's batch cannot fan out into a thread per row.
_notes_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gui-notes")


class NoteBody(BaseModel):
    note: str = ""


class NoteResponse(BaseModel):
    note: str


def _checked(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        raise HTTPException(status_code=400, detail="path must be absolute")
    return p


def _read(path: Path) -> str:
    return notes.read(path)


@router.get("", response_model=NoteResponse)
async def get_note(path: str = Query(..., description="Absolute artifact path")) -> NoteResponse:
    """Read one artifact's note (empty string if it has none)."""
    p = _checked(path)
    note = await asyncio.get_event_loop().run_in_executor(_notes_executor, _read, p)
    return NoteResponse(note=note)


@router.put("", response_model=NoteResponse)
async def put_note(body: NoteBody, path: str = Query(...)) -> NoteResponse:
    """Set one artifact's note. An empty note deletes it."""
    p = _checked(path)

    def _write() -> str:
        return notes.write(p, body.note)

    try:
        note = await asyncio.get_event_loop().run_in_executor(_notes_executor, _write)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except notes.NoteNotRepresentableError as e:
        # The note's text, not the server, is the problem — 400 so the UI can
        # show the reason next to the editor instead of a generic failure.
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return NoteResponse(note=note)


@router.get("/bulk")
async def get_notes(paths: list[str] = Query(default=[])) -> dict[str, str]:
    """Read many notes at once, for decorating a tree.

    Postcondition: one key per requested path — unannotated artifacts map to
    ``""``, so the caller renders one code path. Paths that are not absolute are
    skipped rather than failing the batch, since a tree render must not break on
    one bad row.
    """

    def _read_many() -> dict[str, str]:
        out: dict[str, str] = {}
        # One parse per notes file, not per path: a run's checkpoints all live
        # in the same file and a tree expands them together.
        by_file: dict[Path, list[tuple[str, str]]] = {}
        for raw in paths:
            p = Path(raw)
            if not p.is_absolute():
                continue
            notes_file, key = notes.locate(p)
            by_file.setdefault(notes_file, []).append((raw, key))

        for notes_file, wanted in by_file.items():
            entries = notes.read_all(notes_file)
            for raw, key in wanted:
                out[raw] = entries.get(key, "")
        return out

    return await asyncio.get_event_loop().run_in_executor(_notes_executor, _read_many)
