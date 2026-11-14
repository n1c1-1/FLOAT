"""What the frames' headers say of each session, for the table: read off the GUI thread by the project model's own
service, as the Sources page reads them (services/inspect.project_frames through ``InspectService``: exposure, ISO or
gain, temperature, the nights' dates, the calibration frames' exposures and the darks' matching by the Calibration's
own rules). Nothing is ever deduced from a file's or a folder's name.

A graph held in memory has no project model: nothing is read, the table shows only the listing's counts.
"""
from __future__ import annotations

import json
import logging

from PySide6.QtCore import QObject, Signal

from ...graphcore.model import FILE_KINDS

log = logging.getLogger(__name__)
JOB = "graphview.headers"


def session_sigs(p) -> dict:
    """{session id: what its reading depends on}: its files of each kind (and those left out by hand)."""
    out = {}
    for s in getattr(p, "sessions", None) or []:
        out[s.id] = json.dumps([[str(x) for x in (getattr(s, k, None) or [])] for k in FILE_KINDS] +
                               [getattr(s, "left_out", None) or {}], sort_keys=True, default=str)
    return out


def _read(p, headers):
    """The worker's job: the sessions' signatures and their ProjectFrames, from the same definition."""
    from ..services.inspect import project_frames
    sigs = session_sigs(p)
    return sigs, project_frames(p, None, headers)


class HeaderReadings(QObject):
    """The sessions' header readings of the project model behind the bridge's document (none without one)."""

    changed = Signal()                     # a reading came: the table shows it

    def __init__(self, bridge, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.rows: dict = {}               # session id -> SessionRow (services/inspect)
        self.read_sigs: dict = {}          # session id -> the signature it was read with
        self.asked: dict = {}              # session id -> the signature last asked
        self.camera: dict = {}
        self._token = None
        self._service = None
        self._model = None

    # ------------------------------------------------------------------ the model and its service
    @property
    def model(self):
        m = self.bridge.doc.model
        return m if m is not None and hasattr(m, "p") else None

    def _service_of(self, m):
        if self._model is not m:
            if self._service is not None:
                try:
                    self._service.jobReady.disconnect(self._ready)
                    self._service.jobFailed.disconnect(self._failed)
                except (RuntimeError, TypeError):
                    pass
            from ..services.inspect import service_of
            self._service = service_of(m)
            self._service.jobReady.connect(self._ready)
            self._service.jobFailed.connect(self._failed)
            self._model = m
            self.rows, self.read_sigs, self.asked = {}, {}, {}
        return self._service

    def request(self):
        """Ask the reading again when a session's files changed since the last one (cheap when nothing did)."""
        m = self.model
        if m is None:
            return
        sv = self._service_of(m)
        try:
            sigs = session_sigs(m.p)
        except Exception:  # noqa: BLE001 - a definition being changed: asked at the next change
            return
        if sigs == self.asked:
            return
        self.asked = sigs
        self._token = sv.request_job(JOB, lambda p: _read(p, sv.headers))

    def _ready(self, name, token, result):
        if name != JOB or token != self._token:
            return
        sigs, data = result
        self.rows = {r.id: r for r in data.sessions}
        self.read_sigs = dict(sigs)
        self.camera = dict(data.camera or {})
        self.changed.emit()

    def _failed(self, name, token, why):
        if name != JOB or token != self._token:
            return
        log.warning("graph page: the frames' headers could not be read: %s", why)
        self.read_sigs = dict(self.asked)          # nothing more to wait for: the table says nothing
        self.rows = {}
        self.changed.emit()

    # ------------------------------------------------------------------ what the table reads
    @property
    def active(self) -> bool:
        return self.model is not None

    def _sid(self, block):
        b = self.bridge.graph.blocks.get(block)
        return getattr(b, "sid", None) if b is not None else None

    def pending(self, block=None) -> bool:
        """A reading is asked and not come yet (for this Session block, or for any)."""
        if not self.active:
            return False
        if block is None:
            return any(self.read_sigs.get(k) != v for k, v in self.asked.items())
        sid = self._sid(block)
        return sid is not None and sid in self.asked and self.read_sigs.get(sid) != self.asked.get(sid)

    def row(self, block):
        """The header reading of a Session block (services/inspect.SessionRow) when it is that of its current files;
        None while it is read, or without a project model."""
        sid = self._sid(block)
        if sid is None or self.pending(block) or self.read_sigs.get(sid) != self.asked.get(sid):
            return None
        return self.rows.get(sid)
