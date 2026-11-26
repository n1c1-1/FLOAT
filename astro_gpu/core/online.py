"""The one online service FLOAT still asks, the H-alpha survey of the field (hips2fits, the survey's own tile server),
given little time: a short connect timeout, a total budget per query, and a service that does not answer is set aside for
a while so that the next source is asked at once, with one line in the log saying so.

FLOAT reads no other online service: the star data are files on this computer (core/sky_catalogue.py).

How:
- ``available(name)``: a tiny request to the service (connect 8 s, answer 12 s), once per 10 minutes; a service that
  fails it is set aside for 10 minutes.
- ``call(name, fn, budget)``: ``fn`` runs in a background thread for at most ``budget`` seconds; a network failure or a
  query over budget sets the service aside and raises ``ServiceDown``.
- ``ASTRO_GPU_OFFLINE=1`` sets every service aside (no network at all).

THE RULE: NOTHING IS DOWNLOADED DURING A RUN. A run (the runner's worker process, and any stage run in another process:
``runner.Pipeline._run_stage``) switches ``forbid`` on; from then on every way to the network raises
``NetworkForbidden`` at once, with what was asked and of whom: the request layer here (``call``, ``available``,
``get``), and below it every socket of the process (an audit hook on ``socket.connect`` and the name lookups: a library
that does not go through this module is stopped too; the machine's own address is left alone). The stages read only what
is on the disk. What a run needs is installed before it starts: the star data by the run window (client/services/run.py,
core/sky_catalogue.py), a survey by an explicit action of the user outside the run (the Gradient block's Get survey).
tests/test_no_download_in_run.py holds the rule.
"""
from __future__ import annotations

import os
import socket
import threading
import time
import urllib.error
from collections.abc import Callable
from typing import Any

PROBE_S = (8.0, 12.0)       # (connect, answer) of the reachability probe
DOWN_FOR_S = 600.0          # a service that failed is set aside for this long

LABELS = {
    "hips2fits": "the CDS cut-out service",
    "hips": "the survey's own tile server",
}
PROBES = {
    "hips2fits": "https://alasky.cds.unistra.fr/hips-image-services/hips2fits",
}

_LOCK = threading.Lock()
_DOWN: dict[str, tuple[float, str]] = {}        # name -> (time until which it is set aside, why)
_PROBED: dict[str, float] = {}                  # name -> time of the last successful probe
_PROBE_LOCKS: dict[str, threading.Lock] = {}


class ServiceDown(Exception):
    """A service did not answer (or failed); ``why`` says it in plain words."""

    def __init__(self, name: str, why: str):
        super().__init__(f"{label(name)} {why}")
        self.name, self.why = name, why


class NetworkForbidden(BaseException):
    """The network was asked while a run forbids it (``forbid``). A BaseException, as KeyboardInterrupt: the broad
    ``except Exception`` of a fallback cannot take it for a service that does not answer and go on; the runner makes it
    the stage's failure, with this message."""


def label(name: str | None) -> str:
    return LABELS.get(name or "", name or "")


# ------------------------------------------------------------------ the guard: no network during a run

#: why the network is forbidden now ("the Astrometry block", "this run"), innermost last; empty: allowed. Process-wide,
#: not per thread: the threads a stage starts (the sky tiles, the spectra) are stopped as the stage itself
_FORBID: list[str] = []
#: the socket events stopped while forbidden, and the addresses left alone (the machine itself: a pipe of a library)
_SOCKET_EVENTS = frozenset({"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                            "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg"})
_LOCAL = frozenset({None, "", "localhost", "127.0.0.1", "::1", b"localhost", b"127.0.0.1", b"::1"})
_HOOKED: list[bool] = []


def forbidden() -> str | None:
    """Why the network is forbidden now (the run, the block running), None when it is allowed."""
    return _FORBID[-1] if _FORBID else None


def _refuse(what: str) -> NetworkForbidden:
    return NetworkForbidden(f"nothing is downloaded during a run: asked for {what} while {_FORBID[-1]} ran. What a run "
                            "needs comes from the disk: install it first (Settings › Catalogs & astrometry)")


def guard(what: str) -> None:
    """Raise ``NetworkForbidden`` at once when the network is forbidden; ``what``: what was asked, of whom."""
    if _FORBID:
        raise _refuse(what)


def _audit(event: str, args: tuple) -> None:
    """The socket net under the request layer (an audit hook: it costs one test of a list per audited event)."""
    if not _FORBID or event not in _SOCKET_EVENTS:
        return
    if event.startswith("socket.get"):                       # a name lookup: (host, ...)
        host = args[0] if args else None
    else:                                                    # (socket, address): (host, port[, ...]), a pipe's path
        addr = args[1] if len(args) > 1 else None
        if not isinstance(addr, tuple):
            return                                           # AF_UNIX and the like: the machine itself
        host = addr[0] if addr else None
    if host in _LOCAL or (isinstance(host, str) and host.startswith("127.")):
        return
    raise _refuse(f"a connection to {host if isinstance(host, str) else repr(host)}")


def _hook() -> None:
    if not _HOOKED:                              # an audit hook cannot be taken out: added once, read through _FORBID
        import sys
        sys.addaudithook(_audit)
        _HOOKED.append(True)


class forbid:  # noqa: N801 - used as a function: ``with online.forbid("the Astrometry block"):``
    """No network while the block runs (nested: the innermost reason is said). ``forbid(why).on()`` without ``with``:
    for the rest of the process (the runner's worker)."""

    def __init__(self, why: str):
        self.why = why

    def on(self) -> forbid:
        _hook()
        _FORBID.append(self.why)
        return self

    def __enter__(self) -> forbid:
        return self.on()

    def __exit__(self, *exc) -> None:
        if self.why in _FORBID:
            del _FORBID[len(_FORBID) - 1 - _FORBID[::-1].index(self.why)]


def offline() -> bool:
    return os.environ.get("ASTRO_GPU_OFFLINE", "").strip() not in ("", "0", "false", "no")


def is_down(name: str) -> bool:
    if offline():
        return True
    with _LOCK:
        d = _DOWN.get(name)
    return d is not None and time.monotonic() < d[0]


def why_down(name: str) -> str:
    if offline():
        return "is not asked (offline mode)"
    with _LOCK:
        d = _DOWN.get(name)
    return d[1] if d else "does not answer"


def mark_down(name: str, why: str = "does not answer") -> None:
    with _LOCK:
        _DOWN[name] = (time.monotonic() + DOWN_FOR_S, why)
        _PROBED.pop(name, None)


def reset() -> None:
    """Forget every verdict (tests; a new session)."""
    with _LOCK:
        _DOWN.clear()
        _PROBED.clear()


def _probe(name: str) -> bool:
    url = PROBES.get(name)
    if not url:
        return True
    import urllib.request
    try:
        try:
            with urllib.request.urlopen(url, timeout=PROBE_S[1]) as r:
                status = r.status
        except urllib.error.HTTPError as e:                  # an answer with an error code is an answer
            status = e.code
        # a probe asks a fixed page that exists: "not found" or "gone" there is the service's own outage page, not a
        # wrong query
        ok = status < 500 and status not in (404, 410)
        why = f"answers with an error (HTTP {status})"
    except Exception as e:  # noqa: BLE001 - any failure means: not reachable now
        ok, why = False, _plain(e)
    if not ok:
        mark_down(name, why)
    return ok


def available(name: str) -> bool:
    """False when the service is set aside or does not answer a tiny request (asked once per 10 minutes)."""
    guard(f"an answer of {label(name)}")
    if is_down(name):
        return False
    if name not in PROBES:
        return True
    with _LOCK:
        lk = _PROBE_LOCKS.setdefault(name, threading.Lock())
    with lk:                                     # the threads of a stage wait for one probe
        if is_down(name):
            return False
        with _LOCK:
            t = _PROBED.get(name)
        if t is not None and time.monotonic() - t < DOWN_FOR_S:
            return True
        ok = _probe(name)
        if ok:
            with _LOCK:
                _PROBED[name] = time.monotonic()
        return ok


def network_error(e: BaseException) -> bool:
    """Whether an exception says the service is unreachable / failing (not: our query was wrong)."""
    if isinstance(e, NetworkForbidden):
        return False
    if isinstance(e, urllib.error.HTTPError):
        return e.code >= 500
    if isinstance(e, (TimeoutError, ConnectionError, socket.timeout, urllib.error.URLError, OSError)):
        return True
    txt = f"{type(e).__name__} {e}".lower()
    return any(k in txt for k in ("timed out", "timeout", "connection", "unreachable", "503", "502", "504",
                                  "service unavailable", "remote end closed"))


def unreachable(e: BaseException) -> bool:
    """The service cannot be reached at all (no connection, no name, refused): not merely slow on a query."""
    w = _plain(e)
    return w.startswith(("does not answer (no connection)", "cannot be reached", "refuses"))


def _plain(e: BaseException) -> str:
    name, msg = type(e).__name__.lower(), str(e).lower()
    txt = f"{name} {msg}"
    if "connecttimeout" in name or "connect timeout" in msg or "connecttimeouterror" in msg:
        return "does not answer (no connection)"
    if "timed out" in txt or "timeout" in txt:
        return "does not answer (silent)"
    if "name resolution" in txt or "getaddrinfo" in txt or "nodename" in txt:
        return "cannot be reached (no network?)"
    if "refused" in txt:
        return "refuses the connection"
    return f"failed ({str(e)[:120]})"


def call(name: str, fn: Callable[[], Any], budget: float, what: str = "data") -> Any:
    """``fn()`` in a background thread for at most ``budget`` seconds. Raises ``ServiceDown`` when the service is set
    aside or unreachable, fails with a network error, or takes longer than the budget (the thread is abandoned; its own
    socket timeouts end it); ``NetworkForbidden`` during a run (``what``: what is asked, said then)."""
    guard(f"{what} from {label(name)}")
    if not available(name):
        raise ServiceDown(name, why_down(name))
    box: dict = {}

    def run():
        try:
            box["r"] = fn()
        except BaseException as e:  # noqa: BLE001 - handed to the caller
            box["e"] = e
    t = threading.Thread(target=run, daemon=True, name=f"online-{name}")
    t.start()
    t.join(budget)
    if t.is_alive():
        why = f"took more than {budget:.0f} s"
        mark_down(name, why)
        raise ServiceDown(name, why)
    if "e" in box:
        e = box["e"]
        if network_error(e):
            why = _plain(e)
            mark_down(name, why)
            raise ServiceDown(name, why) from e
        raise e
    return box["r"]


def get(url: str, timeout: float = 60.0) -> bytes:
    """The bytes at ``url`` (one GET; for ``call``): the only plain download of the stages' modules, through the guard."""
    guard(f"the file {url}")
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()
