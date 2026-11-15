"""The star width and the picture size a workflow's settings use are that workflow's: in a project of H-alpha, O III
and S II, the Selection's maximum star size of S II typed in pixels is converted with S II's star width, not with that
of the first workflow computed (H-alpha). A workflow not registered yet has no star width (its threshold is in times
the median), whatever another workflow measured."""
from __future__ import annotations

from types import SimpleNamespace

from astro_gpu.client import model as modelmod
from astro_gpu.client.model import ProjectModel


def _project(monkeypatch):
    measured = {(1, "register"): {"fwhm_px": 2.0}, (1, "stack"): {"grid": [4000, 3000]},
                (3, "register"): {"fwhm_px": 3.5}, (3, "stack"): {"grid": [3990, 2990]}}

    def outputs(p, wid, stage, which=None):
        m = measured.get((wid, stage))
        return None if m is None or which else SimpleNamespace(metrics=m)

    def unsolved(p, wid):
        raise ValueError("not solved")
    monkeypatch.setattr(modelmod.api, "stage_outputs", outputs)
    monkeypatch.setattr(modelmod.api, "sky_objects", unsolved)
    p = SimpleNamespace(workflows=[SimpleNamespace(id=i) for i in (1, 2, 3)], sessions=[], root=None)
    return modelmod._facts_job(p)


def _shown(facts, pid):
    class Shown:
        _facts = facts
        path = SimpleNamespace(pid=pid)
        _measured = ProjectModel._measured
    return Shown()


def test_each_workflow_has_its_own_star_width_and_picture_size(monkeypatch):
    facts = _project(monkeypatch)
    ha, o3, s2 = (_shown(facts, i) for i in (1, 2, 3))
    assert ProjectModel.fwhm_median(ha) == 2.0 and ProjectModel.image_size(ha) == (4000, 3000)
    assert ProjectModel.fwhm_median(s2) == 3.5 and ProjectModel.image_size(s2) == (3990, 2990)   # H-alpha's before
    assert ProjectModel.fwhm_median(o3) is None and ProjectModel.image_size(o3) is None          # not computed yet
    # the project's facts (the Report's): unchanged, from the first workflow computed
    assert facts.register == {"fwhm_px": 2.0} and facts.stack == {"grid": [4000, 3000]}
