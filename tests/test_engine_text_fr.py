"""The stages' progress lines that carry numbers or names reach the French window in French."""
from __future__ import annotations

import pytest

from astro_gpu.client import i18n
from astro_gpu.client.services.run import engine_text


@pytest.fixture(autouse=True)
def _fr():
    i18n.set_language("fr")
    yield
    i18n.set_language("en")


@pytest.mark.parametrize("msg,want", [
    ("reading NSNS DR0.2 H-alpha (northern sky, ~1')", "lecture de NSNS DR0.2 H-alpha (ciel nord, ~1')"),
    ("detecting the stars of 432 frames (1 processes)", "détection des étoiles de 432 images (1 processus)"),
    ("classic: rejection (linear_fit) 253/2822 rows", "classique : rejet (linear_fit) 253/2822 lignes"),
    ("24/432 frames joined", "24/432 images jointes"),
    ("the scratch folder keeps 11.1 GB of calibrated frames for the next run: C:/x",
     "le dossier de travail garde 11,1 Go d'images calibrées pour le prochain calcul : C:/x"),
])
def test_progress_lines_in_french(msg, want):
    assert engine_text(msg) == want
