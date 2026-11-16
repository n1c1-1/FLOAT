"""The blend mode "Linear dodge (add)" has one French name wherever the graph and Post name it."""
import json
from pathlib import Path

FR = Path(__file__).resolve().parents[2] / "astro_gpu" / "client" / "locale" / "fr"


def test_linear_dodge_has_one_french_name():
    post = json.loads((FR / "post.json").read_text(encoding="utf-8"))
    name = post["Linear dodge (add)"]
    for file in ("post.json", "graph.json"):
        for english, french in json.loads((FR / file).read_text(encoding="utf-8")).items():
            if "Linear dodge" in english:
                assert name in french, f"{file}: {english[:60]}"
