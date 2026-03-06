"""tests/test_integration.py — Run full pipeline on the gem-boxes reference image."""
import json
import pytest
from pathlib import Path
from analyzer.pipeline import Pipeline

GEM_BOXES = Path("/home/sisawat/working/design/claude-to-figma/gem-boxes-mj.png")

@pytest.mark.skipif(not GEM_BOXES.exists(), reason="gem-boxes-mj.png not available")
class TestGemBoxesIntegration:

    def test_full_pipeline(self):
        pipe = Pipeline()
        result = pipe.run(str(GEM_BOXES))
        assert result["metadata"]["filename"] == "gem-boxes-mj.png"
        assert len(result["colors"]["dominant"]) >= 3
        assert len(result["spatial"]["regions"]) >= 1
        assert result["textures"]["global"]["roughness"] >= 0.0

    def test_dominant_colors_make_sense(self):
        pipe = Pipeline(analyzers=["color"])
        result = pipe.run(str(GEM_BOXES))
        # The image is mostly dark — dominant color should be dark
        top_color = result["colors"]["dominant"][0]
        r, g, b = top_color["rgb"]
        assert max(r, g, b) < 100, f"Expected dark dominant color, got {top_color}"

    def test_output_to_file(self, tmp_path):
        pipe = Pipeline()
        result = pipe.run(str(GEM_BOXES))
        out = tmp_path / "gem-analysis.json"
        out.write_text(json.dumps(result, indent=2))
        loaded = json.loads(out.read_text())
        assert loaded["metadata"]["filename"] == "gem-boxes-mj.png"
        # Print for manual inspection
        print(f"\nFull analysis written to: {out}")
        print(f"Colors found: {len(loaded['colors']['dominant'])}")
        print(f"Regions found: {len(loaded['spatial']['regions'])}")
        print(f"Contours found: {len(loaded['edges']['contours'])}")
