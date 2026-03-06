"""tests/test_cli.py"""
import json
import subprocess
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
CLI_MODULE = [sys.executable, "-m", "analyzer.cli"]

def test_analyze_outputs_json():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "grid_layout.png")],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "metadata" in data
    assert "colors" in data

def test_analyze_with_output_file(tmp_path):
    outfile = tmp_path / "result.json"
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "grid_layout.png"), "-o", str(outfile)],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    assert outfile.exists()
    data = json.loads(outfile.read_text())
    assert "metadata" in data

def test_analyze_with_only_flag():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "grid_layout.png"), "--only", "color,edge"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "colors" in data
    assert "edges" in data
    assert "spatial" not in data

def test_analyze_nonexistent_file():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", "/nonexistent/file.png"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode != 0

def test_verbose_flag():
    result = subprocess.run(
        [*CLI_MODULE, "analyze", str(FIXTURES / "solid_red.png"), "-v"],
        capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )
    assert result.returncode == 0
    # Verbose output goes to stderr
    assert "Analyzing" in result.stderr or "Running" in result.stderr
