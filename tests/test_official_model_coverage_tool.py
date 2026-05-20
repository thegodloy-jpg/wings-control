import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


def load_tool(name):
    import sys

    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    spec = importlib.util.spec_from_file_location(name, TOOLS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _by_model(report, model_name):
    return next(item for item in report["models"] if item["model_name"] == model_name)


def test_load_supported_models_follows_model_utils_inventory():
    tool = load_tool("check_official_model_coverage")

    tables = tool.load_supported_model_tables(
        REPO_ROOT / "wings_control" / "utils" / "model_utils.py"
    )
    models = tool.iter_supported_models(tables)

    assert len(models) == 56
    assert {
        "model_name": "Qwen3-Embedding-0.6B",
        "model_type": "embedding",
        "architecture": "Qwen3ForCausalLM",
    } in models
    assert {
        "model_name": "bge-reranker-large",
        "model_type": "rerank",
        "architecture": "XLMRobertaForSequenceClassification",
    } in models


def test_report_marks_recipe_status_and_official_source_candidates():
    tool = load_tool("check_official_model_coverage")

    report = tool.build_coverage_report(REPO_ROOT)

    deepseek = _by_model(report, "DeepSeek-V3.1")
    assert deepseek["recipe_status"] == "model_recipe"
    assert "sglang" in [item["engine"] for item in deepseek["source_candidates"]]
    assert "vllm_ascend" in [item["engine"] for item in deepseek["source_candidates"]]
    assert deepseek["gaps"] == []

    bge_embedding = _by_model(report, "bge-m3")
    assert bge_embedding["recipe_status"] == "missing"
    assert "missing_recipe" in bge_embedding["gaps"]
    assert "sglang" in [item["engine"] for item in bge_embedding["source_candidates"]]


def test_json_and_text_outputs_are_stable_enough_for_scheduled_checks():
    tool = load_tool("check_official_model_coverage")

    report = tool.build_coverage_report(REPO_ROOT, model_name_filter="Qwen3-32B")
    json_payload = tool.format_json_report(report)
    text_payload = tool.format_text_report(report)

    assert '"models_total": 1' in json_payload
    assert '"model_name": "Qwen3-32B"' in json_payload
    assert "Official model coverage report" in text_payload
    assert "Qwen3-32B" in text_payload
