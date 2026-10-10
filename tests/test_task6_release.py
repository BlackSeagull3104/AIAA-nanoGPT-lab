import ast
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_assignments(relative_path):
    source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)
    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return source, tree, assignments


def test_manifest_is_explicitly_pending_and_uses_delivery_ref():
    manifest = json.loads(
        (REPO_ROOT / "submission_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["repository_url"].endswith("BlackSeagull3104/AIAA-nanoGPT-lab")
    assert manifest["delivery_ref"] == "course/nanogpt-experiments"
    assert manifest["tested_source_sha"] is None
    assert manifest["release_tag"]["status"] == "pending"
    reproduction = manifest["task6_reproduction"]
    assert reproduction["status"] == "pending"
    assert reproduction["lightweight_wandb_run"]["run_id"] is None
    assert reproduction["runtime_seconds"] is None


def test_manifest_paths_exist_or_are_declared_pending():
    manifest = json.loads(
        (REPO_ROOT / "submission_manifest.json").read_text(encoding="utf-8")
    )
    for relative_path in manifest["config_paths"].values():
        assert (REPO_ROOT / relative_path).is_file(), relative_path
    report = manifest["task6_reproduction"]["clean_clone_report"]
    assert manifest["task6_reproduction"]["clean_clone_report_status"] == "not_created"
    assert not (REPO_ROOT / report).exists()


def test_readme_references_release_entry_points():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    required = (
        "requirements.txt",
        "submission_manifest.json",
        "config/train_task6_repro.py",
        "scripts/task6_hf_gpt2_inference.py",
        "--sample_output_path=out-task6-repro/sample.jsonl",
        "Task 6 clean-clone reproduction: **pending**",
    )
    for text in required:
        assert text in readme


def test_sample_output_paths_are_configurable_and_backward_compatible():
    source, tree, _ = load_assignments("sample.py")
    assert 'sample_output_path = os.path.join("reports", "samples_scratch_before.jsonl")' in source
    assert 'stats_output_path = os.path.join("reports", "samples_scratch_before_stats.json")' in source
    configurator_call_index = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Expr) and "configurator.py" in ast.unparse(node)
    )
    assignment_indexes = {
        node.targets[0].id: index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignment_indexes["sample_output_path"] < configurator_call_index
    assert assignment_indexes["stats_output_path"] < configurator_call_index
    assert "sample_path = sample_output_path" in source
    assert "stats_path = stats_output_path" in source


def test_hf_entry_point_is_single_prompt_cpu_safe_and_lazy():
    source, tree, assignments = load_assignments("scripts/task6_hf_gpt2_inference.py")
    assert assignments["MODEL_ID"] == "openai-community/gpt2"
    assert 'torch.device("cpu")' in source
    assert "--local-files-only" in source
    assert "do_sample=False" in source
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported = {alias.name for node in top_level_imports for alias in node.names}
    assert "torch" not in imported
    assert "transformers" not in imported
    assert "wandb" not in source


def test_task6_training_config_stays_within_smoke_boundaries():
    _, _, config = load_assignments("config/train_task6_repro.py")
    assert config["device"] == "cpu"
    assert config["dtype"] == "float32"
    assert config["compile"] is False
    assert config["wandb_log"] is False
    assert config["max_iters"] == 2
    assert config["eval_iters"] == 1
    assert config["batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 1
    assert config["block_size"] <= 32
    assert config["n_layer"] == 1
    assert config["n_head"] == 1
    assert config["n_embd"] <= 32
