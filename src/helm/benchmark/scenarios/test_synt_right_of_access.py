import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.synt_right_of_access_scenario import SyntRightOfAccessScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, TEST_SPLIT


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = ["prompt", "context", "label"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Constructor + class attributes.
# ---------------------------------------------------------------------------


def test_init_stores_data_path():
    scenario = SyntRightOfAccessScenario(data_path="/tmp/right_of_access.csv")
    assert scenario.data_path == "/tmp/right_of_access.csv"


def test_class_attributes():
    assert SyntRightOfAccessScenario.name == "synt_right_of_access"
    # Three choices: A (must grant), B (may deny without review), C (may deny on reviewable ground)
    assert SyntRightOfAccessScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B", "C"]
    assert SyntRightOfAccessScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "RightOfAccess" in SyntRightOfAccessScenario.description
    assert "HIPAA" in SyntRightOfAccessScenario.description


# ---------------------------------------------------------------------------
# `create_benchmark` — CSV parsing logic.
# ---------------------------------------------------------------------------


def test_create_benchmark_single_row_produces_single_entry():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Should access be granted to the individual?",
                    "context": "Individual requests their own health records. No grounds for denial.",
                    "label": "A",
                }
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Should access be granted to the individual?" in prompt
    assert "Individual requests their own health records" in prompt


def test_create_benchmark_prompt_includes_hipaa_right_of_access_framing():
    """The prompt must reference HIPAA Right of Access and three possible outcomes."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "HIPAA Right of Access" in prompt
    assert "45 CFR 164.524" in prompt
    assert "'A' if access must be granted" in prompt
    assert "'B' if access may be denied without review" in prompt
    assert "'C' if access may be denied on a reviewable ground" in prompt


def test_create_benchmark_multiple_rows_produce_multiple_entries():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "C"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "B", "C"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """`create_benchmark` keys results by composite prompt; identical prompts collapse to one
    entry with the LATER label winning."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "C"},  # overrides
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "C"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` is set at module load to allow large clinical
    messages."""
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "B"}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SyntRightOfAccessScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    """Label 'A' (must grant access) should mark first reference as correct."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B", "C"]
    assert refs[0].is_correct
    assert not refs[1].is_correct
    assert not refs[2].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    """Label 'B' (may deny without review) should mark second reference as correct."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert refs[1].is_correct
    assert not refs[2].is_correct


def test_get_instances_label_c_marks_third_reference_correct():
    """Label 'C' (may deny on reviewable ground) should mark third reference as correct."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "C"}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert not refs[1].is_correct
    assert refs[2].is_correct


def test_get_instances_references_always_have_three_choices_in_fixed_order():
    """All instances must have exactly 3 reference choices in order A, B, C."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "C"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["A", "B", "C"]


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    """Each instance must have exactly one correct reference."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "C"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if CORRECT_TAG in ref.tags]
        assert len(correct) == 1


def test_get_instances_uses_test_split_for_every_instance():
    """All instances must be in TEST_SPLIT."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "C"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert all(instance.split == TEST_SPLIT for instance in instances)


def test_get_instances_input_text_contains_question_and_context():
    """Input text must include both the prompt question and context."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Can individual access records containing psychotherapy notes?",
                    "context": "Records are in hybrid format; psychotherapy notes are segregated.",
                    "label": "C",
                }
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Can individual access records containing psychotherapy notes?" in text
    assert "hybrid format" in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "D", "1", "ABC", "a", "b", "c"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """Only 'A', 'B', and 'C' are valid labels; anything else must raise AssertionError."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    """Must raise an exception when the data file does not exist."""
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SyntRightOfAccessScenario(data_path=non_existent)
        with pytest.raises(Exception):
            scenario.get_instances(output_path=tmp)


def test_get_instances_dedupes_identical_prompts_documented_quirk():
    """Identical composite prompts collapse to one instance; latest label wins."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "C"},  # overrides
                {"prompt": "Different?", "context": "Same", "label": "B"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same Q?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "C"


def test_get_instances_preserves_unicode_in_question_and_context():
    """Non-ASCII characters in clinical scenarios must be preserved."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Puede el paciente acceder a sus registros médicos?",
                    "context": "El paciente solicita acceso a registros de psicoterapia.",
                    "label": "B",
                }
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "¿Puede el paciente acceder a sus registros médicos?" in text
    assert "psicoterapia" in text


# ---------------------------------------------------------------------------
# Important failure cases and edge cases.
# ---------------------------------------------------------------------------


def test_get_instances_mixed_valid_and_invalid_labels_fails_on_first_invalid():
    """If any row has an invalid label, the entire batch should fail."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "INVALID"},  # This should fail
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_handles_empty_prompt_field():
    """Empty prompt should still create an instance."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "",
                    "context": "Individual requests access to medical records.",
                    "label": "A",
                }
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert "Individual requests access" in instances[0].input.text


def test_get_instances_handles_empty_context_field():
    """Empty context should still create an instance."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Must access be granted?",
                    "context": "",
                    "label": "C",
                }
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert "Must access be granted?" in instances[0].input.text


def test_get_instances_case_sensitive_labels():
    """Labels must be exactly 'A', 'B', or 'C' (case-sensitive)."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "a"}])
        scenario = SyntRightOfAccessScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_all_three_labels_in_single_batch():
    """Should correctly handle a batch with all three labels present."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "C"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    answers = [next(ref.output.text for ref in inst.references if ref.is_correct) for inst in instances]
    assert set(answers) == {"A", "B", "C"}


def test_get_instances_special_characters_in_fields():
    """Special characters in prompt/context must be preserved."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Q: 45 CFR §164.524 applies? [Legal Reference]",
                    "context": "Patient & representative; records marked 'CONFIDENTIAL'",
                    "label": "A",
                }
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "§164.524" in text or "$" in text


def test_get_instances_large_batch_processing():
    """Scenario should handle a large number of instances."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        choices = ["A", "B", "C"]
        rows = [{"prompt": f"Q{i}?", "context": f"Context {i}", "label": choices[i % 3]} for i in range(1000)]
        _write_csv(path, rows)
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1000
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 3 for instance in instances)


def test_get_instances_all_a_labels():
    """Should handle batch with all 'A' labels."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "A"},
                {"prompt": "Q3?", "context": "C3", "label": "A"},
            ],
        )
        scenario = SyntRightOfAccessScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if ref.is_correct]
        assert len(correct) == 1
        assert correct[0].output.text == "A"


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SyntRightOfAccessScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "synt_right_of_access"
    assert metadata.display_name == "RightOfAccess"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"


def test_metadata_description_mentions_hipaa_right_of_access():
    scenario = SyntRightOfAccessScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "HIPAA" in description or "Right of Access" in description
    assert "45 CFR 164.524" in description
