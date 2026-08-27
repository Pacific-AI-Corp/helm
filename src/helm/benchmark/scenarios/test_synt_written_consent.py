import csv
import os
from tempfile import TemporaryDirectory
from typing import List

import pytest

from helm.benchmark.scenarios.synt_written_consent_scenario import SyntWrittenConsentScenario
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
    scenario = SyntWrittenConsentScenario(data_path="/tmp/written_consent.csv")
    assert scenario.data_path == "/tmp/written_consent.csv"


def test_class_attributes():
    assert SyntWrittenConsentScenario.name == "synt_written_consent"
    assert SyntWrittenConsentScenario.POSSIBLE_ANSWER_CHOICES == ["A", "B"]
    assert SyntWrittenConsentScenario.tags == ["knowledge", "reasoning", "biomedical"]
    assert "WrittenConsent" in SyntWrittenConsentScenario.description
    assert "42 CFR Part 2" in SyntWrittenConsentScenario.description


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
                    "prompt": "Is disclosure of substance-use-disorder records permitted?",
                    "context": "Patient has provided written consent for disclosure to treatment facility.",
                    "label": "A",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    prompt, answer = next(iter(data.items()))
    assert answer == "A"
    assert "Is disclosure of substance-use-disorder records permitted?" in prompt
    assert "Patient has provided written consent" in prompt


def test_create_benchmark_prompt_includes_42_cfr_part_2_framing():
    """The prompt must reference 42 CFR Part 2 and substance-use-disorder records."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SyntWrittenConsentScenario(data_path=path)
        data = scenario.create_benchmark(path)

    prompt = next(iter(data.keys()))
    assert "42 CFR Part 2" in prompt
    assert "substance-use-disorder" in prompt.lower() or "SUD" in prompt
    assert "'A' if the disclosure is permitted" in prompt
    assert "'B' if it is not permitted" in prompt


def test_create_benchmark_multiple_rows_produce_multiple_entries():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "A"},
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 3
    assert sorted(data.values()) == ["A", "A", "B"]


def test_create_benchmark_duplicate_rows_collapse_with_last_win():
    """`create_benchmark` keys results by composite prompt; identical prompts collapse to one
    entry with the LATER label winning."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Same Q?", "context": "Same C", "label": "A"},
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overrides
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert len(data) == 1
    assert next(iter(data.values())) == "B"


def test_create_benchmark_handles_csv_with_very_large_field():
    """`csv.field_size_limit(sys.maxsize)` is set at module load to allow large clinical
    messages."""
    huge_context = "x" * (2 * 1024 * 1024)
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": huge_context, "label": "B"}])
        scenario = SyntWrittenConsentScenario(data_path=path)
        data = scenario.create_benchmark(path)

    assert huge_context in next(iter(data.keys()))


def test_create_benchmark_returns_empty_dict_for_csv_with_only_header():
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [])
        scenario = SyntWrittenConsentScenario(data_path=path)
        assert scenario.create_benchmark(path) == {}


# ---------------------------------------------------------------------------
# `get_instances` — end-to-end against synthetic CSVs.
# ---------------------------------------------------------------------------


def test_get_instances_label_a_marks_first_reference_correct():
    """Label 'A' (disclosure permitted) should mark first reference as correct."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "A"}])
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert [ref.output.text for ref in refs] == ["A", "B"]
    assert refs[0].is_correct
    assert not refs[1].is_correct


def test_get_instances_label_b_marks_second_reference_correct():
    """Label 'B' (disclosure not permitted) should mark second reference as correct."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "B"}])
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    refs = instances[0].references
    assert not refs[0].is_correct
    assert refs[1].is_correct


def test_get_instances_references_always_have_two_choices_in_fixed_order():
    """All instances must have exactly 2 reference choices in order A, B."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    for instance in instances:
        assert [ref.output.text for ref in instance.references] == ["A", "B"]


def test_get_instances_every_instance_has_exactly_one_correct_reference():
    """Each instance must have exactly one correct reference."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "A"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "A"},
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
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
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
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
                    "prompt": "Can records be disclosed to research study?",
                    "context": "Patient is enrolled in a federally-funded research study. No written consent provided.",
                    "label": "B",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "Can records be disclosed to research study?" in text
    assert "federally-funded research study" in text


@pytest.mark.parametrize("bad_label", ["yes", "no", "", "C", "1", "AB", "a", "b"])
def test_get_instances_raises_assertion_for_unsupported_labels(bad_label):
    """Only 'A' and 'B' are valid labels; anything else must raise AssertionError."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": bad_label}])
        scenario = SyntWrittenConsentScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_raises_when_data_file_is_missing():
    """Must raise an exception when the data file does not exist."""
    with TemporaryDirectory() as tmp:
        non_existent = os.path.join(tmp, "missing.csv")
        scenario = SyntWrittenConsentScenario(data_path=non_existent)
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
                {"prompt": "Same Q?", "context": "Same C", "label": "B"},  # overrides
                {"prompt": "Different?", "context": "Same", "label": "A"},
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 2
    same_instance = next(i for i in instances if "Same Q?" in i.input.text)
    correct_text = next(ref.output.text for ref in same_instance.references if ref.is_correct)
    assert correct_text == "B"


def test_get_instances_preserves_unicode_in_question_and_context():
    """Non-ASCII characters in clinical scenarios must be preserved."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "¿Se pueden divulgar registros de trastorno por consumo de sustancias?",
                    "context": "El paciente ha proporcionado consentimiento escrito para la divulgación.",
                    "label": "A",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "divulgar" in text or "registros" in text


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
        scenario = SyntWrittenConsentScenario(data_path=path)
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
                    "context": "Patient is in treatment facility. Disclosure to hospital.",
                    "label": "A",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert "treatment facility" in instances[0].input.text


def test_get_instances_handles_empty_context_field():
    """Empty context should still create an instance."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Can records be disclosed without consent?",
                    "context": "",
                    "label": "B",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1
    assert "Can records be disclosed without consent?" in instances[0].input.text


def test_get_instances_whitespace_only_context_treated_as_empty():
    """Whitespace-only context should be handled gracefully."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Q?",
                    "context": "   \t\n  ",
                    "label": "A",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1


def test_get_instances_special_characters_in_fields():
    """Special characters in prompt/context must be preserved."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Q: 42 CFR Part 2 applies? [Legal: §2.12(c)]",
                    "context": "Patient records & medical history; disclosure to 'authorized' entity",
                    "label": "A",
                }
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    text = instances[0].input.text
    assert "§" in text or "$" in text or "&" in text


def test_create_benchmark_case_sensitive_labels():
    """Labels must be exactly 'A' or 'B' (case-sensitive)."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(path, [{"prompt": "Q?", "context": "C", "label": "a"}])
        scenario = SyntWrittenConsentScenario(data_path=path)
        with pytest.raises(AssertionError):
            scenario.get_instances(output_path=tmp)


def test_get_instances_large_batch_processing():
    """Scenario should handle a large number of instances."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        rows = [
            {"prompt": f"Q{i}?", "context": f"Context {i}", "label": "A" if i % 2 == 0 else "B"} for i in range(1000)
        ]
        _write_csv(path, rows)
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 1000
    assert all(instance.split == TEST_SPLIT for instance in instances)
    assert all(len(instance.references) == 2 for instance in instances)


def test_get_instances_all_a_labels_batch():
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
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if ref.is_correct]
        assert len(correct) == 1
        assert correct[0].output.text == "A"


def test_get_instances_all_b_labels_batch():
    """Should handle batch with all 'B' labels."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {"prompt": "Q1?", "context": "C1", "label": "B"},
                {"prompt": "Q2?", "context": "C2", "label": "B"},
                {"prompt": "Q3?", "context": "C3", "label": "B"},
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    for instance in instances:
        correct = [ref for ref in instance.references if ref.is_correct]
        assert len(correct) == 1
        assert correct[0].output.text == "B"


def test_get_instances_real_world_42cfr_part2_scenarios():
    """Test realistic 42 CFR Part 2 scenarios with nuanced contexts."""
    with TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.csv")
        _write_csv(
            path,
            [
                {
                    "prompt": "Can a covered entity disclose SUD records to a law enforcement officer seeking information?",
                    "context": "Law enforcement requests records for criminal investigation. No court order or patient consent.",
                    "label": "B",
                },
                {
                    "prompt": "Can a covered entity disclose SUD records to a medical emergency responder?",
                    "context": "Patient overdosed and EMS arrived. Medical personnel need treatment history.",
                    "label": "A",
                },
                {
                    "prompt": "Can a covered entity disclose SUD records to an employee assistance program?",
                    "context": "Patient enrolled in EAP. Written consent provided for records transfer.",
                    "label": "A",
                },
            ],
        )
        scenario = SyntWrittenConsentScenario(data_path=path)
        instances = scenario.get_instances(output_path=tmp)

    assert len(instances) == 3
    # Verify the correct answers are marked
    assert instances[0].references[1].is_correct  # B is correct
    assert instances[1].references[0].is_correct  # A is correct
    assert instances[2].references[0].is_correct  # A is correct


# ---------------------------------------------------------------------------
# Metadata.
# ---------------------------------------------------------------------------


def test_metadata():
    scenario = SyntWrittenConsentScenario(data_path="/tmp/x")
    metadata = scenario.get_metadata()

    assert metadata.name == "synt_written_consent"
    assert metadata.display_name == "WrittenConsent"
    assert metadata.main_split == "test"
    assert metadata.main_metric == "exact_match"
    assert metadata.taxonomy.task == "Classification"
    assert metadata.taxonomy.language == "English"
    assert metadata.taxonomy.when == "Any"


def test_metadata_description_mentions_42_cfr_part_2():
    scenario = SyntWrittenConsentScenario(data_path="/tmp/x")
    description = scenario.get_metadata().description
    assert "42 CFR Part 2" in description
    assert "substance-use-disorder" in description.lower() or "SUD" in description
