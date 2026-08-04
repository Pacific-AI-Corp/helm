import json
import os
from typing import Any, Dict, Iterator, List, Optional

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.benchmark.scenarios.scenario import (
    CORRECT_TAG,
    TEST_SPLIT,
    Input,
    Instance,
    Output,
    Reference,
    Scenario,
    ScenarioMetadata,
)
from helm.common.general import ensure_directory_exists, ensure_file_downloaded


class HeadQAScenario(Scenario):
    """
    From "HEAD-QA: A Healthcare Dataset for Complex Reasoning" (Vilares et al.), HEAD-QA is a multi-choice
    question-answering dataset designed to evaluate reasoning on challenging healthcare-related questions.
    The questions are sourced from Spanish healthcare exams for specialized positions, covering various topics
    such as Medicine, Nursing, Psychology, Chemistry, Pharmacology, and Biology.

    Example from the dataset:

    Question:
    The excitatory postsynaptic potentials:

    A) They are all or nothing.
    B) They are hyperpolarizing.
    C) They can be added.
    D) They spread long distances.

    Answer:
    The answer is C. Explanation: None provided in this dataset.

    @InProceedings{HEAD-QA,
    author = {David Vilares and Manuel Vilares and Carlos Gómez-Rodríguez},
    title = {HEAD-QA: A Healthcare Dataset for Complex Reasoning},
    year = {2019},
    abstract = {We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex
    reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system,
    and are challenging even for highly specialized humans. We then consider monolingual (Spanish) and
    cross-lingual (to English) experiments with information retrieval and neural techniques. We show that:
    (i) HEAD-QA challenges current methods, and (ii) the results lag well behind human performance,
    demonstrating its usefulness as a benchmark for future work.}}


    Task:
    Given a question and its multiple-choice answers, models must identify the correct answer, corresponding to the
    `ra` field in the dataset. The dataset spans six healthcare domains and is challenging even for experts.
    """

    # Hugging Face `datasets>=4` removed dataset-script loading (`trust_remote_code`).
    # Download the upstream archive referenced by `dvilares/head_qa` directly.
    SOURCE_URL = "https://huggingface.co/datasets/dvilares/head_qa/resolve/main/data/head-qa-es-en-pdfs.zip"
    LANGUAGE_DIRS = {"es": "HEAD", "en": "HEAD_EN"}
    SKIP_VQA: bool = True
    SKIP_TEXTQA: bool = False

    name = "head_qa"
    description = (
        "HeadQA is a benchmark consisting of biomedical multiple-choice questions intended to"
        "evaluate a model's medical knowledge and reasoning. Each instance presents a clinical"
        "or scientific question with four answer options, requiring the model to select the most"
        "appropriate answer."
    )
    tags = ["question_answering", "biomedical", "medicine"]

    def __init__(self, language: str = "en", category: Optional[str] = None):
        """Initialize the HEAD-QA scenario.

        Args:
            language (str, optional): Language of the dataset. Defaults to "en".
            category (str, optional): Category of the dataset. If None, all categories are used.
        """
        super().__init__()
        self.language: str = language
        self.category: Optional[str] = category
        assert (
            self.SKIP_VQA or self.SKIP_TEXTQA
        ), "Failed to initialize HeadQAScenario, one of `SKIP_VQA` or `SKIP_TEXTQA` must be True."
        assert self.language in self.LANGUAGE_DIRS, f"Unsupported HEAD-QA language: {self.language}"

    def _iter_test_examples(self, output_path: str) -> Iterator[Dict[str, Any]]:
        data_path: str = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        archive_path = os.path.join(data_path, "head-qa-es-en-pdfs")
        ensure_file_downloaded(
            source_url=self.SOURCE_URL,
            target_path=archive_path,
            unpack=True,
        )

        language_dir = self.LANGUAGE_DIRS[self.language]
        filepath = os.path.join(archive_path, language_dir, f"test_{language_dir}.json")
        with open(filepath, encoding="utf-8") as f:
            head_qa = json.load(f)

        for exam in head_qa["exams"]:
            content = head_qa["exams"][exam]
            name = content["name"].strip()
            year = content["year"].strip()
            category = content["category"].strip()
            for question in content["data"]:
                image_path = (question.get("image") or "").strip()
                yield {
                    "name": name,
                    "year": year,
                    "category": category,
                    "qid": int(str(question["qid"]).strip()),
                    "qtext": question["qtext"].strip(),
                    "ra": int(str(question["ra"]).strip()),
                    "image": os.path.join(archive_path, image_path) if image_path else None,
                    "answers": [
                        {
                            "aid": int(answer["aid"]),
                            "atext": str(answer["atext"]).strip(),
                        }
                        for answer in question["answers"]
                    ],
                }

    def get_instances(self, output_path: str) -> List[Instance]:
        instances: List[Instance] = []
        for example in self._iter_test_examples(output_path):
            # Whether to process Visual Question Answering (VQA) examples
            if self.SKIP_VQA and example["image"] is not None:
                continue

            # Whether to process Text Question Answering (TextQA) examples
            if self.SKIP_TEXTQA and example["image"] is None:
                continue

            # If specified, filter by category
            if self.category is not None and example["category"] != self.category:
                continue

            instances.append(
                Instance(
                    input=Input(text=example["qtext"]),
                    references=[
                        Reference(
                            Output(text=option["atext"]),
                            tags=[CORRECT_TAG] if option["aid"] == example["ra"] else [],
                        )
                        for option in example["answers"]
                    ],
                    split=TEST_SPLIT,
                    extra_data={
                        "id": example["qid"],
                        "name": example["name"],
                        "category": example["category"],
                        "year": example["year"],
                    },
                )
            )

        return instances

    def get_metadata(self):
        return ScenarioMetadata(
            name="head_qa",
            display_name="HeadQA",
            description="HeadQA is a benchmark consisting of biomedical multiple-choice questions "
            "intended to evaluate a model's medical knowledge and reasoning. Each instance "
            "presents a clinical or scientific question with four answer options, requiring "
            "the model to select the most appropriate answer [(Vilares et al., "
            "2019)](https://arxiv.org/abs/1906.04701).",
            taxonomy=TaxonomyInfo(
                task="Question answering",
                what="Medical knowledge testing",
                when="Any",
                who="Medical student, Researcher",
                language="English",
            ),
            main_metric="exact_match",
            main_split="test",
        )
