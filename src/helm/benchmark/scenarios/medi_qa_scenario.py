import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from helm.benchmark.presentation.taxonomy_info import TaxonomyInfo
from helm.common.general import ensure_directory_exists, ensure_file_downloaded
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TEST_SPLIT,
    CORRECT_TAG,
    Input,
    Output,
    ScenarioMetadata,
)


class MediQAScenario(Scenario):
    """
    MEDIQA-QA is a dataset designed to benchmark large language models (LLMs) on medical
    question answering (QA) tasks.
    Each instance in the dataset includes a medical question, a set of candidate answers,
    relevance annotations for ranking, and additional context to evaluate understanding
    and retrieval capabilities in a healthcare setting.

    The dataset encompasses diverse question types, including consumer health queries
    and clinical questions, making it suitable for assessing LLMs' ability to answer
    consumer healthcare questions.

    This dataset comprises two training sets of 104 instances each, a validation set
    of 25 instances, and a testing set of 150 instances.

    Dataset: https://huggingface.co/datasets/bigbio/mediqa_qa
    Paper: https://aclanthology.org/W19-5039/

    Sample Prompt:
        Answer the following consumer health question.

        Question: Noonan syndrome. What are the references with noonan syndrome
        and polycystic renal disease?
        Answer:

    @inproceedings{MEDIQA2019,
        author    = {Asma {Ben Abacha} and Chaitanya Shivade and Dina Demner{-}Fushman},
        title     = {Overview of the MEDIQA 2019 Shared Task on Textual Inference,
                     Question Entailment and Question Answering},
        booktitle = {ACL-BioNLP 2019},
        year      = {2019}
    }
    """

    # Hugging Face `datasets>=4` removed dataset-script loading (`trust_remote_code`).
    # Load the upstream MEDIQA 2019 release directly instead of `bigbio/mediqa_qa`.
    SOURCE_URL = "https://github.com/abachaa/MEDIQA2019/archive/refs/heads/master.zip"
    TEST_XML_RELATIVE_PATH = os.path.join(
        "MEDIQA_Task3_QA",
        "MEDIQA2019-Task3-QA-TestSet-wLabels.xml",
    )

    name = "medi_qa"
    description = (
        "MEDIQA is a benchmark designed to evaluate a model's ability to generate"
        "medically accurate answers to patient-generated questions. Each instance includes a"
        "consumer health question, a set of candidate answers (used in ranking tasks), relevance"
        "annotations, and optionally, additional context. The benchmark focuses on supporting"
        "patient understanding and accessibility in health communication."
    )
    tags = ["knowledge", "biomedical"]

    def _get_highest_ranked_answer(self, answers: List[Dict[str, Dict[str, str]]]) -> str:
        best_answer: str = ""
        for answer in answers:
            if answer["Answer"]["ReferenceRank"] != 1:
                continue
            best_answer = answer["Answer"]["AnswerText"]
            break
        return best_answer

    def process_csv(self, data, split: str) -> List[Instance]:
        instances: List[Instance] = []
        hlog(f"Processing data for {split} split")
        total_tokens: int = 0
        counter = 0
        for row in data:
            row = row["QUESTION"]
            question = row["QuestionText"]
            ground_truth_answer = self._get_highest_ranked_answer(row["AnswerList"])
            id = row["QID"]
            counter += 1
            total_tokens += len(ground_truth_answer.split())
            instances.append(
                Instance(
                    input=Input(question),
                    references=[Reference(Output(ground_truth_answer), tags=[CORRECT_TAG])],
                    split=split,
                    id=id,
                )
            )
        return instances

    @staticmethod
    def _parse_xml(filepath: str) -> List[Dict[str, Any]]:
        """Parse a MEDIQA Task-3 QA XML file into the source-schema rows used by `process_csv`."""
        root = ET.parse(filepath).getroot()
        rows: List[Dict[str, Any]] = []
        for question in root.iterfind("Question"):
            answer_list_element = question.find("AnswerList")
            if answer_list_element is None:
                continue
            answer_list = []
            for answer in answer_list_element.findall("Answer"):
                answer_list.append(
                    {
                        "Answer": {
                            "AID": answer.attrib["AID"],
                            "SystemRank": int(answer.attrib["SystemRank"]),
                            "ReferenceRank": int(answer.attrib["ReferenceRank"]),
                            "ReferenceScore": int(answer.attrib.get("ReferenceScore", "0")),
                            "AnswerURL": (answer.findtext("AnswerURL") or ""),
                            "AnswerText": (answer.findtext("AnswerText") or ""),
                        }
                    }
                )
            rows.append(
                {
                    "QUESTION": {
                        "QID": question.attrib["QID"],
                        "QuestionText": question.findtext("QuestionText") or "",
                        "AnswerList": answer_list,
                    }
                }
            )
        return rows

    def _load_test_rows(self, output_path: str) -> List[Dict[str, Any]]:
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        # GitHub zips contain a single top-level directory; ensure_file_downloaded renames it
        # to target_path when unpacking.
        archive_path = os.path.join(data_path, "MEDIQA2019")
        ensure_file_downloaded(
            source_url=self.SOURCE_URL,
            target_path=archive_path,
            unpack=True,
        )
        test_xml_path = os.path.join(archive_path, self.TEST_XML_RELATIVE_PATH)
        return self._parse_xml(test_xml_path)

    def get_instances(self, output_path: str) -> List[Instance]:
        # Limit to zero shot setting (test split only).
        return self.process_csv(self._load_test_rows(output_path), TEST_SPLIT)

    def get_metadata(self):
        return ScenarioMetadata(
            name="medi_qa",
            display_name="MEDIQA",
            description="MEDIQA is a benchmark designed to evaluate a model's ability to retrieve and "
            "generate medically accurate answers to patient-generated questions. Each "
            "instance includes a consumer health question, a set of candidate answers (used "
            "in ranking tasks), relevance annotations, and optionally, additional context. "
            "The benchmark focuses on supporting patient understanding and accessibility in "
            "health communication.",
            taxonomy=TaxonomyInfo(
                task="Text generation",
                what="Generate medically accurate answers to patient-generated questions.",
                when="Any",
                who="Clinician, Medical Student",
                language="English",
            ),
            main_metric="medi_qa_accuracy",
            main_split="test",
        )
