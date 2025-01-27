import datetime
import pathlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import tiktoken
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, csv_dataset
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import choice
from inspect_ai.solver import Generate, TaskState, chain, multiple_choice, solver
from inspect_ai.util import resource

USER_PROMPT = r"""
Provide a summary of this scientific question while keeping all critical technical terms and relationships and preserving the core question being asked:

{input}

Provide only the summary, no additional explanation, use all the text you need.

Summarized question:
""".strip()

encoding = tiktoken.encoding_for_model("gpt-4o")


@dataclass
class SummarizerState:
    results: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=[
                "sample_id",
                "original_question",
                "summarized_question",
                "compression_ratio",
                "original_length_tokens",
                "expected_summarized_length_tokens",
                "actual_summarized_length_tokens",
                "percent_difference_in_length_tokens",
                "summarization",
            ]
        )
    )

    def add_result(
        self,
        sample_id: str,
        original_question: str,
        summary: str,
        compression_ratio: float,
        summarization: str,
    ) -> None:
        """Add a summarization result to the results DataFrame"""
        original_length = len(encoding.encode(original_question))
        expected_length = int(original_length * compression_ratio)
        actual_length = len(encoding.encode(summary))

        new_row = pd.DataFrame(
            {
                "sample_id": [sample_id],
                "original_question": [original_question],
                "summarized_question": [summary],
                "compression_ratio": [compression_ratio],
                "original_length_tokens": [original_length],
                "expected_summarized_length_tokens": [expected_length],
                "actual_summarized_length_tokens": [actual_length],
                "percent_difference_in_length_tokens": [
                    (actual_length - expected_length) / expected_length
                ],
                "summarization": [summarization],
            }
        )

        self.results = pd.concat([self.results, new_row], ignore_index=True)

    def save_results(
        self,
        filename: str | None = None,
        compression_ratio: float | None = None,
        summarization: str | None = None,
    ) -> None:
        """Save results to CSV file"""
        pathlib.Path(
            f"summarizer_logs/comp-ratio-{compression_ratio}/summarization-{summarization}"
        ).mkdir(parents=True, exist_ok=True)
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            filename = f"summarizer_results_{timestamp}.csv"
        self.results.to_csv(
            f"summarizer_logs/comp-ratio-{compression_ratio}/summarization-{summarization}/{filename}",
            index=False,
        )


@solver
def summarize(
    summarizer_model: str = "openai/gpt-4o",
    compression_ratio: float = 0.5,
    summarization: str = "llm",
) -> Any:
    summarizer = get_model(summarizer_model)
    prompt = resource(USER_PROMPT)
    state = SummarizerState()

    async def solve(task_state: TaskState, generate: Generate) -> TaskState:
        expected_length = int(len(encoding.encode(task_state.input_text)) * compression_ratio)
        if summarization == "llm":
            summary = await summarizer.generate(
                prompt.format(input=task_state.input_text),
                config=GenerateConfig(max_tokens=expected_length),
            )
            summary = summary.completion
        elif summarization == "cut":
            encoded_input = encoding.encode(task_state.input_text)
            summary = encoding.decode(encoded_input[:expected_length])
        else:
            raise ValueError(
                f"Summarization method {summarization} not supported. "
                "Valid methods are 'llm' and 'cut'."
            )

        state.add_result(
            sample_id=str(task_state.sample_id),
            original_question=task_state.input_text,
            summary=summary,
            compression_ratio=compression_ratio,
            summarization=summarization,
        )

        state.save_results(compression_ratio=compression_ratio, summarization=summarization)

        task_state.user_prompt.text = summary
        return task_state

    return solve


@solver
def summarize_and_answer(
    cot: bool = True, compression_ratio: float = 0.5, summarization: str = "llm"
) -> Any:
    """Two-stage solver that first summarizes the question and then answers it."""
    return chain(
        summarize(
            summarizer_model="openai/gpt-4o",
            compression_ratio=compression_ratio,
            summarization=summarization,
        ),
        multiple_choice(shuffle=True, cot=cot),
    )


@task
def gpqa_diamond_compressed(
    cot: bool = True, compression_ratio: float = 0.5, summarization: str = "llm"
) -> Task:
    """GPQA evaluation with compressed questions.

    Args:
        cot: Enable chain-of-thought prompting
        compression_ratio: The ratio of the original question length to the compressed question length
        summarization: The summarization method to use. Valid methods are 'llm' and 'cut'.
    """
    return Task(
        dataset=csv_dataset(
            csv_file="https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
            sample_fields=record_to_sample,
        ),
        solver=summarize_and_answer(
            cot=cot, compression_ratio=compression_ratio, summarization=summarization
        ),
        scorer=choice(),
        config=GenerateConfig(temperature=0.5),
        epochs=4,  # *default* epochs to run eval for
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )
