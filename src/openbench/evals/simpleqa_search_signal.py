from inspect_ai import task, Task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate, system_message, use_tools
from inspect_ai.tool import web_search

from openbench.datasets.simpleqa import get_dataset
from openbench.scorers.simpleqa_search_signal import simpleqa_search_signal_scorer



@task
def simpleqa_search_signal(
    grader_model: str = "openai/gpt-5-mini",
) -> Task:
    """SimpleQA variant that treats tool calls as abstention signals.

    The solver exposes a web_search tool but does not execute tool calls
    (generate(tool_calls="none")). If the assistant calls any tool without
    providing an answer, it is scored as NOT_ATTEMPTED.
    """

    return Task(
        dataset=get_dataset(),
        solver=[
            use_tools(web_search(["openai", "tavily"])),
            generate(tool_calls="none"),
        ],
        scorer=simpleqa_search_signal_scorer(model=grader_model),
        name="simpleqa_search_signal",
        config=GenerateConfig(
            temperature=0.0,
        ),
    )
