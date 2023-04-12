import time
import openai
from pathlib import Path
import os
import logging
import inspect
import json
from typing import List

from utils import truncate_response, get_plan_list, log_conversation

openai.api_key = os.environ["OPENAI_API_KEY"]


class Agent:
    def __init__(
        self,
        args,
        exemplar_name: str,
        log_path: Path,
    ):
        self.args = args
        self.exemplar_name = exemplar_name
        self.subtasks = None
        self.prompts = None
        self.description = None
        self.num_obs_filter_layers = None
        self.obs_filter_examples = None
        self.reformat_examples = None
        self.planning_examples = None
        self.setup_prompts()
        self.state_match_idx = 0
        self.action_plan_str = None
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def reformat_subtasks(self, task: str):
        prompt = "\n" + self.prompts["SubtaskReformation"] + "\n"
        prompt += "\nTask:\n" + task + "\nSubtasks:\n"
        subtasks = truncate_response(self.llm(prompt), "\n\nTask:")
        self.subtasks = get_plan_list(subtasks)
        assert len(self.subtasks) == len(self.prompts["Demonstrations"])

    def update_exemplars(self) -> bool:
        self.state_match_idx += 1
        return self.state_match_idx >= len(self.prompts["Demonstrations"])

    def obs_filter(self, raw_obs: str) -> str:
        assert not self.args.no_filter
        obs = raw_obs
        for num_layer in range(self.num_obs_filter_layers):
            if self.exemplar_name == "book-flight" and self.state_match_idx == 1:
                prompt = self.obs_filter_examples[self.state_match_idx][num_layer] + obs
            else:
                prompt = (
                    self.obs_filter_examples[self.state_match_idx][num_layer]
                    + "\nState:\n"
                    + obs
                    + "\nObservation:\n"
                )
            obs = truncate_response(self.llm(prompt), "\n\nState:")

        return obs

    def task_reformation(self, obs: str, extra: str = None) -> str:
        if self.args.plugin_name == "minimax":
            from utils import minimax

            move = minimax(eval(obs))
            return f"The next move is position {move[0] * 3 + move[1]}."
        else:
            prompt = self.reformat_examples[self.state_match_idx] + "\n"
            prompt += "Observation:\n" + obs
            if extra is not None:
                prompt += "\n" + extra
            prompt += "\nReformation:\n"
            return truncate_response(self.llm(prompt), "\n\nObservation:")

    def plan(self, reformation: str) -> List[str]:
        prompt = self.description
        prompt += self.planning_examples[self.state_match_idx]
        prompt += "\nObservation:\n" + reformation
        if self.args.exemplar_decomposition and "SubtaskReformation" in self.prompts:
            prompt += "\n" + self.subtasks[self.state_match_idx]
        prompt += "\nPlan:\n"
        self.action_plan_str = truncate_response(self.llm(prompt), "\n\nObservation:")
        return get_plan_list(self.action_plan_str)

    def setup_prompts(self):
        with open(f"exemplars.json", "r") as rf:
            self.prompts = json.load(rf)[self.exemplar_name]

        self.description = (
            "We have an autonomous agent that can perform atomic actions specified by natural language. Here are the available actions:\n"
            + self.prompts["Description"]
            + "\n"
        )

        if self.args.exemplar_decomposition:
            all_demo_list = self.prompts["Demonstrations"]
        else:
            all_demo_list = [sum(self.prompts["Demonstrations"], [])]

        # Setup observation filter prompting.
        if not self.args.no_filter:
            assert (
                not self.args.task_as_reformation and self.args.reformat_input != "task"
            )
            assert isinstance(all_demo_list[0], list)
            assert isinstance(all_demo_list[0][0]["Observation"], list)
            # self.num_obs_filter_layers = 1
            self.num_obs_filter_layers = len(all_demo_list[0][0]["Observation"])
            self.obs_filter_examples = []
            for num_subtask, demo_list in enumerate(all_demo_list):
                examples = []
                for num_layer in range(self.num_obs_filter_layers):
                    if num_layer == 0:
                        example_list = [
                            "\nState:\n"
                            + d["State"]
                            + "\nObservation:\n"
                            + d["Observation"][num_layer]
                            + "\n"
                            for d in demo_list
                            if d["State"] != "" and d["Observation"][num_layer] != ""
                        ]
                    else:
                        example_list = [
                            "\nState:\n"
                            + d["Observation"][num_layer - 1]
                            + "\nObservation:\n"
                            + d["Observation"][num_layer]
                            + "\n"
                            for d in demo_list
                            if d["Observation"][num_layer - 1] != ""
                            and d["Observation"][num_layer] != ""
                        ]
                    example_str = "".join(example_list)
                    if "ObsFilterPrefix" in self.prompts:
                        example_str = (
                            self.prompts["ObsFilterPrefix"][num_subtask][num_layer]
                            + "\n"
                            + example_str
                        )
                    examples.append(example_str)
                self.obs_filter_examples.append(examples)

        # Setup reformation prompting.
        if (
            self.args.reformat_input is not None
            and not self.args.task_as_reformation
            and self.args.plugin_name is None
        ):
            self.reformat_examples = []
            for demo_list in all_demo_list:
                if self.args.reformat_input == "task":
                    obs_key = "Task"
                elif not self.args.no_filter:
                    obs_key = "Observation"
                else:
                    obs_key = "State"
                example_list = [
                    "\nObservation:\n"
                    + (d[obs_key][-1] if obs_key == "Observation" else d[obs_key])
                    + (
                        "\n" + d["Task"]
                        if self.args.reformat_input == "obs_task"
                        else ""
                    )
                    + "\nReformation:\n"
                    + d["Reformation"]
                    + "\n"
                    for d in demo_list
                    if (
                        d[obs_key][-1] != ""
                        if obs_key == "Observation"
                        else d[obs_key] != ""
                    )
                    and d["Reformation"] != ""
                ]
                example_str = "".join(example_list)
                if "ReformationPrefix" in self.prompts:
                    example_str = self.prompts["ReformationPrefix"] + "\n" + example_str

                self.reformat_examples.append(example_str)

        # Setup planning prompting.
        self.planning_examples = []
        for demo_list in all_demo_list:
            if self.args.task_as_reformation:
                assert not self.args.exemplar_decomposition
                obs_key = "Task"
            elif self.args.reformat_input is not None:
                obs_key = "Reformation"
            elif not self.args.no_filter:
                obs_key = "Observation"
            else:
                obs_key = "State"
            example_list = [
                "\nObservation:\n"
                + (d[obs_key][-1] if obs_key == "Observation" else d[obs_key])
                + (
                    "\n" + d["Reformation"]
                    if self.args.exemplar_decomposition and "Reformation" in d
                    else ""
                )
                + "\nPlan:\n"
                + d["Plan"]
                + "\n"
                for d in demo_list
                if (
                    d[obs_key][-1] != ""
                    if obs_key == "Observation"
                    else d[obs_key] != ""
                )
                and d["Plan"] != ""
            ]
            example_str = "".join(example_list)
            if "PlanningPrefix" in self.prompts:
                example_str = "\n" + self.prompts["PlanningPrefix"] + "\n" + example_str
            if "PlanningSuffix" in self.prompts:
                if isinstance(self.prompts["PlanningSuffix"], list):
                    idx = len(self.planning_examples)
                    suffix = self.prompts["PlanningSuffix"][idx]
                    if len(suffix) > 0:
                        example_str = example_str + "\n" + suffix + "\n"
                else:
                    example_str = (
                        example_str + "\n" + self.prompts["PlanningSuffix"] + "\n"
                    )

            self.planning_examples.append(example_str)

    def llm(self, context: str) -> str:
        """Send a request to the language model"""

        logging.info(
            f"Send a request to the language model from {inspect.stack()[1].function}"
        )
        max_tokens = self.args.max_tokens

        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.args.api_name,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": context},
                    ],
                )

                message = response["choices"][0]["message"]["content"]

            except Exception as e:
                print(e)
                if "maximum context" in str(e):
                    if max_tokens <= 256:
                        raise ValueError
                    else:
                        max_tokens = max_tokens // 2
                time.sleep(10)
            else:
                if message:
                    break

        log_conversation(context, message, self.log_path)

        return message
