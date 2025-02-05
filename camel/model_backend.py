# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from abc import ABC, abstractmethod
from typing import Any, Dict

#import openai
import tiktoken
import config

import camel.typing
from camel.localai import LocalAI, LocalChatCompletion
from camel.typing import ModelType
from chatdev.statistics import prompt_cost
from chatdev.utils import log_visualize

try:
    from openai.types.chat import ChatCompletion

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version

import os

DECENTRALIZE = False

if 'BASE_URL' in os.environ:
    BASE_URL = os.environ['BASE_URL']
else:
    BASE_URL = None
if 'RUN_LOCALLY' in os.environ:
    RUN_LOCALLY = os.environ['RUN_LOCALLY']
    if 'DECENTRALIZE' in os.environ:
        DECENTRALIZE = os.environ['DECENTRALIZE']
else:
    RUN_LOCALLY = False
    OPENAI_API_KEY = "DNU"


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs):
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass


class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type: ModelType, model_config_dict: Dict) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def run(self, *args, **kwargs):
        string = "\n".join([message["content"] for message in kwargs["messages"]])
        #encoding = tiktoken.encoding_for_model(self.model_type.value)
        #num_prompt_tokens = len(encoding.encode(string))
        #gap_between_send_receive = 15 * len(kwargs["messages"])
        #num_prompt_tokens += gap_between_send_receive

        if RUN_LOCALLY:
            client = LocalAI(
                base_url=BASE_URL,
                decentralize=DECENTRALIZE,
            )

            # numbers in this map are more dependent on the host's hardware rather than the model itself
            num_max_token_map = {
                'openhermes': 4096,
                'llama2-uncensored:7b': 4096,
            }

            # ERR: Could not automatically map llama2-uncensored:7b to a tokeniser.
            #      Please use `tiktok.get_encoding` to explicitly get the tokeniser you expect
            # We do not have to set this, it's just a tokenizing agent and estimates are enough.
            # note: Enum entry was added, this did not fix the underlying issue
            #self.model_type = ModelType('llama2-uncensored:7b')

            #num_max_token = num_max_token_map['openhermes']
            #num_max_completion_tokens = num_max_token - num_prompt_tokens
            self.model_config_dict['max_tokens'] = config.tokenEstimator(string)

            response = config.runManualGenerate(string, config.tokenEstimator(string))

            # note: removed cost calculation, completely unnecessary for locally run models

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\n".format(
                    response.get('prompt_eval_count'), response.get('eval_count'), response['response']))

            # todo: response is not an instance of ChatCompletion, causing this error to trigger
            #       either remove any constraints, or perfectly imitate ChatCompletion and overwrite typename
            # for now opting-in for the former option - removing constrains while recreating the necessary parts
            if not isinstance(response, LocalChatCompletion):
                raise RuntimeError("Unexpected return from ollama API")
            return response
        elif openai_new_api:
            # Experimental, add base_url

            response = config.runManualGenerate(string, config.tokenEstimator(string))

            cost = prompt_cost(
                self.model_type.value,
                num_prompt_tokens=response.get('prompt_eval_count'),
                num_completion_tokens=response.get('eval_count')
            )

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response.get('prompt_eval_count'), response.get('eval_count'),
                    int(response.get('prompt_eval_count')) + int(response.get('eval_count')), cost))
            if not isinstance(response, ChatCompletion):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response
        else:
            response = config.runManualGenerate(string, config.tokenEstimator(string))

            cost = prompt_cost(
                self.model_type.value,
                num_prompt_tokens=response.get('prompt_eval_count'),
                num_completion_tokens=response.get('eval_count')
            )

            log_visualize(
                "**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\ncost: ${:.6f}\n".format(
                    response.get('prompt_eval_count'), response.get('eval_count'),
                    int(response.get('prompt_eval_count')) + int(response.get('eval_count')), cost))
            if not isinstance(response, Dict):
                raise RuntimeError("Unexpected return from OpenAI API")
            return response


class StubModel(ModelBackend):
    r"""A dummy model used for unit tests."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        ARBITRARY_STRING = "Lorem Ipsum"

        return dict(
            id="stub_model_id",
            usage=dict(),
            choices=[
                dict(finish_reason="stop",
                     message=dict(content=ARBITRARY_STRING, role="assistant"))
            ],
        )


class ModelFactory:
    r"""Factory of backend models.

    Raises:
        ValueError: in case the provided model type is unknown.
    """

    @staticmethod
    def create(model_type: ModelType, model_config_dict: Dict) -> ModelBackend:
        default_model_type = ModelType.STUB

        model_class = StubModel

        if model_type is None:
            model_type = default_model_type

        # log_visualize("Model Type: {}".format(model_type))
        inst = model_class(model_type, model_config_dict)
        return inst
