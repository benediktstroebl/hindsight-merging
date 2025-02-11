from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class Response:
    response: Union[str, list[str]]
    num_completion_tokens: Union[int, list[int]]
    num_input_tokens: int
    index: Optional[int] = None

    @classmethod
    def from_ray_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a rayllm response.

        Args:
            response: Ray response object containing generated text and token information

        Returns:
            Responses: New instance initialized with Ray response data
        """
        if isinstance(response["generated_text"], list):
            # n > 1 samples
            num_completion_tokens = [
                int(response["num_generated_tokens"][i])
                for i in range(len(response["num_generated_tokens"]))
            ]
        else:
            num_completion_tokens = int(response["num_generated_tokens"])
        return cls(
            response=response["generated_text"],
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=int(response["num_input_tokens"]),
            index=response["index"],
        )

    @classmethod
    def from_openai_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from an OpenAI response.

        Args:
            response: OpenAI response object containing message content and token information

        Returns:
            Responses: New instance initialized with OpenAI response data
        """
        # TODO: allow for multiple samples
        return cls(
            response=response.choices[0].message.content,
            num_completion_tokens=response.usage.completion_tokens,
            num_input_tokens=response.usage.prompt_tokens,
        )

    @classmethod
    def from_vllm_response(cls, response) -> "Response":
        """
        Factory method to create a Response instance from a vLLM response.

        Args:
            response: vLLM response object containing output text and token information

        Returns:
            Responses: New instance initialized with vLLM response data
        """
        response_text = (
            [response.outputs[i].text for i in range(len(response.outputs))]
            if len(response.outputs) > 1
            else response.outputs[0].text
        )
        num_completion_tokens = (
            [len(s) for s in response_text]
            if not isinstance(response_text, str)
            else len(response_text)
        )
        return cls(
            response=response_text,
            num_completion_tokens=num_completion_tokens,
            num_input_tokens=len(response.prompt_token_ids),
        )
