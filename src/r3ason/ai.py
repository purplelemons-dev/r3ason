from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from json import loads, dumps
from pydantic import BaseModel
from timeit import default_timer as dt


class Step(BaseModel):
    header: str
    details: str
    number: int

    def serialize(self):
        return {"header": self.header, "details": self.details, "number": self.number}


class Reasoning(BaseModel):
    interpretation: str
    steps: list[Step]
    revisions: list[Step]
    final_answer: str


class AI:

    def __init__(self, api_key: str = None, org_id: str = None):
        self.model = "gpt-4o-2024-08-06"
        self.openai = OpenAI(api_key=api_key, organization=org_id)
        self.messages: list[ChatCompletionMessageParam] = []

    @property
    def reasoning_sys_message(self):
        return {
            "role": "system",
            "content": 'Your response must be in JSON format. You will interpret the user\'s prompt \
                and consider contextual factors and attempt to mitigate ambiguity. You will then generate \
                a list of steps that you would need to take to answer the question or complete the task, \
                plus a final answer. Each step header should begin with a gerund (e.g., "Thinking about [...]", \
                "Considering [...]") You will consider all possible factors and how each step relates to the \
                others. Revisions are needed if you need to make a choice that you did not settle on yet so \
                that you could proceed with reasoning. The user will not see the steps that you reasoning \
                about, so you should address each step with it\'s own step-by-step instructions in your final answer.',
        }

    def generate_text(self, message: str, show_output: bool = False) -> str:
        self.messages.append({"role": "user", "content": message})
        # Generate reasoning steps
        st = dt()
        if show_output:
            response = ""
            with self.openai.beta.chat.completions.stream(
                model=self.model,
                messages=[self.reasoning_sys_message, *self.messages],
                response_format=Reasoning,
            ) as stream:
                times = []
                for event in stream:
                    if event.type == "content.delta":
                        delta_time = dt()
                        content = event.delta
                        print(content, flush=True, end="")
                        response += content
                        times.append(dt() - delta_time)

            response = Reasoning(**loads(response))
            avg_time = sum(times) / len(times) if times else -1
        else:
            avg_time = -1
            response = Reasoning(
                **loads(
                    self.openai.beta.chat.completions.parse(
                        model=self.model,
                        messages=[self.reasoning_sys_message, *self.messages],
                        response_format=Reasoning,
                    )
                    .choices[0]
                    .message.content
                )
            )

        end = dt() - st

        self.messages.append({"role": "assistant", "content": response.final_answer})

        return (
            response.interpretation,
            "\n".join(
                f"{idx+1}. {dumps(step.serialize(), indent = 2)}"
                for idx, step in enumerate(response.steps)
            ),
            "\n".join(
                f"{idx+1}. {dumps(revision.serialize(), indent = 2)}"
                for idx, revision in enumerate(response.revisions)
            ),
            response.final_answer,
            f"Time taken: {end:.2f}s"
            + (f" (Avg: {avg_time*1000:.3f}ms)" if avg_time > 0 else ""),
        )


if __name__ == "__main__":
    ai = AI()
    print(
        "\n\n".join(
            ai.generate_text(
                "Create a website (python or sveltekit) where the user registers a passkey and can then write data (e.g., small text files) into local storage. I have plans to upgrade and let users choose between storing the encrypted data locally or in the cloud, so write the storage api to be flexible.",
                show_output=True,
            )
        )
    )
