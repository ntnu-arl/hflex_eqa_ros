# BSD 3-Clause License

# Copyright (c) 2026, NTNU Autonomous Robots Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
"""OpenAI client for vlms_python."""

import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image


@dataclass
class OpenAIClientConfig:
    """Configuration for the OpenAI client."""

    model: str = "gpt-4o"
    system_prompt: str = "You are a vision model that detects objects."


class OpenAIClient:
    """Client for OpenAI API to generate navigation or vision-based prompts."""

    def __init__(self, config: OpenAIClientConfig) -> None:
        """Initialize OpenAI client with model, max tokens, and system prompt."""
        self.config = config
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=self.api_key)

    def _encode_image(self, image: str | np.ndarray) -> str:
        """
        Convert an image (path or np.ndarray) to base64 string for API input.
        :param image: Either a file path (str) or an
                      image array (np.ndarray, RGB or BGR).
        :return: Base64-encoded string representation
                 of the image (JPEG format).
        """
        if isinstance(image, str):
            # If a file path is passed
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        elif isinstance(image, np.ndarray):
            # Encode to JPEG in memory
            pil_img = Image.fromarray(image.astype(np.uint8))
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded

        else:
            raise TypeError("Input must be a file path or a NumPy image array.")

    def _parse_json_from_model_output(self, raw_text: str):
        """
        Extracts and parses JSON content from a model response that
        may contain Markdown code fences.
        Works even if the model wraps the JSON in ```json ... ``` or ``` ... ```.
        """
        if not raw_text:
            return None

        # Remove Markdown code fences (```json ... ``` or ``` ... ```)
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL
        )

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Sometimes the model returns multiple objects or text + json:
            # try to extract the first JSON block
            match = re.search(r"\{.*\}|\[.*\]", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            raise ValueError(
                f"Failed to parse JSON from model output: {e}\nRaw text:\n{raw_text}"
            ) from None

    def inference(
        self,
        prompt: str,
        images: list[str] | list[np.ndarray] | None = None,
        images_descriptions: list[str] | None = None,
        log: bool = False,
    ) -> tuple[dict, bool]:
        """
        Use OpenAI’s VLM.
        :param prompt: Text prompt.
        :param image: Image input as file path or np.ndarray.
        :param images_descriptions: Optional descriptions for each image, if needed.
        :param log: Whether to print debug logs.
        :return: A tuple (result_dict, success_flag)
        """
        try:
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                    ],
                },
            ]
            if images is not None and len(images) > 0:
                for i, image in enumerate(images):
                    base64_image = self._encode_image(image)
                    messages[1]["content"].append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    )
                    if images_descriptions and i < len(images_descriptions):
                        messages[1]["content"].append(
                            {
                                "type": "input_text",
                                "text": images_descriptions[i],
                            }
                        )

            response = self.client.responses.create(
                model=self.config.model,
                input=messages,
            )

            raw_text = response.output_text.strip()

            if log:
                print(f"[OpenAIClient] Prompt: {prompt}")
                print(f"[OpenAIClient] Response: {raw_text}")

            # Try parsing structured JSON-like output
            try:
                parsed = self._parse_json_from_model_output(raw_text)
                return parsed, True
            except Exception as e:
                # If model didn't return valid JSON, handle gracefully
                print(
                    f"[OpenAIClient] Warning: Failed to parse "
                    f"JSON from model output: {e}"
                )
                return {"error": "Failed to parse JSON"}, False

        except Exception as e:
            print(f"[OpenAIClient] Error during object detection: {e}")
            return {"error": str(e)}, False
