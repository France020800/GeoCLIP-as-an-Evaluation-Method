import os
import ast
from google import genai
from google.genai import types

def generate_dictionary(text_prompt, API_KEY=None):
    if not API_KEY:
        raise ValueError('API_KEY must be provided.')

    client = genai.Client(
        api_key=API_KEY,
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=text_prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        response_mime_type="text/plain",
    )

    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text

    try:
        result = ast.literal_eval(response)
        if not isinstance(result, dict):
            raise ValueError("Response is not a dictionary")
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse response as dictionary: {e}")

if __name__ == "__main__":
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    dict_size = 10

    prompt = (
        f"Generate a valid Python dictionary (not code block) with exactly {dict_size} entries, where the keys are prompts to generate city images, "
        "and the value is the GPS location of the city. Only output the dictionary."
    )
    city_dict = generate_dictionary(prompt, API_KEY=GEMINI_API_KEY)
    print(len(city_dict))