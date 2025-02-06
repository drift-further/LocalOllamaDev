import time
import ollama
import base64
import io
from PIL import Image

from dataclasses import dataclass
import json
import re
from dataclasses import dataclass
from typing import List, Optional

MODEL_DEFAULT="deepseek-r1:14b"
MODEL_VISION_DEFAULT="llama3.2-vision:11b"

MAX_TOKENS = 20480

def tokenEstimator(incomingText = None):
    estimated_tokens = len(incomingText) / 3.5

    # Round up to the nearest power of 2 (or specified values)
    possible_values = [2048, 4096, 8192, 16384, MAX_TOKENS]  # Up to 20k

    for value in possible_values:
        if estimated_tokens <= value:
            return value

    if incomingText is None:
        return MAX_TOKENS

    return MAX_TOKENS

@dataclass
class Message:
    role: str
    content: str


@dataclass
class OllamaResponse:
    model: str
    created_at: str
    done: bool
    done_reason: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    prompt_eval_duration: int
    eval_count: int
    eval_duration: int
    message: Message
    images: Optional[List]  # Adjust type if needed
    tool_calls: Optional[List] # Adjust type if needed


def parse_ollama_response(response_string: str) -> OllamaResponse:

    # Regular expression to capture key-value pairs
    pattern = r"(\w+)=([^'\s]+|'[^']+')"  # Handles quoted and unquoted values
    matches = re.findall(pattern, response_string)
    data = dict(matches)

    # Convert some values to appropriate types
    data['done'] = data['done'] == 'True'
    data['total_duration'] = int(data['total_duration'])
    data['load_duration'] = int(data['load_duration'])
    data['prompt_eval_count'] = int(data['prompt_eval_count'])
    data['prompt_eval_duration'] = int(data['prompt_eval_duration'])
    data['eval_count'] = int(data['eval_count'])
    data['eval_duration'] = int(data['eval_duration'])


    # for message in data['message']:
    #     role = message_match.group(1)
    #     content = message_match.group(2)
    #     message = Message(role=role, content=content)
    # else:
    #     raise NotImplementedError

    message_start = response_string.find("message=Message(") + len("message=Message(")
    message_end = response_string.find(")", message_start)
    message_str = response_string[message_start:message_end]

    # Split the message string by commas, but be careful about commas *inside* the content
    message_parts = []
    current_part = ""
    open_quotes = 0
    for char in message_str:
        if char == "'" or char == '"':
            open_quotes = 1 - open_quotes  # simple count for even or odd number of quotes
        if char == "," and open_quotes == 0:
            message_parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    message_parts.append(current_part.strip())  # append last part

    role = ""
    content = ""
    for part in message_parts:
        if part.startswith("role="):
            role = part[len("role="):].strip("'")
        elif part.startswith("content="):
            content = part[len("content="):].strip('"').replace("\\n", "\n").replace("\\'", "'")

    if not role or not content:
        raise ValueError("Could not parse message string: role or content missing")

    message = Message(role=role, content=content)

    print("********* RESP MESSAGE: " + content)

    #Handle images and tool_calls (if present)
    images_match = re.search(r"images=([^,]+)", response_string)
    images = eval(images_match.group(1)) if images_match else None

    tool_calls_match = re.search(r"tool_calls=([^)]+)", response_string)
    tool_calls = eval(tool_calls_match.group(1)) if tool_calls_match else None

    return OllamaResponse(
        model=data['model'],
        created_at=data['created_at'],
        done=data['done'],
        done_reason=data['done_reason'],
        total_duration=data['total_duration'],
        load_duration=data['load_duration'],
        prompt_eval_count=data['prompt_eval_count'],
        prompt_eval_duration=data['prompt_eval_duration'],
        eval_count=data['eval_count'],
        eval_duration=data['eval_duration'],
        message=message,
        images=images,
        tool_calls=tool_calls
    )

def runManualGenerate(text, token_count = 2046, model = MODEL_DEFAULT):

    print("Running:" + str(text))

    if model == "" or model is None:
        model = MODEL_DEFAULT

    try:
        # Time the model loading
        load_start = time.time()

        # Time the response generation
        response_start = time.time()
        ollama.Options.num_ctx = token_count
        print("starting...")
        #cresponse = ollama.generate(model, text)
        cresponse = ollama.chat(model, text)
        print("crspn" + str(cresponse))
        ollama_data = parse_ollama_response(str(cresponse))
        print("ollama_data" + str(ollama_data))
        response_end = time.time()
        response_time = response_end - response_start

        load_end = time.time()
        load_time = load_end - load_start

        # Print the results
        print(f"Model: {model}")
        print(f"Question: {text}")
        print(f"Response: {str(ollama_data.message)}")
        print(f"Load Time: {load_time:.2f} seconds")
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"prompt_eval_count: {ollama_data.prompt_eval_count} ")
        print(f"eval_count: {ollama_data.eval_count} ")
        print("-" * 20)

        return cresponse

    except Exception as error:
        print(f"Error generating response for model {model}: {error}")
        raise error

def image_to_base64(image_path):
    # Open the image file
    with Image.open(image_path) as img:
        # Create a BytesIO object to hold the image data
        buffered = io.BytesIO()
        # Save the image to the BytesIO object in a specific format (e.g., PNG)
        img.save(buffered, format="PNG")
        # Get the byte data from the BytesIO object
        img_bytes = buffered.getvalue()
        # Encode the byte data to base64
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

def runImageQuestion(text, imagePath, model = MODEL_VISION_DEFAULT):

    try:
        # Time the model loading
        load_start = time.time()

        # Time the response generation
        response_start = time.time()
        cresponse = ollama.chat(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": text,
                            "images": [image_to_base64(imagePath)]
                        }],
                    )
        response_end = time.time()
        response_time = response_end - response_start

        load_end = time.time()
        load_time = load_end - load_start

        # Print the results
        # model='llama3.2-vision:11b'
        # created_at='2025-02-05T17:10:13.820110828Z'
        # done=True
        # done_reason='stop'
        # total_duration=105398596899
        # load_duration=6329356510
        # prompt_eval_count=25
        # prompt_eval_duration=95329000000
        # eval_count=22
        # eval_duration=3674000000
        # message=Message(role='assistant', content='The text in the image is "HOW TO COMBINE TEXT AND IMAGE IN ELEARNING DESIGN."', images=None, tool_calls=None)

        messageText = cresponse['message']['content'].strip()

        cresponse.message = messageText

        print(f"Model: {model}")
        print(f"Question: {text}")
        print(f"Response: {messageText}")
        print(f"Response full: {cresponse}")
        print(f"Load Time: {load_time:.2f} seconds")
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"prompt_eval_count: {cresponse.get('prompt_eval_count')} ")
        print(f"eval_count: {cresponse.get('eval_count')} ")
        print("-" * 20)

        return cresponse

    except Exception as error:
        print(f"Error generating response for model {model}: {error}")
