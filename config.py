import time
import ollama
import base64
import io
from PIL import Image

MODEL_DEFAULT="deepseek-r1:14b"
MODEL_VISION_DEFAULT="llama3.2-vision:11b"

def tokenEstimator(incomingText = None):
    estimated_tokens = len(incomingText) / 3.5

    # Round up to the nearest power of 2 (or specified values)
    possible_values = [2048, 4096, 8192, 16384, 20480]  # Up to 20k

    for value in possible_values:
        if estimated_tokens <= value:
            return value

    if incomingText is None:
        return 20480

    return 20480

def runManualGenerate(text, token_count = 2046, model = MODEL_DEFAULT):

    if model == "" or model is None:
        model = MODEL_DEFAULT

    print("Running:" + text)

    try:
        # Time the model loading
        load_start = time.time()

        # Time the response generation
        response_start = time.time()
        ollama.Options.num_ctx = token_count
        cresponse = ollama.generate(model, text)
        response_end = time.time()
        response_time = response_end - response_start

        load_end = time.time()
        load_time = load_end - load_start

        # Print the results
        print(f"Model: {model}")
        print(f"Question: {text}")
        print(f"Response: {cresponse['response']}")
        print(f"Response full: {cresponse}")
        print(f"Load Time: {load_time:.2f} seconds")
        print(f"Response Time: {response_time:.2f} seconds")
        print(f"prompt_eval_count: {cresponse.get('prompt_eval_count')} ")
        print(f"eval_count: {cresponse.get('eval_count')} ")
        print("-" * 20)

        return cresponse

    except Exception as error:
        print(f"Error generating response for model {model}: {error}")

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
