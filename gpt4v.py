from openai import OpenAI

import base64
import requests
from PIL import Image
import io

from torchvision import transforms

# OpenAI API Key
api_key = "Replace with your own OPENAI KEY."

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def encode_tensor_image(tensor):
    # print(tensor.shape)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)  
    # toPIL = transforms.ToPILImage() 
    # image = toPIL(tensor)
    tensor = tensor.squeeze(0).permute(1, 2, 0)
    image = Image.fromarray(tensor.mul(255).byte().numpy()).convert('RGB')


    buffer = io.BytesIO()
    buffers='./test.jpg'
    image.save(buffer, format="JPEG")  
    image.save(buffers, format="JPEG")  
    buffer.seek(0)

    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return image_base64


def gpt4v_observe(image_tensor, text_prompt):

  # Path to your image
  # image_path = "Imagenet/train/n01440764/n01440764_39.JPEG"

  # Getting the base64 string
  base64_image = encode_tensor_image(image_tensor)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": text_prompt 
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
  }

  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

  return response.json()


