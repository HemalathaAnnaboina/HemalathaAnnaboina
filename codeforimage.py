from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import IPython.display as display

# Load the Stable Diffusion model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Move to GPU if available
if torch.cuda.is_available():
    pipe.to("cuda")
else:
    print("Running on CPU.")

# Function to generate an image from a text prompt
def generate_image_from_text(prompt):
    image = pipe(prompt).images[0]
    return image

# Get user input for the image prompt
prompt = input("Enter a description for the image: ")

# Generate the image and show it in Colab
generated_image = generate_image_from_text(prompt)

# Display the image in Colab
display.display(generated_image)