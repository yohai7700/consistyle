from diffusers import DiffusionPipeline
import torch

print(torch.cuda.is_available())

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0").to('cpu')

prompt = "realistic photo of dog on the beach"
seed = 40
generator = torch.Generator(device="cpu").manual_seed(seed)  # Set your seed here
num_steps = 10
images = pipeline(prompt, generator=generator, num_inference_steps=num_steps).images
for i in range(len(images)):
    images[i].save(f'./image-dog2-forest-{num_steps}.png')
print("finished")