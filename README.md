# Detect Advertisement Images

## Usage
--------------

```
pip install adlike
```

```python3
import requests
from PIL import Image

from adlike import ad_openai_clip_vitl_patch14_336

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = ad_openai_clip_vitl_patch14_336()

url = "https://literacyideas.com/wp-content/uploads/2021/08/christmas_advertising.jpg"
image = Image.open(requests.get(url, stream=True).raw)


with torch.no_grad():
    image = preprocess(img).to(device).unsqueeze(0)
    probs = model(image)

print("Label probs:", probs)  # prints: tensor([[0.9753]])

```
