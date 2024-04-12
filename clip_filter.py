import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# def clip_filter(model, preprocess, img_path, text, device):
def clip_filter(img_path, text):

    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    text = clip.tokenize([text]).to(device)


    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        # logits_per_image, logits_per_text = model(image, text)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

    print("Cosine similarity:", cosine_similarity.item())
    return cosine_similarity.item()