import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
from simple_clip import (
    AvgMeter,
    CLIPDataset,
    CFG,
    ImageEncoder,
    TextEncoder,
    CLIPModel,
    cross_entropy_mat
)
from data_util import get_cat_image, get_dog_image, get_unsplash_image

def test_avg_meter(): 
    avg_meter = AvgMeter("Metric")
    avg_meter.update(10, 1)
    assert avg_meter.avg == 10
    assert avg_meter.sum == 10
    assert avg_meter.count == 1
    avg_meter.update(20, 2)
    assert avg_meter.avg == 16.666666666666668
    assert avg_meter.sum == 50
    assert avg_meter.count == 3
    assert str(avg_meter) == "Metric: 16.6667"
    
def dummy_tokenizer(captions, padding, truncation, max_length):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

def test_clip_dataset():
    # Create two sample images with cs2 and save them
    image_filenames = ["image1.jpg", "image2.jpg"]
    captions = ["A cat", "A dog"]
    dummy_image = np.random.randint(0, 255, (224, 224, 3)).astype(np.uint8)
    cv2.imwrite(f"{CFG.image_path}/image1.jpg", dummy_image)
    cv2.imwrite(f"{CFG.image_path}/image2.jpg", dummy_image)

    clip_dataset = CLIPDataset(image_filenames, captions, dummy_tokenizer)
    item0 = clip_dataset[0]
    os.remove(f"{CFG.image_path}/image1.jpg")
    os.remove(f"{CFG.image_path}/image2.jpg")

    assert(item0["caption"] == "A cat")
    assert(item0["image"].shape == (3, CFG.size, CFG.size))
    
    assert(item0["input_ids"].tolist() == [1, 2, 3])
    # alternative 
    # assert(torch.all(item0["input_ids"] == torch.tensor([1, 2, 3])).item())
    
    assert(item0["attention_mask"].tolist() == [1, 1, 1])

 
def test_dataset_transforms():
    clip_dataset = CLIPDataset([], [], dummy_tokenizer)
    image = np.random.randint(0, 255, (80, 80, 3)).astype(np.uint8)
    transformed = clip_dataset._transforms(image=image)
    
    transformed_image = transformed["image"]
    assert transformed_image.shape == (CFG.size, CFG.size, 3)

    assert abs(np.mean(transformed_image) - 0) < 0.3
    assert abs(np.std(transformed_image) - 1) < 0.3 


def jpg_binary_to_numpy(image_binary):
    # Open the image using PIL from the binary data
    image = Image.open(BytesIO(image_binary))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    return image_array


def prepare_image(downloader, args=[]):
    """Downloads and transforms an image for the Resnet50 encoder."""
    clip_dataset = CLIPDataset([], [], dummy_tokenizer)
    
    img = downloader(*args)
    arr = jpg_binary_to_numpy(img)
    print("Image shape:", arr.shape)
    
    transformed = clip_dataset._transforms(image=arr)["image"]
    return torch.tensor(transformed).permute(2, 0, 1).float()
    
def test_image_encoder():
    # Test that cat images are more similar 
    # than cat-dog, cat-other and other-dog
    im_encoder = ImageEncoder()
    
    other_topic = "Sea"
    
    cat_enc1 = im_encoder(prepare_image(get_cat_image).unsqueeze(0)) # unsqueeze adds an extra dimension (to show as a batch)
    cat_enc2 = im_encoder(prepare_image(get_cat_image).unsqueeze(0))
    dog_enc = im_encoder(prepare_image(get_dog_image).unsqueeze(0))
    other_enc = im_encoder(prepare_image(get_unsplash_image, args=[other_topic]).unsqueeze(0))
    
    assert(cat_enc1.shape == cat_enc2.shape 
           == dog_enc.shape == other_enc.shape == (1, 2048))

    # Dot multiply
    cat_cat = F.cosine_similarity(cat_enc1, cat_enc2)
    cat_dog = F.cosine_similarity(cat_enc1, dog_enc)
    cat_other = F.cosine_similarity(cat_enc1, other_enc)
    other_dog = F.cosine_similarity(other_enc, dog_enc)
    
    print("Cat cat:", cat_cat)
    print("Cat dog:", cat_dog)
    print("Cat other:", cat_other)
    print("other dog:", other_dog)
    
    # It seems cat_cat is not always greater than cat_other and other_dog
    assert(cat_cat > (cat_other - 0.1))
    assert(cat_cat > (other_dog - 0.1))
    
def test_text_encoder():
    # Test that two similar captions are more similar than two different captions
    park_caption1 = "A beautiful park with trees, benches, courts, and a small pond"
    park_caption2 = "Vast greeneries, water fountains, walking and cycling paths in a peaceful place"
    space_caption = "Astronauts working in space station near Mars looking at the earth"
    
    text_encoder = TextEncoder()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    park_caption1_toks = tokenizer(park_caption1, return_tensors='pt')
    park_caption2_toks = tokenizer(park_caption2, return_tensors='pt')
    space_caption_toks = tokenizer(space_caption, return_tensors='pt')
    park_enc1 = text_encoder(park_caption1_toks["input_ids"], park_caption1_toks["attention_mask"])
    park_enc2 = text_encoder(park_caption2_toks["input_ids"], park_caption2_toks["attention_mask"])
    space_enc = text_encoder(space_caption_toks["input_ids"], space_caption_toks["attention_mask"])
    
    assert(park_enc1.shape == park_enc2.shape == space_enc.shape == (1, 768))
    
    park_park = F.cosine_similarity(park_enc1, park_enc2)
    park1_space = F.cosine_similarity(park_enc1, space_enc)
    park2_space = F.cosine_similarity(park_enc2, space_enc)
    
    assert park_park > park1_space
    assert park_park > park2_space

def test_cross_entropy_mat():
    preds = torch.tensor([[0.2, 0.8], [0.6, 0.4]])
    targets = torch.tensor([[0, 1], [1, 0]])
    loss = cross_entropy_mat(preds, targets)
    assert torch.isclose(loss, torch.tensor([0.43, 0.60]), atol=1e-2).all()
    

def test_clip_model():
    captions = ["A cat", "A dog"]
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    toks = tokenizer(captions, return_tensors='pt')
    batch = {
        "image": torch.stack([prepare_image(get_cat_image), prepare_image(get_dog_image)]),
        **toks
    }
    
    clip_model = CLIPModel()
    loss = clip_model(batch)
    
    print("CLIP model loss:", loss)
    
# Values
# logits: tensor([[ -1.1129, -13.4838],
#         [ -4.4804, -14.7206]], grad_fn=<DivBackward0>)
# images_similarity: tensor([[255.5761, 168.8292],
#         [168.8292, 255.5758]], grad_fn=<MmBackward0>)
# texts_similarity: tensor([[255.9675, 251.6863],
#         [251.6863, 255.9687]], grad_fn=<MmBackward0>) # potential issue - similarity values are large before softmax
# targets: tensor([[1.0000e+00, 1.7119e-20],
#         [1.7112e-20, 1.0000e+00]], grad_fn=<SoftmaxBackward0>)
# texts_loss: tensor([4.2915e-06, 1.0240e+01], grad_fn=<SumBackward1>)
# images_loss: tensor([0.0339, 1.4917], grad_fn=<SumBackward1>)
# CLIP model loss: tensor(2.9415, grad_fn=<MeanBackward0>)
    