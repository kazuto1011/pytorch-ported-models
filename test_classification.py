import click
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from models.classification.xception import XceptionV1
from models.classification.resnet import resnet50


@click.command()
@click.option("-i", "--image-path", type=str, required=True)
def main(image_path):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    classes = list()
    with open("data/imagenet/synsets.txt") as lines:
        for line in lines:
            classes.append(line.strip())
    metadata = {}
    with open("data/imagenet/metadata.txt") as lines:
        for line in lines:
            wordnetid, desc = line.split("\t")
            metadata[wordnetid] = desc.strip()

    model = XceptionV1(n_classes=1000)
    model.load_from_keras()
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    image = torch.FloatTensor(np.asarray(image))
    image = image.permute(2, 0, 1)[None, ...]
    # image = F.interpolate(image, size=(299, 299), mode="bilinear", align_corners=False)
    image /= 127.5
    image -= 1.0
    image = image.to(device)
    logit = model(image)
    probs = F.softmax(logit, dim=1)

    print("[Score] [Label]")
    for prob in probs:
        scores, ids = prob.sort(dim=0, descending=True)
        scores = scores.cpu().numpy()
        for score, idx in list(zip(scores, ids))[:5]:
            print("{:.5f}".format(score), metadata[classes[idx]])


if __name__ == "__main__":
    main()
