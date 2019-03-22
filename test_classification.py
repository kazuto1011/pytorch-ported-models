import click
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

import models.classification as models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@click.command()
@click.option("-a", "--arch", type=click.Choice(model_names), required=True)
@click.option("-i", "--image-path", type=str, required=True)
def main(arch, image_path):

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

    if arch == "xception_v1":
        n_classes = 1000
        idx_offset = 0
    else:
        n_classes = 1001
        idx_offset = 1

    model = models.__dict__[arch](n_classes=n_classes, pretrained=True)
    model.eval()
    model.to(device)

    image = Image.open(image_path).convert("RGB")
    image = torch.FloatTensor(np.asarray(image)).to(device)
    image -= model.mean.to(device)
    image /= model.std.to(device)
    image = image.permute(2, 0, 1)[None, ...]
    image = F.interpolate(
        image, size=model.image_shape, mode="bilinear", align_corners=False
    )
    logit = model(image)
    probs = F.softmax(logit, dim=1)

    print("[Score] [Label]")
    for prob in probs:
        scores, ids = prob.sort(dim=0, descending=True)
        scores = scores.cpu().numpy()
        for score, idx in list(zip(scores, ids))[:5]:
            print("{:.5f}".format(score), metadata[classes[idx + idx_offset]])


if __name__ == "__main__":
    main()
