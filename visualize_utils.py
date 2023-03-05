import gc
from pathlib import Path

import PIL.Image
import numpy as np

import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from matplotlib import pyplot as plt
from segmentation_models_pytorch.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms



def show_distribution(target, output_predictions, image_name, loss):
    df = pd.DataFrame()
    df["target"] = (target).clone().cpu().numpy().flatten()
    df["output_predictions"] = (
        output_predictions.clone().detach().cpu().flatten().numpy()
    )
    plt.figure()
    sns.histplot(df)
    del df
    gc.collect()
    plt.savefig(
        f'sat_results/distribution/distribution_{Path(image_name).with_suffix("")}_loss_{loss}.jpg'
    )
    plt.close("all")


def set_vis_tool():
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return palette, colors


def show_results(
    target, output_predictions, image_path, loss, global_step,writer: SummaryWriter = None
):
    image_name = Path(image_path).name
    pred_array = output_predictions.argmax(0).byte().cpu().numpy()
    pred_image = Image.fromarray(pred_array)
    trg = target.byte().cpu().numpy()
    target_array = np.zeros((trg.shape[1:])) + 6
    target_array = target_array.astype(np.uint8)
    palette, colors = set_vis_tool()
    for count, i in enumerate(trg):
        target_array[i == 1] = count
    sat_image = PIL.Image.open(Path("data/geo/train") / image_name)
    pred_image.putpalette(colors)
    plt.subplot(131)
    plt.imshow(pred_image)
    plt.xticks([])
    plt.yticks([])
    plt.title("pred image")

    target_image = Image.fromarray(target_array, mode="P")
    target_image.putpalette(colors)
    plt.subplot(132)
    plt.xticks([])
    plt.yticks([])
    plt.title("target image image")
    plt.imshow(target_image)

    plt.subplot(133)
    plt.xticks([])
    plt.yticks([])
    plt.title("satellite image")
    plt.imshow(sat_image)
    current_loss = DiceLoss(mode="multiclass", from_logits=False)(
        output_predictions.unsqueeze(0).cpu(),
        transforms.ToTensor()(target_array).type(torch.int64),
    )
    plt.suptitle(
        f"{image_name}  batch loss: {round(float(loss.detach()), 4)} \n current loss :{round(float(current_loss.detach()), 4)}"
    )
    img_path = f'sat_results/{Path(image_name).with_suffix("")}_loss_{loss}.jpg'
    plt.savefig(img_path)
    # plt.show()
    if writer:
        image = PIL.Image.open(img_path)
        image = transforms.ToTensor()(image)
        writer.add_image(f"example image {image_name}", image, global_step)
    plt.close("all")
