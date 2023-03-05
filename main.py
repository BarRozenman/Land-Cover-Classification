from pprint import pprint

from glob import glob
import warnings

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from pathlib import Path

from personal_utils.flags import flags
from segmentation_models_pytorch.losses import DiceLoss
import GPUtil
import pandas as pd
import torch
from PIL import Image
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A

from dataloader import CustomImageDataset
from model import Net
from visualize_utils import show_distribution, set_vis_tool, show_results

matplotlib.use("TkAgg")

torch.cuda.empty_cache()
warnings.filterwarnings("ignore")


def load_image_and_mask(img_path, preprocessing_fn, rgb_mask, class_rgb, train=True):
    use_preprocessing_fn = False
    # mask = torch.cat([transforms.ToTensor()(cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel)) for x in mask])
    if train:
        over_size = (
            int(img_size[0] + img_size[0] / 10),
            int(img_size[1] + img_size[1] / 10),
        )
        mask = preprocess_mask_image(rgb_mask, class_rgb, over_size)
        pil_img = Image.open(img_path)  # .resize(over_size)
        _transform = [
            transforms.Resize(over_size, interpolation=InterpolationMode.NEAREST),
            # lambda x: preprocessing_fn((np.array(x))),
        ]
        aug_special = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomCrop(*img_size, p=1),
            ],
            additional_targets={"image": "image", "mask": "mask"},
        )
        aug_colors = A.Compose(
            [A.RandomBrightnessContrast(p=0.8), A.RandomGamma(p=0.8)]
        )
        comp = transforms.Compose(_transform)
        ans = aug_special(image=np.array(comp(pil_img)), mask=mask)
        ans["image"] = aug_colors(image=ans["image"])["image"]
        if use_preprocessing_fn:
            ans["image"] = preprocessing_fn(np.array(ans["image"]))
        img = transforms.ToTensor()(ans["image"])
        mask = transforms.ToTensor()(ans["mask"])
        if flags.debug:
            plt.subplot(121)
            plt.imshow(ans["image"])
            plt.subplot(122)
            plt.imshow(np.argmax(ans["mask"], -1))
            plt.show()
    else:
        mask = preprocess_mask_image(rgb_mask, class_rgb, image_size=img_size)
        mask = transforms.ToTensor()(mask)
        img = Image.open(img_path).resize(img_size)
        if use_preprocessing_fn:
            img = preprocessing_fn(np.array(img))
        img = transforms.ToTensor()(img)
    # img = comp(transforms.ToPILImage()(preprocessing_fn((np.array(pil_img)))))
    return img, mask


def read_class_dict():
    """
    :return: class_rgb, class_name_to_catg_int
    class_rgb:  dict has keys of class and values of rgb
    class_name_to_catg_int: dict has keys of int class and values of str class name

    """
    class_name_to_catg_int = {count: x["name"] for count, x in enumerate(class_dict)}
    class_rgb = []
    for i, dic in enumerate(class_dict):
        class_rgb.append({"class": i, "rgb": [dic["r"], dic["g"], dic["b"]]})
    return class_rgb, class_name_to_catg_int


def preprocess_mask_image(rgb_mask, class_rgb, image_size):
    kernel = np.ones((4, 4), np.uint8)
    # mask = np.empty((7, *img_size))
    mask = np.zeros((7, *image_size))
    for dd in class_rgb:
        rgb_mask[rgb_mask > 128] = 255
        rgb_mask[rgb_mask <= 128] = 0
        mask[dd["class"], np.all(rgb_mask == np.array(dd["rgb"]), 2)] = 1
    mask[6, np.all(mask == 0, 0)] = 1
    mask = np.concatenate(
        [(cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel))[..., None] for x in mask], -1
    )
    # mask = torch.cat([transforms.ToTensor()(cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel)) for x in mask])
    # plt.imshow(opening)
    # plt.show()
    # set([tuple(y) for x in rgb_mask.tolist() for y in x])
    # plt.imshow(rgb_mask)
    # plt.imshow(mask)
    # plt.imshow(rgb_mask[0,...])
    # plt.show()
    # set(mask.detach().numpy().flatten())
    return mask


def compute_total_variation_loss(img, weight):
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)


if __name__ == "__main__":
    writer = SummaryWriter(f"seg_runs/{flags.timestamp}")
    performance_metrics = []
    flags.debug = False
    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = [
        "urban_land",
        "agriculture_land",
        "rangeland",
        "forest_land",
        "water",
        "barren_land",
        "unknown",
    ]
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    root_dir = "data/geo"
    class_dict = pd.read_csv("data/geo/class_dict.csv").to_dict(orient="records")
    img_size = (int(224 * 1.5), int(224 * 1.5))
    img_size = (352, 352)
    palette, colors = set_vis_tool()
    run_test = True
    class_rgb, class_name_to_catg_int = read_class_dict()
    device = "cuda"
    batch_size = 2
    metadata_path = Path(root_dir) / "metadata.csv"
    img_dir_path = Path(root_dir)
    dataset_df = pd.read_csv(metadata_path)
    dataset_df = dataset_df[dataset_df["split"] == "train"].dropna(axis=0)
    train_df = dataset_df.sample(frac=0.8, random_state=200)
    test_df = dataset_df.drop(train_df.index)
    train_ds = CustomImageDataset(
        train_df,
        img_dir_path,
        load_image_and_mask,
        class_rgb,
        target_transform=preprocess_mask_image,
        img_size=img_size,
        preprocessing_fn=preprocessing_fn,
    )
    test_ds = CustomImageDataset(
        test_df,
        img_dir_path,
        load_image_and_mask,
        class_rgb,
        train=False,
        target_transform=preprocess_mask_image,
        img_size=img_size,
        preprocessing_fn=preprocessing_fn,

    )
    num_workers = 8
    train_data_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    # a = iter(train_data_loader)
    # b = next(a)
    # b = next(a)
    # b = next(a)

    # plt.figure()
    # plt.imshow(transforms.ToPILImage()(b['mask']))
    # plt.show()
    # inp = b['sat_image']
    # inp = transforms.Resize([300,300])(inp).unsqueeze(0)
    # # for i in a:
    # #     i
    # plt.imshow(transforms.ToPILImage()(b['mask']))
    # plt.show()

    # model = deeplabv3.deeplabv3_resnet50(pretrained=True).eval()
    # oo = model(load_image('/home/bar/projects/personal/Taranis/sheep.jpg').unsqueeze(0))['out'][0, ...]
    # plt.imshow(transforms.ToPILImage()(oo['out'][0,0,:,:]))
    # plt.show()
    # ann = torch.autograd.Variable(ann, requires_grad=False).to(device)
    # loss_func = CrossEntropyLoss(reduction='mean')
    files = glob("models/sat_model_epoch_*_batch_*.pt")
    epoch_num_from_file = 0
    batch_num_from_file = 0
    lr = 0.001
    if len(files) > 0:
        files = natsorted(files)
        last_run = files[-1]
        epoch_num_from_file = int(last_run.split("_")[3])
        batch_num_from_file = int(last_run.split("_")[5].split(".")[0])
        # net = torch.load(last_run)
        net = Net(device)
        loaded_data = torch.load(last_run)
        net.load_state_dict(loaded_data["state_dict"])
        opt = optim.AdamW(net.parameters(), lr=lr)
        opt.load_state_dict(loaded_data["optimizer"])
        # scheduler = ReduceLROnPlateau(opt, 'min',patience=1)
        # scheduler.load_state_dict(loaded_data['optimizer'])
    else:
        net = Net(device)
        opt = optim.AdamW(net.parameters(), lr=lr)
    lambda1 = lambda batch: 2 / (batch + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

    # scheduler = ReduceLROnPlateau(opt, 'min',patience=1)

    net = net.to(device)
    loss_func = DiceLoss(mode="multiclass", from_logits=False)
    # loss_func = MSELoss(mode="multiclass", from_logits=False)
    # scheduler = ReduceLROnPlateau(opt, 'min')
    cross_ = torch.nn.CrossEntropyLoss()
    lrs = []

    for epoch in range(9990):
        for batch_count, batch in tqdm(
            enumerate(train_data_loader), total=len(train_data_loader)
        ):
            if batch["sat_image"].dim() == 3:
                sat_img = batch["sat_image"].unsqueeze(0).to(device)
            else:
                sat_img = batch["sat_image"].to(device)
            torch.cuda.empty_cache()
            gt_mask = batch["mask"].to(device)
            pred_mask = net(sat_img)
            if flags.debug:
                gt_1 = torch.argmax(gt_mask[0, ...], 0).type(torch.float)
                pred_1 = torch.argmax(pred_mask[0, ...], 0).type(torch.float)
                gt_im = transforms.ToPILImage()(gt_1)
                pred_im = transforms.ToPILImage()(pred_1)
                tp, fp, fn, tn = smp.metrics.get_stats(
                    gt_1.long(), pred_1.long(), mode="multiclass", num_classes=7
                )

                plt.imshow(gt_im)
                plt.show()

            classes_loss = cross_(torch.softmax(pred_mask, 1), torch.argmax(gt_mask, 1))
            IoU_classes_loss = loss_func(pred_mask, torch.argmax(gt_mask, 1))
            classes_lambda = 0.25
            variation_lambda = variation_loss = 0
            loss = (
                classes_lambda * classes_loss
                + variation_loss * variation_lambda
                + IoU_classes_loss
            )  # + std_lambda * std_loss
            loss.backward()
            global_step = (epoch + 1) * (1 + batch_count)
            writer.add_scalar("loss", loss, global_step)
            writer.add_scalar(
                "classes loss", classes_lambda * classes_loss, global_step
            )
            writer.add_scalar("IoU classes  loss", IoU_classes_loss, global_step)
            tp, fp, fn, tn = smp.metrics.get_stats(
                torch.argmax(pred_mask, 1),
                torch.argmax(gt_mask, 1),
                mode="multiclass",
                num_classes=7,
            )
            print(loss)
            performance_metrics.append(
                {
                    "loss": loss,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }
            )
            if (batch_count + 1) % 12 == 0:
                opt.step()
                opt.zero_grad()
                scheduler.step()
            if (batch_count) % 20 == 0:
                tp = torch.cat([x["tp"] for x in performance_metrics])
                fp = torch.cat([x["fp"] for x in performance_metrics])
                fn = torch.cat([x["fn"] for x in performance_metrics])
                tn = torch.cat([x["tn"] for x in performance_metrics])
                performance_metrics = []
                per_image_iou = smp.metrics.iou_score(
                    tp, fp, fn, tn, reduction="micro-imagewise"
                )
                batch_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
                metrics = {
                    f"{global_step}_per_image_iou": per_image_iou,
                    f"{global_step}_batch_iou": batch_iou,
                }
                writer.add_scalar("test dataset_iou", batch_iou, global_step)
                GPUtil.showUtilization()
                pprint({"loss": float(loss.clone().cpu())})
                pprint({"metrics": metrics})

                # a = {'classes_loss': classes_lambda * classes_loss}
                # 'variation_loss': variation_loss * variation_lambda, }
                # 'std_loss': std_lambda * std_loss}
                # pprint([{k: float(v.clone().cpu())} for k, v in a.items()])
                example_image_idx = np.random.choice(np.arange(len(batch["sat_path"])))
                img_name = Path(batch["sat_path"][example_image_idx]).name
                show_results(
                    batch["mask"][example_image_idx, ...],
                    pred_mask[example_image_idx, ...],
                    batch["sat_path"][example_image_idx],
                    loss,
                    global_step
                )
                writer.add_scalar(
                    "current_learning rate",
                    opt.param_groups[0]["lr"],
                    global_step=global_step,
                )
                print(opt.param_groups[0]["lr"], "learning rate")
                show_distribution(batch['mask'][example_image_idx, ...], pred_mask[example_image_idx, ...], img_name, loss)
                torch.save(
                    {"state_dict": net.state_dict(), "optimizer": opt.state_dict()},
                    f"models/sat_model_epoch_{epoch + epoch_num_from_file}_batch_{batch_count + batch_num_from_file}.pt",
                )
                with torch.no_grad():
                    if not run_test:
                        print("skipping calc test loss")
                        continue
                    valid_idx = np.random.choice(
                        range(len(test_data_loader)), int(len(test_data_loader) / 4)
                    )
                    total_test_loss = 0
                    for batch_count_test, batch in enumerate(test_data_loader):

                        if batch_count_test not in valid_idx:
                            continue
                        sat_img = batch["sat_image"].to(device)
                        pred_mask = net(sat_img)
                        gt_mask = batch["mask"].to(device)
                        classes_loss = loss_func(pred_mask, torch.argmax(gt_mask, 1))
                        total_test_loss += classes_loss
                    mean_total_test_loss = total_test_loss / len(test_data_loader)
                    pprint({"test loss": mean_total_test_loss})
                    example_image_idx = np.random.choice(
                        np.arange(len(batch["sat_path"]))
                    )
                    writer.add_scalar("test loss", mean_total_test_loss, global_step)
                if valid_idx[0] % 2 == 0:
                    show_results(
                        batch["mask"][example_image_idx, ...],
                        pred_mask[example_image_idx, ...],
                        batch["sat_path"][example_image_idx],
                        loss,
                        global_step,
                        writer,
                    )
    writer.close()
