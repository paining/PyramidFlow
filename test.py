import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import time, argparse
from sklearn.metrics import roc_auc_score

from model import PyramidFlow
from util import MVTecAD_DAC as MVTecAD, BatchDiffLoss
from util import fix_randseed, compute_pro_score_fast, getLogger

from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import matplotlib.pyplot as plt
import os.path as osp
import os
import cv2
import imageutils
from torchmetrics import ROC, F1Score


def test(
    logger,
    save_name,
    cls_name,
    datapath,
    resnetX,
    num_layer,
    vn_dims,
    ksize,
    channel,
    num_stack,
    device,
    batch_size,
    save_memory,
    num_epochs,
    save_path,
    ckpt,
):
    # save config
    save_dict = {
        "cls_name": cls_name,
        "resnetX": resnetX,
        "num_layer": num_layer,
        "vn_dims": vn_dims,
        "ksize": ksize,
        "channel": channel,
        "num_stack": num_stack,
        "batch_size": batch_size,
    }
    loader_dict = fix_randseed(seed=0)
    os.makedirs(osp.join(save_path, "result"), exist_ok=True)

    # model
    flow = PyramidFlow(
        resnetX, channel, num_layer, num_stack, ksize, vn_dims, saveMem
    )
    flow(torch.zeros(1, 3, 416, 2592))
    flow.load_state_dict(torch.load(ckpt, map_location="cpu"))
    flow.to(device)
    # x_size = 256 if resnetX==0 else 1024
    x_size = (416, 4096)
    y_size = (104, 1024)

    # dataset
    val_dataset = MVTecAD(
        cls_name,
        mode="val",
        x_size=x_size,
        y_size=y_size,
        datapath=datapath,
        logger=logger,
    )
    test_dataset = MVTecAD(
        cls_name,
        mode="test",
        x_size=x_size,
        y_size=y_size,
        datapath=datapath,
        logger=logger,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        drop_last=False,
        **loader_dict,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        **loader_dict,
    )

    # training & evaluation
    pixel_auroc_lst = [0]
    pixel_pro_lst = [0]
    image_auroc_lst = [0]
    losses_lst = [0]
    val_losses_lst = [0]
    t0 = time.time()

    # val for template
    flow.eval()
    feat_sum, cnt = [0 for _ in range(num_layer)], 0
    for val_dict in tqdm(val_loader, desc="Calculate MEAN", leave=False):
        image = val_dict["images"].to(device)
        with torch.no_grad():
            pyramid2 = flow(image)
            cnt += 1
        feat_sum = [p0 + p for p0, p in zip(feat_sum, pyramid2)]
    feat_mean = [p / cnt for p in feat_sum]

    # test
    flow.eval()
    diff_list, labels_list = [], []
    for test_dict in tqdm(test_loader, desc="Test", leave=False):
        image, labels = test_dict["images"].to(device), test_dict["labels"]
        with torch.no_grad():
            pyramid2 = flow(image)
            pyramid_diff = [
                (feat2 - template).abs()
                for feat2, template in zip(pyramid2, feat_mean)
            ]
            diff = flow.pyramid.compose_pyramid(pyramid_diff).mean(
                1, keepdim=True
            )  # b,1,h,w
            diff_list.append(diff.cpu())
            labels_list.append(labels.cpu() == 1)  # b,1,h,w

    labels_all = torch.concat(labels_list, dim=0)  # b1hw
    amaps = torch.concat(diff_list, dim=0)  # b1hw
    amaps, labels_all = amaps[:, 0], labels_all[:, 0]  # both b,h,w
    pixel_auroc = roc_auc_score(
        labels_all.flatten(), amaps.flatten()
    )  # pixel score
    image_auroc = roc_auc_score(
        labels_all.amax((-1, -2)), amaps.amax((-1, -2))
    )  # image score
    pixel_pro = compute_pro_score_fast(amaps, labels_all)  # pro score
    logger.info(f"   TEST Pixel-AUROC: {pixel_auroc}, time: {time.time()-t0:.1f}s")
    logger.info(f"   TEST Image-AUROC: {image_auroc}, time: {time.time()-t0:.1f}s")
    logger.info(f"   TEST Pixel-PRO: {pixel_pro}, time: {time.time()-t0:.1f}s")

    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(amaps.amax((-1, -2)), labels_all.amax((-1, -2)))
    threshold = thresholds[torch.argmax(tpr-fpr)].item()

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.axvline(threshold, color="r", linestyle=":")
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax.set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax.grid(True, "both", "both", alpha=0.2)
    ax.set_title("ROC")
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(osp.join(save_path, "result", "ROC.png"))
    plt.close(fig)

    vmin, vmax = amaps.min().item(), amaps.max().item()
    for i, amap in enumerate(diff_list):
        file_path = test_dataset.files[i]
        filename = osp.splitext(osp.basename(file_path))[0]
        typename = osp.basename(osp.dirname(file_path))
        amap_np = amap.squeeze().cpu().detach().numpy()
        amap_np = cv2.resize(amap_np, (amap_np.shape[-1]*4, amap_np.shape[-2]*4))
        boundary = (amap_np >= threshold).astype(np.uint8)
        boundary = cv2.morphologyEx(
            boundary,
            cv2.MORPH_ERODE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        ) - boundary
        amap_np = 255.*(amap_np - vmin)/(vmax - vmin)
        heatmap = imageutils.cvt2heatmap(amap_np.astype(np.uint8))
        heatmap[boundary != 0] = (0, 0, 255)
        os.makedirs(osp.join(save_path, "result", typename), exist_ok=True)
        savefile = osp.join(save_path, "result", typename, f"{filename}.png")
        cv2.imwrite(savefile, heatmap)

    if pixel_auroc > np.max(pixel_auroc_lst):
        save_dict["state_dict_pixel"] = {
            k: v.cpu() for k, v in flow.state_dict().items()
        }  # save ckpt
    if pixel_pro > np.max(pixel_pro_lst):
        save_dict["state_dict_pro"] = {
            k: v.cpu() for k, v in flow.state_dict().items()
        }  # save ckpt
    pixel_auroc_lst.append(pixel_auroc)
    pixel_pro_lst.append(pixel_pro)
    image_auroc_lst.append(image_auroc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training on MVTecAD")
    parser.add_argument(
        "--cls",
        type=str,
        default="tile",
        choices=[
            "tile",
            "leather",
            "hazelnut",
            "toothbrush",
            "wood",
            "bottle",
            "cable",
            "capsule",
            "pill",
            "transistor",
            "carpet",
            "zipper",
            "grid",
            "screw",
            "metal_nut",
            "augmentation",
        ],
    )
    parser.add_argument("--datapath", type=str, default="../mvtec_anomaly_detection")
    # hyper-parameters of architecture
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet18",
        choices=["none", "resnet18", "resnet34"],
    )
    parser.add_argument(
        "--numLayer", type=str, default="auto", choices=["auto", "2", "4", "8"]
    )
    parser.add_argument(
        "--volumeNorm", type=str, default="auto", choices=["auto", "CVN", "SVN"]
    )
    # non-key parameters of architecture
    parser.add_argument("--kernelSize", type=int, default=7, choices=[3, 5, 7, 9, 11])
    parser.add_argument("--numChannel", type=int, default=16)
    parser.add_argument("--numStack", type=int, default=4)
    # other parameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batchSize", type=int, default=2)
    parser.add_argument("--saveMemory", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="./saveDir")
    parser.add_argument("--ckpt", type=str)

    args = parser.parse_args()
    cls_name = args.cls
    resnetX = 0 if args.encoder == "none" else int(args.encoder[6:])
    if args.volumeNorm == "auto":
        vn_dims = (
            (0, 2, 3)
            if cls_name in ["carpet", "grid", "bottle", "transistor"]
            else (0, 1)
        )
    elif args.volumeNorm == "CVN":
        vn_dims = (0, 1)
    elif args.volumeNorm == "SVN":
        vn_dims = (0, 2, 3)
    if args.numLayer == "auto":
        num_layer = 4
        if cls_name in ["metal_nut", "carpet", "transistor"]:
            num_layer = 8
        elif cls_name in [
            "screw",
        ]:
            num_layer = 2
    else:
        num_layer = int(args.numLayer)
    ksize = args.kernelSize
    numChannel = args.numChannel
    numStack = args.numStack
    gpu_id = args.gpu
    batchSize = args.batchSize
    saveMem = args.saveMemory
    datapath = args.datapath
    num_epochs = args.num_epochs

    logger, save_name = getLogger(args.save_path)
    logger.info(f"========== Config ==========")
    logger.info(f"> Class: {cls_name}")
    logger.info(f"> MVTecAD dataset root: {datapath}")
    logger.info(f"> Encoder: {args.encoder}")
    logger.info(f"> Volume Normalization: {'CVN' if len(vn_dims)==2 else 'SVN'}")
    logger.info(f"> Num of Pyramid Layer: {num_layer}")
    logger.info(f"> Conv Kernel Size in NF: {ksize}")
    logger.info(f"> Num of Channels in NF: {numChannel}")
    logger.info(f"> Num of Stack Block: {numStack}")
    logger.info(f"> Training Epochs: {num_epochs}")
    logger.info(f"> Batch Size: {batchSize}")
    logger.info(f"> GPU device: cuda:{gpu_id}")
    logger.info(f"> Save Training Memory: {saveMem}")
    logger.info(f"============================")

    test(
        logger,
        save_name,
        cls_name,
        datapath,
        resnetX,
        num_layer,
        vn_dims,
        ksize=ksize,
        channel=numChannel,
        num_stack=numStack,
        device=f"cuda:{gpu_id}",
        batch_size=batchSize,
        save_memory=saveMem,
        num_epochs=num_epochs,
        save_path=args.save_path,
        ckpt = args.ckpt
    )
