import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader

import numpy as np
import time, argparse
from sklearn.metrics import roc_auc_score

from model import PyramidFlow, PyramidFlow_CFA
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
    save_path = osp.join(save_path, "result", osp.splitext(osp.basename(ckpt))[0])
    os.makedirs(save_path, exist_ok=True)

    # model
    flow = PyramidFlow_CFA(
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
    for val_dict in tqdm(val_loader, desc="Calculate MEAN", leave=True):
        image = val_dict["images"].to(device)
        with torch.no_grad():
            pyramid2 = flow(image)
            cnt += 1
        feat_sum = [p0 + p for p0, p in zip(feat_sum, pyramid2)]
    feat_mean = [p / cnt for p in feat_sum]

    targets = {
        'P1' : ['SI-06-G01-06', 'SI-06-G02-18', ],
        'P2' : ['SI-06-G14-03', ],
        'P3' : ['SI-06-G02-16', 'SI-06-G04-18', 'SI-06-G06-13', 'SI-06-G06-24', 'SI-06-G06-31', 'SI-06-G13-12', 'SI-06-G13-13', 'SI-06-G14-02', 'SI-06-G18-30', ],
    }
    filelist = []
    log_str = ('Targets : LeftBranch\n' +
               f"\n".join([f"{k:4s} : {v}" for k, v in targets.items()]))
    logger.info(log_str)
    # test
    flow.eval()
    diff_list, labels_list = [], []
    gi_np_masks, bi_np_masks, ap_masks = [], [], []
    for test_dict in tqdm(test_loader, desc="Test", leave=True):
        image, labels = test_dict["images"].to(device), test_dict["labels"]
        filename, imgtype = test_dict["filename"][0], test_dict['imgtype'][0]
        if imgtype != 'good' and osp.splitext(osp.basename(filename))[0] not in targets[imgtype]:
            continue
        # tqdm.write(f"{imgtype:5s} : {osp.splitext(osp.basename(filename))[0]}")
        filelist.append(filename)
        if resnetX != 0: labels = get_patch_y(labels)
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
            if imgtype == "good":
                gi_np_masks.append(torch.ones_like(labels, dtype=torch.bool).flatten())
                bi_np_masks.append(torch.zeros_like(labels, dtype=torch.bool).flatten())
                ap_masks.append(torch.zeros_like(labels, dtype=torch.bool).flatten())
            else:
                gi_np_masks.append(torch.zeros_like(labels, dtype=torch.bool).flatten())
                kwargs = {'kernel_size': 17, 'padding': 8, 'stride': 1} if resnetX==0 else {'kernel_size': 5, 'padding': 2, 'stride': 1}
                erodes = TF.max_pool2d(labels.float(), **kwargs)
                bi_np_masks.append(erodes.cpu().flatten() != 1)
                ap_masks.append(labels.cpu().flatten() == 1)

    labels_all = torch.concat(labels_list, dim=0)  # b1hw
    amaps = torch.concat(diff_list, dim=0)  # b1hw
    amaps, labels_all = amaps[:, 0], labels_all[:, 0]  # both b,h,w
    gi_np_masks = torch.concat(gi_np_masks)
    bi_np_masks = torch.concat(bi_np_masks)
    ap_masks = torch.concat(ap_masks)
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

    vmin, vmax = amaps.min().item(), amaps.max().item()

    logger.info("Calculate Image ROC...")
    roc = ROC(task="binary")
    i_fpr, i_tpr, i_thresholds = roc(amaps.amax((-1, -2)), labels_all.amax((-1, -2)))
    logger.info("Calculate Image ROC...  Done")
    i_threshold = i_thresholds[torch.argmax(i_tpr - i_fpr)].item()
    logger.info("Calculate Patch ROC...")
    # roc = ROC(task="binary", thresholds=torch.linspace(vmin, vmax, 1000))
    fpr, tpr, thresholds = roc(amaps.flatten(), labels_all.flatten())
    logger.info("Calculate Patch ROC...  Done")

    fig, ax = plt.subplots()
    ax.plot(i_fpr, i_tpr, alpha=0.7, label="Image")
    ax.plot(fpr, tpr, alpha=0.7, label="Patch")
    ax.set_yticks(np.arange(0, 1, 0.1), minor=True)
    ax.set_xticks(np.arange(0, 1, 0.1), minor=True)
    ax.grid(True, "both", "both", alpha=0.2)
    ax.set_title("ROC")
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(osp.join(save_path, "Target-ROC.png"))
    plt.close(fig)

    ###### Find ITPR 100%, IFPR 0% Threshold and Performance
    idx = torch.min(torch.where(i_tpr == 1)[0])
    itpr1_thr = i_thresholds[idx].item()
    p_idx = torch.max(torch.where(thresholds >= itpr1_thr)[0])
    logger.info(f"ITPR 100% Threshold = {itpr1_thr}")
    logger.info(f"ITPR = {i_tpr[idx]:6f}, IFPR = {i_fpr[idx]:6f}")
    logger.info(f"PTPR = {tpr[p_idx]:6f}, PFPR = {fpr[p_idx]:6f}")

    idx = torch.max(torch.where(i_fpr == 0)[0])
    ifpr0_thr = i_thresholds[idx].item()
    p_idx = torch.max(torch.where(thresholds >= ifpr0_thr)[0])
    logger.info(f"IFPR   0% Threshold = {ifpr0_thr}")
    logger.info(f"ITPR = {i_tpr[idx]:6f}, IFPR = {i_fpr[idx]:6f}")
    logger.info(f"PTPR = {tpr[p_idx]:6f}, PFPR = {fpr[p_idx]:6f}")


    amap_f = amaps.flatten()
    leftbranch_threshold = amap_f[gi_np_masks].max().item()
    idx = torch.max(torch.where(i_thresholds >= leftbranch_threshold)[0])
    p_idx = torch.max(torch.where(thresholds >= leftbranch_threshold)[0])
    logger.info(f"IFPR   0% Threshold = {leftbranch_threshold}")
    logger.info(f"ITPR = {i_tpr[idx]:6f}, IFPR = {i_fpr[idx]:6f}")
    logger.info(f"PTPR = {tpr[p_idx]:6f}, PFPR = {fpr[p_idx]:6f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(
        amap_f[gi_np_masks].cpu().detach().numpy(),
        bins=100, range=(vmin, vmax), histtype="step", label="GI_NP", color="g"
    )
    ax.hist(
        amap_f[bi_np_masks].cpu().detach().numpy(),
        bins=100, range=(vmin, vmax), histtype="step", label="BI_NP", color="b"
    )
    ax.hist(
        amap_f[ap_masks].cpu().detach().numpy(), bins=100,
        range=(vmin, vmax), histtype="step", label="AP", color="r"
    )
    ax.axvline(leftbranch_threshold, color="r", linestyle=":", label="max(GI_NP)")
    ax.axvline(itpr1_thr, color="b", linestyle=":", label="ITPR 100%")
    ax.axvline(ifpr0_thr, color="g", linestyle=":", label="IFPR 0%")
    ax.set_xlabel("Anomaly Scores")
    ax.set_ylabel("Number of Patches")
    ax.set_yscale("log")
    ax.set_title(f"PyramidFlow Score Distribution-{f'resnet{resnetX}' if resnetX != 0 else 'Conv'}-{osp.basename(ckpt)}")
    ax.legend()
    ax.grid(True, "both", "both", alpha=0.2)
    fig.tight_layout()
    fig.savefig(osp.join(save_path, "Target-distribution.png"))
    plt.close(fig)

    for i, amap in enumerate(diff_list):
        file_path = filelist[i]
        filename = osp.splitext(osp.basename(file_path))[0]
        typename = osp.basename(osp.dirname(file_path))
        amap_np = amap.squeeze().cpu().detach().numpy()
        if resnetX != 0: amap_np = cv2.resize(amap_np, (amap_np.shape[-1]*4, amap_np.shape[-2]*4), interpolation=cv2.INTER_LINEAR)
        boundary = (amap_np > leftbranch_threshold).astype(np.uint8)
        boundary = cv2.morphologyEx(
            boundary,
            cv2.MORPH_DILATE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        ) - boundary
        amap_np = 255.*(amap_np - vmin)/(vmax - vmin)
        heatmap = imageutils.cvt2heatmap(amap_np.astype(np.uint8))
        heatmap[boundary != 0] = (0, 0, 255)
        os.makedirs(osp.join(save_path, typename), exist_ok=True)
        savefile = osp.join(save_path, typename, f"{filename}.png")
        cv2.imwrite(savefile, heatmap)
        max_score = amap.max().item()
        if typename == "good":
            if max_score > leftbranch_threshold:
                result_str = "FP"
            else:
                result_str = "TN"
        else:
            if max_score > leftbranch_threshold:
                result_str = "TP"
            else:
                result_str = "FN"

        logger.info(f"{typename:10s},{filename:40s},{max_score:10f},{result_str}")

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


def get_patch_y(y, patch_size=4, strong_patch=1):
    B, C, H, W = y.shape
    if C != 1:
        y = y.mean(dim=1, keepdim=True)
    if patch_size == 1:
        return y
    ys = TF.unfold(y.float(), kernel_size=patch_size, stride=patch_size).sum(dim=1) >= strong_patch
    ys = ys.reshape(B, 1, H//patch_size, W//patch_size)
    return ys

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
        choices=["none", "resnet18", "resnet34", "resnet_cfa"],
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
    try:
        resnetX = 0 if args.encoder == "none" else int(args.encoder[6:])
    except ValueError as e:
        if args.encoder[6:] == "_cfa":
            resnetX = "_cfa"
        else:
            raise ValueError("Wrong Encoder parameter!\n" + str(e))
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
