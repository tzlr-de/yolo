#!/usr/bin/env python

import hydra
import structlog
import numpy as np
import re
import cv2
import shutil

from omegaconf import DictConfig
from backgroundremover.u2net.detect import load_model
from backgroundremover.u2net.detect import predict
from PIL import Image as PILImage
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def load_data(cfg: DictConfig, *, ordered: bool = True):
    global logger
    assert cfg.data.root is not None, 'Data root is not specified!'
    paths = list(Path(cfg.data.root).rglob(cfg.data.glob_pattern))
    logger.info(f'Found {len(paths)} images')
    if ordered:
        return sorted(paths)
    return paths

def process(im: np.ndarray, model, *, kernel_size=15):
    mask = predict(model, im).convert("L")
    mask = mask.resize(im.shape[:2][::-1], PILImage.LANCZOS)
    mask = np.where(np.array(mask) < 128, 0, 1).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size*3, kernel_size*3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones((kernel_size//3, kernel_size//3)))

    masked = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    masked[..., 3] = mask * 255

    rect = (x, y, w, h) = cv2.boundingRect(cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((kernel_size, kernel_size))))

    contours,_ = cv2.findContours(mask, 1, 2)
    cnt = contours[0]
    cropped = masked[y:y+h, x:x+w].copy()
    return masked, mask, cropped, rect, cnt


def show(im, masked, mask, cropped, rect, cnt):
    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    [ax.axis("off") for ax in axs.ravel()]
    axs[0,0].imshow(im)
    axs[0,1].imshow(im)
    axs[0,1].imshow(mask, alpha=0.7)
    #masked = np.where(mask[..., None], im, 255)

    # contours,_ = cv2.findContours(mask, 1, 2)
    # cnt = contours[0]
    # (y, x), (w, h), angle = cv2.minAreaRect(cnt)
    # rect = tuple(map(int, (x, y, w, h)))
    cv2.rectangle(masked, rect, (255, 0, 0, 255), 2)
    cv2.drawContours(masked, [cnt], -1, (0, 255, 0, 255), 2)
    axs[1,0].imshow(masked)

    axs[1,1].imshow(cropped)
    return fig, axs

@hydra.main(config_path="conf", config_name="segment", version_base=None)
def main(cfg: DictConfig):
    global logger
    impaths = load_data(cfg)
    model = load_model(model_name=cfg.model.name)
    dest = None if cfg.dest is None else Path(cfg.dest)

    for impath in tqdm(impaths):

        if impath.name in cfg.data.exclude:
            logger.info(f'Skipping {impath}')
            continue

        cls_name, idx = re.match(r"(\w+)_(\d+).png", impath.name).groups()
        idx = int(idx)


        im = cv2.imread(str(impath))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        masked, mask, cropped, rect, cnt = process(im, model)

        if dest is None:
            show(im, masked, mask, cropped, rect, cnt)
            plt.tight_layout()
            plt.show()
            plt.close()
        else:
            orig_dest = dest / "original" / cls_name
            masked_dest = dest / "masked" / cls_name
            cropped_dest = dest / "cropped" / cls_name

            orig_dest.mkdir(parents=True, exist_ok=True)
            masked_dest.mkdir(parents=True, exist_ok=True)
            cropped_dest.mkdir(parents=True, exist_ok=True)

            shutil.copy(impath, orig_dest / f"{idx:04d}{impath.suffix}")
            cv2.imwrite(str(masked_dest / f"{idx:04d}{impath.suffix}"),
                cv2.cvtColor(masked, cv2.COLOR_RGBA2BGRA))
            cv2.imwrite(str(cropped_dest / f"{idx:04d}{impath.suffix}"),
                cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA))


if __name__ == "__main__":
    logger = structlog.get_logger()
    main()
