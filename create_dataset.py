#!/usr/bin/env python

import hydra
import structlog
import numpy as np
import cv2
import yaml

from omegaconf import DictConfig
from pathlib import Path
from itertools import cycle
from tqdm.auto import tqdm
from PIL import Image


@hydra.main(config_path="conf", config_name="create_dataset", version_base=None)
def main(cfg: DictConfig):
    global logger

    assert cfg.data.images_root is not None, "Set data.images_root=<path to image crops>!"
    assert cfg.data.backgrounds_root is not None, "Set data.backgrounds_root=<path to background images>!"
    assert cfg.dest is not None, "Set dest=<path to save the dataset>!"

    images = list(Path(cfg.data.images_root).rglob(cfg.data.glob_pattern))
    classes = sorted(set([image.parent.name for image in images]))
    backgrounds = list(Path(cfg.data.backgrounds_root).rglob(cfg.data.glob_pattern))
    logger.info(f"Founds {len(images)} images for {len(classes)} classes and {len(backgrounds)} backgrounds")

    logger.info("Class index mapping:")
    for cls_idx, cls in enumerate(classes):
        print(f"{cls_idx: >d}: {cls}")
    cls_images = {cls: [image for image in images if image.parent.name == cls] for cls in classes}
    backgrounds = cycle(backgrounds)

    data_config = dict(
        path=str(Path(cfg.dest).resolve()),
        nc=len(classes),
        names=classes,
    )

    output_size = tuple(cfg.output_size)

    for name in ["train", "val", "test"]:
        n_samples = cfg.subsets.get(name)
        if n_samples is None or not isinstance(n_samples, int):
            logger.info(f"Skipping subset {name}")
            continue
        images_dest = Path(cfg.dest) / "images" / name
        labels_dest = Path(cfg.dest) / "labels" / name
        data_config[name] = str(images_dest.relative_to(cfg.dest))


        with tqdm(total=n_samples * len(classes)) as pbar:
            for cls_idx, cls in enumerate(classes):
                img_cls_dest = images_dest / cls
                lab_cls_dest = labels_dest / cls
                img_cls_dest.mkdir(parents=True, exist_ok=True)
                lab_cls_dest.mkdir(parents=True, exist_ok=True)

                i = 0
                for image in cycle(cls_images[cls]):
                    bkg = next(backgrounds)

                    with Image.open(image) as img, Image.open(bkg) as bkg:
                        output = bkg.resize(output_size).convert('RGBA')
                        W, H = output.size
                        img = img.rotate(np.random.randint(0, 360), expand=True)
                        w, h = img.size

                        # n_w, n_h = W // w, H // h

                        offset_x = np.random.randint(0, W - w)
                        offset_y = np.random.randint(0, H - h)

                        new_im = Image.new("RGBA", output_size, (0, 0, 0, 0))
                        new_im.paste(img, (offset_x, offset_y))

                        mask = new_im.getchannel('A')
                        contours, _ = cv2.findContours(np.array(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        xs, ys = contours[0].transpose(1, 2, 0)[0]

                        output = Image.alpha_composite(output, new_im).convert("RGB")

                        output.save(img_cls_dest / f"{i:06d}.jpg")

                        with open(lab_cls_dest / f"{i:06d}.txt", 'w') as f:
                            f.write(f"{cls_idx:d}")
                            for x, y in zip(xs, ys):
                                f.write(f" {x / W:.7f} {y / H:.7f}")

                            f.write("\n")

                    i += 1
                    pbar.update(1)
                    pbar.set_description(f"Class {cls} ({cls_idx+1}/{len(classes)}) - {i}/{n_samples}")
                    if i >= n_samples:
                        break



    with open(Path(cfg.dest) / "data.yaml", "w") as f:
        yaml.dump(data_config, f)


if __name__ == "__main__":
    logger = structlog.get_logger()
    main()
