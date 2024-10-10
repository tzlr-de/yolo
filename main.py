#!/usr/bin/env python

import hydra
import structlog

from ultralytics import YOLO
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig):
    global logger

    weights = cfg.model.weights if cfg.model.weights is not None else f'yolo{cfg.model.version}-seg.pt'
    logger.info(f'Using weights: {weights}')
    model = YOLO(weights)

    if cfg.mode.name == 'train':
        model.train(
            data=cfg.data.config,
            imgsz=cfg.data.image_size,
            epochs=cfg.mode.epochs,

            optimizer=cfg.mode.optimizer,
            batch=cfg.mode.batch_size,
            workers=cfg.mode.num_workers,

            lr0=cfg.mode.learning_rate,
            lrf=cfg.mode.learning_rate_final,

            plots=True,
        )
    elif cfg.mode.name == 'validate':
        model.val(
            data=cfg.data.config,
            imgsz=cfg.data.image_size,
            batch=cfg.mode.batch_size,
            workers=cfg.mode.num_workers,
        )

if __name__ == "__main__":
    logger = structlog.get_logger()
    main()
