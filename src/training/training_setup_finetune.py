"""
File: training_setup_finetune.py

Description:
    Fine-tune pretrained Swin-B (timm) on food_type classification.
    Establishes baseline performance for comparison with custom architecture.

Usage:
    python -m src.training.training_setup_finetune
"""

import ast
import json
import os

import pandas as pd
import timm
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.config import (
    DATA_DIR,
    EFFECTIVE_BATCH_SIZE,
    LEARNING_RATE,
    MICRO_BATCH_SIZE,
    ROOT_DIR,
)
from src.training.dataset import (
    FoodDataset,
    build_class_weights,
    build_multilabel_vocab,
    build_vocab,
    collate_fn,
    load_image,
    set_transform,
)
from src.training.evaluation import build_log_row
from src.training.losses import LossCombiner
from src.training.training import (
    load_checkpoint,
    stress_test,
    train_epoch,
)

CROP_SIZE = 384

if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────────

    model_name = "swin_b_finetune"
    log_path = os.path.join(ROOT_DIR, "stats", "losses_finetune.csv")
    checkpoint_path = os.path.join(os.path.dirname(log_path), "checkpoint_finetune.pt")
    val_log_path = os.path.join(ROOT_DIR, "stats", "validation_log_finetune.csv")
    test_log_path = os.path.join(ROOT_DIR, "stats", "test_log_finetune.csv")
    val_report_dir = os.path.join(ROOT_DIR, "stats", "val_reports_finetune")

    # ──────────────────────────────────────────────
    # Load data
    # ──────────────────────────────────────────────

    train = pd.read_parquet(os.path.join(DATA_DIR, "train_labels_sorted.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val_labels_sorted.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test_labels_sorted.parquet"))

    with open(os.path.join(ROOT_DIR, "stats", "data", "regression_labels_stats.json")) as f:
        regression_norms = json.load(f)

    # Resize shorter side to 384, center crop to 384×384, ImageNet normalize
    train_transform = transforms.Compose([
        transforms.Resize(CROP_SIZE, antialias=True),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    set_transform(train_transform)

    # ──────────────────────────────────────────────
    # Build vocabularies
    # ──────────────────────────────────────────────

    food_type_vocab, _ = build_vocab(train["food_type"])
    dish_name_vocab, _ = build_vocab(train["dish_name"])
    cooking_method_vocab, _ = build_multilabel_vocab(train["cooking_method"])

    train["portion_size"] = train["portion_size"].apply(ast.literal_eval)
    val["portion_size"] = val["portion_size"].apply(ast.literal_eval)
    test["portion_size"] = test["portion_size"].apply(ast.literal_eval)

    portion_ingredient_vocab, _ = build_multilabel_vocab(
        train["portion_size"].apply(lambda x: list(i[0] for i in x))
    )
    ingredients_vocab, _ = build_multilabel_vocab(train["ingredients"])

    # ──────────────────────────────────────────────
    # Datasets + DataLoaders
    # ──────────────────────────────────────────────

    dataset_kwargs = dict(
        food_type_vocab=food_type_vocab,
        dish_name_vocab=dish_name_vocab,
        cooking_method_vocab=cooking_method_vocab,
        ingredients_vocab=ingredients_vocab,
        portion_ingredients_vocab=portion_ingredient_vocab,
        regression_normalization=regression_norms,
        image_dir="images",
    )

    train_dataset = FoodDataset(train, **dataset_kwargs)
    val_dataset = FoodDataset(val, **dataset_kwargs)
    test_dataset = FoodDataset(test, **dataset_kwargs)

    train_set = DataLoader(
        train_dataset, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )
    val_set = DataLoader(
        val_dataset, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn,
        num_workers=1, pin_memory=True, persistent_workers=False, prefetch_factor=2,
    )
    test_set = DataLoader(
        test_dataset, batch_size=MICRO_BATCH_SIZE, collate_fn=collate_fn,
        num_workers=1, pin_memory=True, persistent_workers=False, prefetch_factor=2,
    )

    # ──────────────────────────────────────────────
    # Model — pretrained Swin-B, 5 food classes
    # ──────────────────────────────────────────────

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    backbone = timm.create_model(
        "swin_base_patch4_window12_384",
        pretrained=True,
        num_classes=len(food_type_vocab),
    )
    backbone.to(device)
    backbone.to(memory_format=torch.channels_last)

    # Wrap to return dict matching our training interface
    class FoodTypeModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x, use_checkpoint=False, active_heads=None):
            if use_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                logits = checkpoint(self.backbone, x, use_reentrant=False)
            else:
                logits = self.backbone(x)
            return {"food_type_logits": logits}

    model = FoodTypeModel(backbone)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Swin-B loaded: {params_m:.1f}M params, {len(food_type_vocab)} classes")

    # ──────────────────────────────────────────────
    # Losses + Optimizers
    # ──────────────────────────────────────────────

    food_type_weights = build_class_weights("food_type_counter.csv", food_type_vocab).to(device)

    loss_dictionary = {"food_type": nn.CrossEntropyLoss(weight=food_type_weights)}
    total_loss = LossCombiner(losses=loss_dictionary)

    HEADERS = ["epoch", "model_name", "total_loss", "food_type_acc", "precision", "recall", "f1"]

    model_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    steps_per_epoch = train.shape[0] // EFFECTIVE_BATCH_SIZE
    epochs = 5
    total_steps = epochs * steps_per_epoch

    scheduler = CosineAnnealingLR(model_optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ──────────────────────────────────────────────
    # Stress test
    # ──────────────────────────────────────────────

    stress_test(
        model, total_loss, model_optimizer, device,
        batch_size=MICRO_BATCH_SIZE,
        height=CROP_SIZE,
        width=CROP_SIZE,
        use_checkpoint=True,
        active_heads={"food_type"},
    )

    assert EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE == 0
    accumulation_steps = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE
    training_step = 0
    start_epoch = 0

    # ──────────────────────────────────────────────
    # Resume
    # ──────────────────────────────────────────────

    if os.path.exists(checkpoint_path):
        start_epoch, training_step = load_checkpoint(
            checkpoint_path, model, total_loss,
            model_optimizer, scheduler, scaler, device,
        )

    # ──────────────────────────────────────────────
    # Init log files
    # ──────────────────────────────────────────────

    for p in [log_path, val_log_path, test_log_path]:
        if not os.path.exists(p):
            with open(p, "w") as f:
                if p == log_path:
                    f.write("epoch,step,total_loss,ema_loss,avg100\n")
                else:
                    f.write(",".join(HEADERS) + "\n")

    # ──────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────

    writer = SummaryWriter(log_dir=os.path.join(ROOT_DIR, "runs"))

    for epoch in range(start_epoch, epochs):
        training_step = train_epoch(
            model=model,
            train_loader=train_set,
            train_dataset=train_dataset,
            loss_dictionary=loss_dictionary,
            total_loss=total_loss,
            model_optimizer=model_optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            accumulation_steps=accumulation_steps,
            model_name=model_name,
            val_loader=val_set,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            val_log_path=val_log_path,
            writer=writer,
            HEADERS=HEADERS,
            build_log_row_fn=build_log_row,
            gradient_checkpointing=True,
            val_save_dir=val_report_dir,
            active_heads={"food_type"},
        )

    # ──────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────

    from src.training.evaluation import evaluate, compute_metrics, make_confusion_matrix, save_validation_report

    test_state = evaluate(model, test_set, device, total_loss, desc="Testing", active_heads={"food_type"})
    test_metrics = compute_metrics(test_state, model_name=model_name, epoch=epochs)
    print(test_metrics)

    row = build_log_row(test_metrics, HEADERS)
    with open(test_log_path, "a") as f:
        f.write(",".join(map(str, row)) + "\n")

    test_report_dir = os.path.join(ROOT_DIR, "stats", "test_reports_finetune")
    save_validation_report(test_state, test_report_dir, epoch=epochs)

    print(make_confusion_matrix(model, test_set, device))
    writer.close()
