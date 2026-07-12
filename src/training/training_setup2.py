"""
File: training_setup2.py

Description:
    Entry point for single-task food_type classification training.
    Optimized for RTX 4070 (8GB): gradient checkpointing + micro_batch=4 + 448x448 center crop.
    Accumulation steps =32/4 = 8.

Usage:
    python -m src.training.training_setup2
"""

import ast
import json
import os

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision import transforms
from torchviz import make_dot

from src.config import (
    DATA_DIR,
    EFFECTIVE_BATCH_SIZE,
    HEADERS,
    MICRO_BATCH_SIZE,
    MODEL_CONFIG,
    ROOT_DIR,
    STRESS_TEST_HEIGHT,
)

LEARNING_RATE = 3e-4
from src.training.dataset import (
    FoodDataset,
    build_class_weights,
    build_multilabel_vocab,
    build_vocab,
    collate_fn,
    set_transform,
)
from src.training.evaluation import build_log_row, make_confusion_matrix
from src.training.losses import LossCombiner
from src.training.model import SwinModel, init_weights
from src.training.training import (
    load_checkpoint,
    stress_test,
    train_epoch,
)

if __name__ == "__main__":
    # ──────────────────────────────────────────────
    # Paths
    # ──────────────────────────────────────────────

    model_name = "single"
    log_path = os.path.join(ROOT_DIR, "stats", "losses2.csv")
    checkpoint_path = os.path.join(os.path.dirname(log_path), "checkpoint2.pt")
    val_log_path = os.path.join(ROOT_DIR, "stats", "validation_log.csv")
    test_log_path = os.path.join(ROOT_DIR, "stats", "test_log.csv")
    val_report_dir = os.path.join(ROOT_DIR, "stats", "val_reports_single")

    # ──────────────────────────────────────────────
    # Load data
    # ──────────────────────────────────────────────

    train = pd.read_parquet(os.path.join(DATA_DIR, "train_labels_sorted.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val_labels_sorted.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test_labels_sorted.parquet"))

    with open(os.path.join(ROOT_DIR, "stats", "data", "regression_labels_stats.json")) as f:
        regression_norms = json.load(f)

    # Resize shorter side to 448, center crop to 448×448, then normalize
    CROP_SIZE = 448
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
    # Model
    # ──────────────────────────────────────────────

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = SwinModel(**MODEL_CONFIG).to(device)
    model.to(memory_format=torch.channels_last)
    model.apply(init_weights)

    x = torch.randn(1, 3, STRESS_TEST_HEIGHT, STRESS_TEST_HEIGHT, device=device)
    out = model(x)
    for i in out:
        print(i, out[i].shape)

    summary(model, input_size=x.shape)

    y = model(x)
    out_sum = sum(v.sum() for v in y.values())
    try:
        dot = make_dot(out_sum, params=dict(model.named_parameters()))
        dot.render("model_graph", format="pdf")
    except Exception as e:
        print(f"Skipping graph rendering: {e}")

    del x, y, out, out_sum
    torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    # Losses + Optimizers
    # ──────────────────────────────────────────────

    food_type_weights = build_class_weights("food_type_counter.csv", food_type_vocab).to(device)

    loss_dictionary = {"food_type": nn.CrossEntropyLoss(weight=food_type_weights)}
    total_loss = LossCombiner(losses=loss_dictionary)

    METRIC_SCHEMA = {
        "meta": ["epoch", "model_name", "total_loss"],
        "classification": ["food_type_acc", "precision", "recall", "f1"],
    }
    HEADERS = sum(METRIC_SCHEMA.values(), [])

    model_optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    steps_per_epoch = train.shape[0] // EFFECTIVE_BATCH_SIZE
    epochs = 15
    total_steps = epochs * steps_per_epoch

    scheduler = CosineAnnealingLR(model_optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # ──────────────────────────────────────────────
    # Stress test
    # ──────────────────────────────────────────────

    stress_test(
        model, total_loss, model_optimizer, device,
        batch_size=MICRO_BATCH_SIZE,
        height=STRESS_TEST_HEIGHT,
        width=STRESS_TEST_HEIGHT,
        use_checkpoint=True,
        active_heads={"food_type"},
    )

    assert EFFECTIVE_BATCH_SIZE % MICRO_BATCH_SIZE == 0, (
        "Batch size must be divisible by micro batch size"
    )
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

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("epoch,step,total_loss,ema_loss,avg100\n")

    if not os.path.exists(val_log_path):
        with open(val_log_path, "w") as f:
            f.write(",".join(HEADERS) + "\n")

    if not os.path.exists(test_log_path):
        with open(test_log_path, "w") as f:
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

    from src.training.evaluation import evaluate, compute_metrics, save_validation_report

    test_state = evaluate(model, test_set, device, total_loss, desc="Testing", active_heads={"food_type"})
    test_metrics = compute_metrics(test_state, model_name=model_name, epoch=epochs)
    print(test_metrics)

    row = build_log_row(test_metrics, HEADERS)
    with open(test_log_path, "a") as f:
        f.write(",".join(map(str, row)) + "\n")

    test_report_dir = os.path.join(ROOT_DIR, "stats", "test_reports_single")
    save_validation_report(test_state, test_report_dir, epoch=epochs)

    print(make_confusion_matrix(model, test_set, device))
    writer.close()
