import torch
import torch.nn.functional as F
import torch.optim as optim

from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
from homework.models import MODEL_FACTORY, save_model


def train(
    model_name: str = "linear_planner",
    transform_pipeline: str = "state_only",
    num_workers: int = 4,
    lr: float = 1e-3,
    batch_size: int = 128,
    num_epoch: int = 40,
    train_path: str = "drive_data/train",
    val_path: str = "drive_data/val",
    device: str = "cuda",
):
    """
    General training function for road planners.

    Example usage:
        for lr in [1e-2, 1e-3, 1e-4]:
            train(
                model_name="linear_planner",
                transform_pipeline="state_only",
                num_workers=4,
                lr=lr,
                batch_size=128,
                num_epoch=40,
            )
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # -------------------------
    # 1) Load the dataset
    # -------------------------
    train_loader = load_data(
        dataset_path=train_path,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = load_data(
        dataset_path=val_path,
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Training model '{model_name}' for {num_epoch} epochs, lr={lr}, batch_size={batch_size} ...")

    # -------------------------
    # 2) Create the model & optimizer
    # -------------------------
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model_name='{model_name}'. Available: {list(MODEL_FACTORY.keys())}")

    # Instantiate the model
    model = MODEL_FACTORY[model_name]()  # if you have custom kwargs, add them here
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = F.smooth_l1_loss  # Huber / L1 / MSE can all work

    # We'll track validation performance with PlannerMetric
    metric = PlannerMetric()

    # -------------------------
    # 3) Training loop
    # -------------------------
    for epoch in range(num_epoch):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            # For "state_only" pipeline, the batch will have track_left, track_right, waypoints, waypoints_mask
            # For "default", it will have image, track_left, track_right, etc.

            if "image" in batch:
                # e.g., training CNN
                inputs = batch["image"].to(device)
                preds = model(inputs)
            else:
                # e.g., training MLP/Transformer/Linear
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                preds = model(track_left=track_left, track_right=track_right)

            # Supervised label
            waypoints = batch["waypoints"].to(device)

            loss = loss_fn(preds, waypoints)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -------------------------
        # 4) Validation
        # -------------------------
        model.eval()
        metric.reset()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if "image" in batch:
                    inputs = batch["image"].to(device)
                    preds = model(inputs)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    preds = model(track_left=track_left, track_right=track_right)

                waypoints = batch["waypoints"].to(device)
                val_loss = loss_fn(preds, waypoints)
                total_val_loss += val_loss.item()

                waypoints_mask = batch["waypoints_mask"].to(device)
                metric.add(preds, waypoints, waypoints_mask)

        avg_val_loss = total_val_loss / len(val_loader)
        results = metric.compute()  # dict with "l1_error", "longitudinal_error", "lateral_error", etc.

        print(
            f"[Epoch {epoch+1}/{num_epoch}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Long Err: {results['longitudinal_error']:.4f} | "
            f"Lat Err: {results['lateral_error']:.4f}"
        )

    # -------------------------
    # 5) Save final model
    # -------------------------
    save_model(model)
    print(f"Training complete. Saved '{model_name}.th'!\n")
