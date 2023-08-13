# Imports
import torchvision
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torch.utils.data import DataLoader

from config import (
    DATA_DIR,
    ROOT_DIR,
    DEVICE,
    CLASSES,
    NUM_EPOCHS,
    BATCH_SIZES,
    LEARNING_RATES,
    train_loader,
    train_dataset,
    loss_fn,
    optimizer,
)
from model import CNN

for batch_size in BATCH_SIZES:
    for learning_rate in LEARNING_RATES:
        step = 0
        # Initialize network
        model = CNN()
        model.to(DEVICE)
        model.train()

        train_loader = train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
        optimizer = optimizer(model, learning_rate)
        writer = SummaryWriter(
            f"{ROOT_DIR}/logs/MNIST/MiniBatchSize {batch_size} LR {learning_rate}"
        )

        # Visualize model in TensorBoard
        images, _ = next(iter(train_loader))
        writer.add_graph(model, images.to(DEVICE))
        writer.close()

        for epoch in range(NUM_EPOCHS):
            losses = []
            accuracies = []

            for batch_idx, (data, targets) in enumerate(train_loader):
                # Get data to cuda if possible
                data = data.to(device=DEVICE)
                targets = targets.to(device=DEVICE)

                # forward
                scores = model(data)
                loss = loss_fn(scores, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate 'running' training accuracy
                features = data.reshape(data.shape[0], -1)
                img_grid = torchvision.utils.make_grid(data)
                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                running_train_acc = float(num_correct) / float(data.shape[0])
                accuracies.append(running_train_acc)

                # Plot things to tensorboard
                class_labels = [CLASSES[label] for label in predictions]
                writer.add_image("mnist_images", img_grid)
                writer.add_histogram("fc1", model.fc1.weight)
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )

                if batch_idx == 230:
                    writer.add_embedding(
                        features,
                        metadata=class_labels,
                        label_img=data,
                        global_step=batch_idx,
                    )
                step += 1

            writer.add_hparams(
                {"lr": learning_rate, "bsize": batch_size},
                {
                    "accuracy": sum(accuracies) / len(accuracies),
                    "loss": sum(losses) / len(losses),
                },
            )
