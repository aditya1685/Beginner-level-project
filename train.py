import os
import torch
import data_setup, engine, model

from torchvision import transforms

NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_dir = r"C:\Users\ASUS\OneDrive\Desktop\github projects\pytorch 1\archive\data\pizza_steak_sushi\train"
test_dir = r"C:\Users\ASUS\OneDrive\Desktop\github projects\pytorch 1\archive\data\pizza_steak_sushi\test"

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
  transforms.Resize((224,224)),
  transforms.ToTensor(),
  transforms.Normalize(mean = [0.485,0.456,0.406],
                        std = [0.229,0.224,0.225])
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=transform,
    batch_size=BATCH_SIZE
)

model = model.Model().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)
