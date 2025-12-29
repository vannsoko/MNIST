import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import optuna

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL Image to tensor (C x H x W)
])

# Load the MNIST dataset
dataset = datasets.MNIST(
    root='./data',          # Directory to store the dataset
    train=True,             # Load the training set
    download=True,          # Download if not available
    transform=transform,    # Apply the transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform,
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

class MNIST(nn.Module):
  def __init__(self, h1, h2, dropout):
    super(MNIST, self).__init__()
    self.flatten = nn.Flatten()
    self.linear_rellu_stack = nn.Sequential(
        nn.Linear(28*28, h1),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(h2, 10),
    )
    self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_rellu_stack(x)
    return self.softmax(logits)

model = MNIST(h1=256, h2=128, dropout=0.1).to(device)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in tqdm(range(20)):
  for images, labels in train_loader:
    # Move tensors to the configured device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass and optimisation
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
  print(f' Loss: {loss.item():.4f}')
  
  
counter = 0
correct = 0
model.eval()
with torch.no_grad():
  for image, true_label in test_dataset:
    image = image.unsqueeze(0).to(device)
    output = model(image)
    predicted_label = torch.argmax(output, dim=1).item()
    if true_label==predicted_label:
      correct += 1
    counter += 1

print(f"{correct=}, {counter=}")
print(f"{100*correct/counter}%")

def train_epoch(model, optimizer, batch_size):
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  criterion = nn.CrossEntropyLoss()
  model.train()
  for images, labels in train_loader:
    # Move tensors to the configured device
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass and optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print(f"Loss: {loss.item():.4f}")
  return loss.item()

def validate(model, dataset, batch_size):
  model.eval()
  criterion = nn.CrossEntropyLoss()
  total_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in dataset:
      # Move tensors to the configured device
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)
      total_loss += loss.item()

      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  avg_loss = total_loss / len(dataset)
  accuracy = 100 * correct / total
  return avg_loss, accuracy

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    hidden_size1 = trial.suggest_int('hidden_size1', 256, 2056)
    hidden_size2 = trial.suggest_int('hidden_size2', 128, 2056)

    dropout = trial.suggest_float('dropout', 0.05, 0.25)
    #n_layers = trial.suggest_int('n_layers', 1, 4)

    # Build model
    model = MNIST(h1=hidden_size1,
                      h2=hidden_size2,
                      dropout=dropout,
                      ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in tqdm(range(40)):
        train_loss = train_epoch(model, optimizer, batch_size)
        val_loss, val_accuracy = validate(model, val_loader, batch_size)
        test_loss, test_accuracy = validate(model, test_loader, batch_size)

        print(f"Test Loss: {test_loss}; Test accuracy: {test_accuracy}")
        trial.set_user_attr('test_loss', test_loss)
        trial.set_user_attr('test_accuracy', test_accuracy)
        trial.set_user_attr('final_train_loss', train_loss)
        # Report intermediate value for pruning
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()



    return test_accuracy

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")