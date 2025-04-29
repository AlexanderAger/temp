import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from dataset import MazeTransform
from VIN import DTVIN, compute_adaptive_highway_loss
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as mcolors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Use CUDA if available

class Input:
    datafile = 'xdt_50l.npz'  #Input Mazes
    dims = 49
    input_channels = 2
    num_actions = 4  #8 possible actions (0-7)
    
    #Tunable hyperparameters
    lr = 0.0004100157498555241
    epochs = 30
    k = 400
    hidden_channels = 70
    batch_size = 32
    min_planning_steps = 1
    highway_connect_freq = 8  

input = Input()

#Instantiate a VIN model
net = DTVIN(input).to(device)
  
#Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=input.lr, eps=1e-5)

#Extract and load batches of train and test data
trainset = MazeTransform(input.datafile, dims=input.dims, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=input.batch_size, shuffle=True)
testset = MazeTransform(input.datafile, dims=input.dims, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=input.batch_size, shuffle=False)

# Training
print(f"Starting imitation learning training on {len(trainset)} samples...")

train_losses = []
train_accs = []
path_success_rates = []
path_ratios = []

for epoch in range(input.epochs):
    net.train()
    losses = []
    correct = 0
    total = 0

    epoch_start_time = time.time()
    
    #Running batches, visualizing progress with tqdm
    for batch in tqdm(trainloader, desc=f"Epoch {epoch+1}/{input.epochs}"):
        batch_start_time = time.time()
        grid, x_coord, y_coord, target_actions, path_lengths = batch
        
        #Skip batches that aren't full size for simplicity
        if grid.shape[0] != input.batch_size:
            continue
        
        #Move data to device
        grid = grid.to(device)
        x_coord = x_coord.to(device)
        y_coord = y_coord.to(device)
        target_actions = target_actions.to(device)
        path_lengths = path_lengths.to(device)
        
        #Forward pass
        optimizer.zero_grad()
        logits = net(grid, x_coord, y_coord, input)

        predicted_actions = torch.argmax(logits, dim=1)  # Shape: [32, 15, 15]

        last_grid = logits[-1, 0]  # Shape: [15, 15]
        #Compute adaptive highway loss
        loss = compute_adaptive_highway_loss(
            net, 
            target_actions, 
            path_lengths, 
            criterion, 
            l_j=input.highway_connect_freq,
        )

        #Backward pass
        loss.backward()
        optimizer.step()
        #batch_size = target_actions.size(0)  # 32
        #grid_size = target_actions.size(1) * target_actions.size(2)  # 15 * 15 = 225
        #logits = logits.squeeze(1)
        #correct += (logits == target_actions).sum().item()
        #total += batch_size * grid_size  # 32 * 225 = 7200

        valid_mask = target_actions >= 0
        correct += (predicted_actions[valid_mask] == target_actions[valid_mask]).sum().item()
        total += valid_mask.sum().item()

        losses.append(loss.item())
    
    epoch_time = time.time() - epoch_start_time
    epoch_loss = np.mean(losses)
    epoch_acc = 100.0 * correct / total if total > 0 else 0
    
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"Epoch {epoch+1}/{input.epochs}, Time: {epoch_time}s, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    # Save checkpoint after each epoch
    checkpoint_path = f'DT-VIN-31_epoch{epoch+1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'train_acc': epoch_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'path_success_rates': path_success_rates,
        'path_ratios': path_ratios,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

#Final Evaluation Phase for single-step accuracy
net.eval()
total_correct = 0
total_samples = 0

print("\nEvaluating model on single-step accuracy...")
with torch.no_grad():
    for batch in tqdm(testloader, desc="Testing"):
        grid, x_coord, y_coord, target_actions, path_lengths = batch
        
        #Skip batches that aren't full size for simplicity
        if grid.shape[0] != input.batch_size:
            continue 
        
        #Move data to device
        grid = grid.to(device)
        x_coord = x_coord.to(device)
        y_coord = y_coord.to(device)
        target_actions = target_actions.to(device)
        
        #Forward pass
        logits = net(grid, x_coord, y_coord, input)
        _, predicted = torch.max(logits, dim=1)
        
        #Calculate accuracy (comparing predicted action with target action)
        total_correct += (predicted == target_actions).sum().item()
        total_samples += target_actions.size(0)

#Results
accuracy = (total_correct / total_samples) * 100
print(f"Single Step Test Accuracy: {accuracy:.2f}% ({total_correct}/{total_samples})")


# Save final model with all metrics
final_model_path = 'DT-VIN-15x15-Official_final.pth'
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accs': train_accs,
    'path_success_rates': path_success_rates,
    'path_ratios': path_ratios,
    'final_single_step_accuracy': accuracy,
    'epoch': input.epochs,
}, final_model_path)
print(f"Final model saved to {final_model_path}")

# Plot training and evaluation metrics
plt.figure(figsize=(12, 10))

# Plot training loss
plt.subplot(2, 2, 1)
plt.plot(range(1, input.epochs+1), train_losses, 'b-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Plot training accuracy
plt.subplot(2, 2, 2)
plt.plot(range(1, input.epochs+1), train_accs, 'g-')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

# Plot path success rate
plt.subplot(2, 2, 3)
plt.plot(range(1, input.epochs+1), path_success_rates, 'r-')
plt.title('Path Completion Success Rate')
plt.xlabel('Epoch')
plt.ylabel('Success Rate (%)')

# Plot average path ratio
plt.subplot(2, 2, 4)
plt.plot(range(1, input.epochs+1), path_ratios, 'y-')
plt.title('Average Path Ratio (compared to optimal)')
plt.xlabel('Epoch')
plt.ylabel('Path Ratio')
plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)  # Optimal ratio line

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.close()

print("Training and evaluation completed!")
