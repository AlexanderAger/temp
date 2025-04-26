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
    datafile = 'maze_data_35x35_test.npz'  #Input Mazes
    dims = 35
    input_channels = 2
    num_actions = 4  #8 possible actions (0-7)
    
    #Tunable hyperparameters
    lr = 0.0005100157498555241
    epochs = 30
    k = 600
    hidden_channels = 100
    batch_size = 32
    min_planning_steps = 1
    highway_connect_freq = 10    

input = Input()

def full_path_evaluation(model, dataset, input_args, num_examples=100, max_factor=2):
    """
    Evaluates the model's ability to complete full paths by iteratively predicting actions
    until reaching the goal or timing out.
    
    Args:
        model: The trained DT-VIN model
        dataset: Test dataset
        input_args: Model input configuration
        num_examples: Number of maze examples to test (default: 100)
        max_factor: Maximum path length multiplier relative to shortest path (default: 2)
    
    Returns:
        tuple: (success_rate, avg_path_ratio, paths_data)
    """
    num_examples = min(num_examples, len(dataset))  #Ensure valid sample count
    
    total_paths_complete, total_failures = 0, 0
    path_ratios = []
    action_map = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
    paths_data = []
    
    # Track failure reasons
    failure_reasons = {
        'collision': 0,
        'out_of_bounds': 0,
        'loop': 0,
        'timeout': 0
    }

    # Evaluation 
    print(f"\nTesting path completion on {num_examples} examples...")
    device = next(model.parameters()).device
    model.eval()  

    # Randomly sample indices from dataset
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    
    for i, idx in enumerate(tqdm(indices, desc="Path Rollouts")):
        # Fetch maze and metadata from dataset
        maze_grid = dataset.get_raw_maze(idx)
        start_x, start_y = dataset.x_coord[idx].item(), dataset.y_coord[idx].item()
        shortest_path = dataset.path_lengths[idx].item()
        
        # Maximum steps as defined by Scaling VINs paper, shortest-path length squared
        max_steps = int(shortest_path * max_factor)
        
        # Retrieve goal as coordinate
        goal_map = maze_grid[1]  # Goal map is second channel
        goal_positions = np.where(goal_map > 0)  # Goal position is where there is a 1, everywhere else is 0
        goal_x, goal_y = goal_positions[0][0], goal_positions[1][0]

        # Path traversal
        path = [(start_x, start_y)]
        visited_states = set([(start_x, start_y)])  # Track visited states to detect loops
        current_x, current_y = start_x, start_y
        grid_tensor = torch.tensor([maze_grid], dtype=torch.float32).to(device)

        reached_goal = False
        steps_taken = 0
        actions_taken = []
        failure_reason = None

        for step in range(max_steps):
            x_coordinate = torch.tensor([current_x], dtype=torch.long).to(device)
            y_coordinate = torch.tensor([current_y], dtype=torch.long).to(device)

            with torch.no_grad():
                logits = model(grid_tensor, x_coordinate, y_coordinate, input_args)
                _, predicted_action = torch.max(logits, dim=1)
                action = predicted_action.item()
                actions_taken.append(action)

            # Apply movement
            dx, dy = action_map[action]  # Action map represents change in x and y for each given action
            next_x, next_y = current_x + dx, current_y + dy

            # Validate move - check if out of bounds
            if not (0 <= next_x < dataset.dims and 0 <= next_y < dataset.dims):
                failure_reason = 'out_of_bounds'
                break  # Out of bounds

            # Check for collision with obstacle
            if maze_grid[0, next_x, next_y] > 0:
                failure_reason = 'collision'
                break  # Obstacle hit

            # Move forward
            current_x, current_y = next_x, next_y
            path.append((current_x, current_y))  # Record path taken
            steps_taken += 1

            # Check if we're in a loop (revisiting a state)
            if (current_x, current_y) in visited_states:
                # We've been here before - stuck in a loop
                failure_reason = 'loop'
                break
            visited_states.add((current_x, current_y))

            # Check if goal reached
            if (current_x, current_y) == (goal_x, goal_y):
                reached_goal = True
                break

        # If we've reached max steps without finding goal or another failure
        if not reached_goal and failure_reason is None:
            failure_reason = 'timeout'
        
        # Track success / failure
        path_ratio = steps_taken / shortest_path if shortest_path > 0 else float('inf')
        
        path_info = {
            'maze_idx': idx,
            'path': path,
            'actions': actions_taken,
            'reached_goal': reached_goal,
            'steps_taken': steps_taken,
            'shortest_path': shortest_path,
            'path_ratio': path_ratio,
            'failure_reason': failure_reason
        }
        paths_data.append(path_info)
        
        if reached_goal:
            total_paths_complete += 1
            path_ratios.append(path_ratio)
        else:
            total_failures += 1
            if failure_reason:
                failure_reasons[failure_reason] += 1

    # Compute results
    success_rate = (total_paths_complete / num_examples) * 100 if num_examples else 0
    avg_ratio = np.mean(path_ratios) if path_ratios else 0

    print("\nPath Completion Test Results:")
    print(f"Success Rate: {success_rate:.2f}% ({total_paths_complete}/{num_examples})")
    print(f"Failure Rate: {(100 - success_rate):.2f}% ({total_failures}/{num_examples})")
    print(f"Average Path Ratio: {avg_ratio:.2f}x optimal")
    
    # Print failure statistics
    print("\nFailure Reasons:")
    for reason, count in failure_reasons.items():
        percentage = (count / total_failures) * 100 if total_failures > 0 else 0
        print(f"- {reason.capitalize()}: {count} instances ({percentage:.1f}% of failures)")

    return success_rate, avg_ratio, paths_data

def plot_path_evaluation_metrics(success_rates, avg_ratios, epochs):
    """
    Plot the success rates and average path ratios over epochs
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot success rate
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Success Rate (%)', color='tab:blue')
    ax1.plot(range(1, epochs+1), success_rates, 'o-', color='tab:blue', label='Success Rate')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 100)
    
    # Create second y-axis for path ratio
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Path Ratio (x optimal)', color='tab:red')
    ax2.plot(range(1, epochs+1), avg_ratios, 's-', color='tab:red', label='Avg Path Ratio')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(1, max(3, max(avg_ratios) * 1.1))
    
    # Title and legend
    plt.title('Path Evaluation Metrics Over Training')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig('path_evaluation_metrics.png')
    plt.close()
    
    return fig

def visualize_maze_with_path(maze_grid, path, goal_pos, epoch, idx):
    """
    Visualize a maze with the agent's path
    """
    plt.figure(figsize=(8, 8))
    
    # Create a colormap: obstacles are black, free space is white, path is blue, start is green, goal is red
    cmap = plt.cm.binary
    
    # Plot the obstacle map
    plt.imshow(maze_grid[0], cmap=cmap, interpolation='none')
    
    # Plot the path
    path_x, path_y = zip(*path)
    plt.plot(path_y, path_x, 'b-', linewidth=2, alpha=0.7)  # Note: x and y are swapped for plotting
    
    # Mark start and goal
    plt.plot(path_y[0], path_x[0], 'go', markersize=10, label='Start')
    plt.plot(goal_pos[1], goal_pos[0], 'ro', markersize=10, label='Goal')
    
    # Add grid lines
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.title(f'Maze Path - Epoch {epoch}, Example {idx}')
    plt.legend()
    
    # Save the figure
    plt.savefig(f'maze_path_epoch{epoch}_ex{idx}.png')
    plt.close()

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
        
        #Calculate accuracy (comparing predicted action with target action)
        _, predicted = torch.max(logits, 1)
        total += target_actions.size(0)

        correct += (predicted == target_actions).sum().item()

        losses.append(loss.item())
    
    epoch_time = time.time() - epoch_start_time
    epoch_loss = np.mean(losses)
    epoch_acc = 100.0 * correct / total if total > 0 else 0
    
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"Epoch {epoch+1}/{input.epochs}, Time: {epoch_time}s, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    
    # Evaluate full paths after each epoch
    print(f"\nPerforming full path evaluation after epoch {epoch+1}...")
    success_rate, avg_ratio, paths_data = full_path_evaluation(
        model=net,
        dataset=testset,
        input_args=input,
        num_examples=100,
        max_factor=2
    )
    
    path_success_rates.append(success_rate)
    path_ratios.append(avg_ratio)
    
    # Visualize a few successful and unsuccessful paths
    if paths_data:
        # Visualize up to 3 successful paths and up to 3 unsuccessful paths
        successful_paths = [p for p in paths_data if p['reached_goal']]
        unsuccessful_paths = [p for p in paths_data if not p['reached_goal']]
        
        for i, path_info in enumerate(successful_paths[:3]):
            maze_idx = path_info['maze_idx']
            maze_grid = testset.get_raw_maze(maze_idx)
            goal_positions = np.where(maze_grid[1] > 0)
            goal_pos = (goal_positions[0][0], goal_positions[1][0])
            visualize_maze_with_path(maze_grid, path_info['path'], goal_pos, epoch+1, f"success_{i}")
            
        for i, path_info in enumerate(unsuccessful_paths[:3]):
            maze_idx = path_info['maze_idx']
            maze_grid = testset.get_raw_maze(maze_idx)
            goal_positions = np.where(maze_grid[1] > 0)
            goal_pos = (goal_positions[0][0], goal_positions[1][0])
            visualize_maze_with_path(maze_grid, path_info['path'], goal_pos, epoch+1, f"fail_{i}")
    
    # Plot and save the evaluation metrics
    plot_path_evaluation_metrics(path_success_rates, path_ratios, epoch+1)
    
    # Save checkpoint after each epoch
    checkpoint_path = f'/pthfiles/DT-VIN-31_epoch{epoch+1}.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_loss,
        'train_acc': epoch_acc,
        'path_success_rate': success_rate,
        'path_ratio': avg_ratio,
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

# Final path evaluation
print("\nPerforming final full path evaluation...")
final_success_rate, final_avg_ratio, _ = full_path_evaluation(
    model=net,
    dataset=testset,
    input_args=input,
    num_examples=100,
    max_factor=2
)

# Save final model with all metrics
final_model_path = '/pthfiles/DT-VIN-15x15-Official_final.pth'
torch.save({
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accs': train_accs,
    'path_success_rates': path_success_rates,
    'path_ratios': path_ratios,
    'final_single_step_accuracy': accuracy,
    'final_path_success_rate': final_success_rate,
    'final_path_ratio': final_avg_ratio,
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