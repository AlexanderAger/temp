import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class ActionSelector(nn.Module):
    def __init__(self, num_plans=1, soft_maxpool=False, soft_maxpool_temperature=1.0):
        super().__init__()
        self.num_plans = num_plans
        self.soft_maxpool = soft_maxpool
        self.temperature = soft_maxpool_temperature
        
    def forward(self, q_values):
        if self.soft_maxpool:
            # Soft max pooling implementation
            q_values = q_values * self.temperature
            softmaxed = F.softmax(q_values, dim=1)
            return (softmaxed * q_values).sum(dim=1, keepdim=True)
        else:
            # Regular max pooling
            return torch.max(q_values, dim=1, keepdim=True)[0]


def attention(tensor, params):
    #Kernel, process sections of the maze rather than the whole image
    x_coord, y_coord, args = params
    batch_size = tensor.size(0)

    s1_expanded = x_coord.expand(args.dims, 1, args.num_actions, batch_size).permute(3, 2, 1, 0)
    s2_expanded = y_coord.expand(1, args.num_actions, batch_size).permute(2, 1, 0)
    
    return tensor.gather(2, s1_expanded).squeeze(2).gather(2, s2_expanded).squeeze(2)


class DTVIN(nn.Module):
    def __init__(self, args):
        super(DTVIN, self).__init__()
        kernel_size = 3
        padding = 2
        
        #First hidden Conv layer used for feature extraction
        #self.hidden_layer = nn.Conv2d(args.input_channels, args.hidden_channels, kernel_size=kernel_size, padding=padding, bias=True)
        
        # Try a deeper feature extractor
        self.hidden_layer = nn.Sequential(
            nn.Conv2d(args.input_channels, args.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.hidden_channels, args.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        #Conv layer to generate reward image
        self.reward_layer = nn.Conv2d(args.hidden_channels, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
        #Dynamic transition mapping module - maps each local patch of the maze to a transition kernel for each action
        self.transition_mapping = nn.Conv2d(
            args.hidden_channels, 
            args.num_actions * kernel_size * kernel_size,
            kernel_size=kernel_size,    
            padding=padding, 
            bias=False
        ) #Output has shape [batch_size, num_actions*kernel_size*kernel_size, height, width]
        
        #Final fully connected layer to give action probabilities
        #self.Action_Layer = nn.Linear(args.num_actions, 4, bias=False)
        
        self.Action_Layer = nn.Sequential(
        nn.Linear(args.num_actions, args.hidden_channels),
        nn.ReLU(),
        nn.Linear(args.hidden_channels, args.num_actions)
        )

        #Store intermediate Q-values and logits for adaptive highway loss
        self.intermediate_q_values = []
        self.intermediate_logits = []
 
        #Save kernel size for reshaping
        self.kernel_size = kernel_size
        self.args = args
        
    def forward(self, grid, x_coord, y_coord, args):
        #Reset intermediate values during each forward pass
        self.intermediate_q_values = []
        self.intermediate_logits = []
        
        hidden = self.hidden_layer(grid) #Get hidden features from observation image
        reward = self.reward_layer(hidden) #Get reward image from hidden features
        dynamic_transitions = self.transition_mapping(hidden) #Get dynamic transition kernels from hidden features
        
        #Initialize value map (zero everywhere)
        v = torch.zeros_like(reward)
        
        #K-iterations of Value Iteration module
        for i in range(args.k):
            #Reshape dynamic transitions to make them easier to work with
            batch_size, _, height, width = dynamic_transitions.size()
            reshaped_transitions = dynamic_transitions.view(
                batch_size, 
                args.num_actions, 
                self.kernel_size * self.kernel_size, 
                height, width
            )
            
            # Apply softmax to normalize transition probabilities
            softmax_transitions = F.softmax(reshaped_transitions, dim=2)
            
            # Extract patches for value map
            unfolded_v = F.unfold(v, kernel_size=self.kernel_size, padding=self.kernel_size//2)
            # Shape: [batch_size, kernel_size*kernel_size, height*width]
            
            # Initialize Q-values
            q = torch.zeros(batch_size, args.num_actions, height, width, device=grid.device)
            
            # Compute Q-values for each action using action-specific transition kernels
            for a in range(args.num_actions):
                # Reshape for proper broadcasting
                # Get transitions for this action: [batch_size, kernel_size*kernel_size, height, width]
                action_transitions = softmax_transitions[:, a]
                
                # Reshape unfolded_v to match spatial dimensions
                v_reshaped = unfolded_v.view(batch_size, self.kernel_size*self.kernel_size, height, width)
                
                # Element-wise multiplication and sum for each spatial location
                # This applies the transition kernel to the value function
                q_values = (action_transitions * v_reshaped).sum(dim=1)
                
                # Add reward
                q[:, a] = q_values + reward.squeeze(1)
            
            # Get max Q-value for each state
            v, indices = torch.max(q, dim=1, keepdim=True)

            passable_mask = (grid[:, 0] == 0).float()

            # Mask for setting walls to -1 (-1 for walls, 0 for passable areas)
            wall_value_mask = (grid[:, 0] == 1).float() * -1


            passable_mask = passable_mask.unsqueeze(1)      # [32, 1, 15, 15]
            wall_value_mask = wall_value_mask.unsqueeze(1)  # [32, 1, 15, 15]

            indices_cropped = indices[:, :, :49, :49] #NEED TO CHANGE WITH MAZE SIZE
            # Apply both masks: zero out walls with first mask, then add -1 to wall locations
            indices_masked = indices_cropped * passable_mask + wall_value_mask
            ###NOW CHANGE FROM USING INTERMEDIATE LOGITS TO USING THE FULL MASKED MAP.

            # Extract Q-values at specified positions
            q_out_i = attention(q, [(x_coord+1).long(), (y_coord+1).long(), args])
            # Convert Q-values to action logits
            # Store intermediate values for highway loss

            self.intermediate_q_values.append(indices_masked)

            # Instead of using the attention function, process the entire q map
            q = q[:, :, :49, :49]
            batch_size, num_actions, height, width = q.size()  # Now height=15, width=15
            
            # Reshape q to apply the Action_Layer to all spatial locations
            q_reshaped = q.permute(0, 2, 3, 1)  # [batch_size, height, width, num_actions]
            q_flattened = q_reshaped.reshape(-1, num_actions)  # [batch_size*height*width, num_actions]
            
            # Apply Action_Layer to all positions
            logits_flattened = self.Action_Layer(q_flattened)  # [batch_size*height*width, num_actions]
            
            # Reshape back to spatial dimensions
            logits_spatial = logits_flattened.reshape(batch_size, height, width, num_actions)
            logits_spatial = logits_spatial.permute(0, 3, 1, 2)  # [batch_size, num_actions, height, width]
            # Store intermediate logits
            self.intermediate_logits.append(logits_spatial)
            
            # For backward compatibility, also compute the specific position's logits
            q_out_i = attention(q, [(x_coord+1).long(), (y_coord+1).long(), args])
            logits_i = self.Action_Layer(q_out_i)   

        
        # Return final logits
        q_out = self.intermediate_q_values[-1]
        logits = self.intermediate_logits[-1]
        return logits
        
    def get_intermediate_q_values(self):
        return self.intermediate_q_values #Intermediate logits for computing adaptive highway loss

def compute_adaptive_highway_loss(model, target_actions, path_lengths, criterion, l_j=5):
    # Get both logits and q_values for completeness
    intermediate_logits = model.intermediate_logits
    num_layers = len(intermediate_logits)
    batch_size = target_actions.size(0)
    device = target_actions.device
    total_loss = 0.0
    total_count = 0
    
    # Get number of action classes
    num_actions = 4  # Update this if your environment has a different number of actions
    
    for batch_idx in range(batch_size):
        path_length = path_lengths[batch_idx].item()
        if path_length <= 0:
            continue
            
        for n in range(num_layers):
            if n >= path_length - 1 and (n % l_j == 0):
                # Get logits map and target map
                logits_map = intermediate_logits[n]  # Shape [batch_size, num_actions, H, W]
                target_map = target_actions[batch_idx]  # Shape [H, W]
                # Create mask for valid positions AND valid target values
                valid_mask = (target_map >= 0) & (target_map < num_actions)
                
                if valid_mask.sum() == 0:
                    continue
                
                # Process only valid positions
                valid_positions = valid_mask.nonzero(as_tuple=True)
                
                # Extract logits for valid positions
                # Reshape logits for easier position indexing: [batch_idx, action, y, x]
                batch_logits = logits_map[batch_idx]  # [num_actions, H, W]
                # Get logits for all actions at valid positions
                valid_logits = batch_logits[:, valid_positions[0], valid_positions[1]]  # [num_actions, num_valid_positions]
                valid_logits = valid_logits.permute(1, 0)  # [num_valid_positions, num_actions]
                
                # Get corresponding targets
                valid_targets = target_map[valid_positions]  # [num_valid_positions]
                
                # Compute loss directly using logits
                loss = criterion(valid_logits, valid_targets.long())
                
                total_loss += loss
                total_count += 1
    
    if total_count == 0:
        return torch.zeros(1, device=device, requires_grad=True)
    
    return total_loss / total_count