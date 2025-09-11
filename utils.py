import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from copy import deepcopy
import copy
import wandb
from shared import get_window_size_blocks

def estimate_loss(model, batch, step, val_steps):
    loss = 0
    for [inputs, targets] in batch:
        loss += model(inputs, targets, get_window_size_blocks(step))
    assert len(batch) == val_steps
    return loss / len(batch)

def average_models(model, checkpoints: list):
    average_model = copy.deepcopy(model)
    state_dict = average_model.state_dict()

    # define the different trajectory models
    for name, param in state_dict.items():
        # Initialize with zeros
        param.data.zero_()
    
        # Sum all checkpoint parameters
        for checkpoint in checkpoints:
            param.data += checkpoint['model_state_dict'][name].data
        
        param.data /= len(checkpoints)

    return average_model

def average_optimizer_states(optimizers, checkpoints: list):
    averaged_optimizers = copy.deepcopy(optimizers)
    
    for opt_idx, optimizer in enumerate(averaged_optimizers):
        state_dict = optimizer.state_dict()
        
        # Zero out the state
        for key in state_dict['state']:
            for state_key, state_value in state_dict['state'][key].items():
                if torch.is_tensor(state_value):
                    state_value.zero_()
        
        # Sum all checkpoint optimizer states
        for checkpoint in checkpoints:
            checkpoint_opt_state = checkpoint['optimizer_state'][opt_idx]
            
            for key in checkpoint_opt_state['state']:
                for state_key, state_value in checkpoint_opt_state['state'][key].items():
                    if torch.is_tensor(state_value):
                        if key not in state_dict['state']:
                            state_dict['state'][key] = {}
                        if state_key not in state_dict['state'][key]:
                            state_dict['state'][key][state_key] = torch.zeros_like(state_value)
                        state_dict['state'][key][state_key] += state_value
        
        # Average the states
        num_checkpoints = len(checkpoints)
        for key in state_dict['state']:
            for state_key, state_value in state_dict['state'][key].items():
                if torch.is_tensor(state_value):
                    state_dict['state'][key][state_key] /= num_checkpoints
        
        optimizer.load_state_dict(state_dict)
    
    return averaged_optimizers

def draw_checkpoint_landscape(last_3_checkpoints, step, val_steps, device, batch, model, grid_size=7):
    # need this last_3_checkpoints to be a list of one sized dictionaries
    
    # Extract parameter vectors (flatten all parameters)
    param_vectors = []
    for checkpoint in last_3_checkpoints:
        params = []
        for param in checkpoint['model_state_dict'].values():
            params.append(param.flatten())
        param_vector = torch.cat(params)
        param_vectors.append(param_vector)
    
    # Convert to numpy and move to CPU
    p1, p2, p3 = [p.cpu().numpy() for p in param_vectors]
    
    # Define plane using 3 points: p1, p2, p3
    # Create two vectors in the plane
    v1 = p2 - p1  # Vector from p1 to p2
    v2 = p3 - p1  # Vector from p1 to p3
    
    # Gram-Schmidt orthonormalization to create orthonormal basis
    u1 = v1 / np.linalg.norm(v1)  # First basis vector (normalized)
    
    # Second basis vector (orthogonal to u1)
    u2_unnormalized = v2 - np.dot(v2, u1) * u1
    norm = np.linalg.norm(u2_unnormalized)
    if norm < 1e-12:
        print("Landscape skipped: checkpoints nearly collinear")
        return
    u2 = u2_unnormalized / np.linalg.norm(u2_unnormalized)
    
    # Project the 3 checkpoints onto the 2D plane
    coords_2d = []
    for p in [p1, p2, p3]:
        relative_p = p - p1  # Relative to p1 (origin)
        x = np.dot(relative_p, u1)
        y = np.dot(relative_p, u2)
        coords_2d.append((x, y))
    
    # Define grid centered between p2 and p3 (the two most recent)
    center_x = (coords_2d[1][0] + coords_2d[2][0]) / 2
    center_y = (coords_2d[1][1] + coords_2d[2][1]) / 2
    
    # Calculate grid range to include all 3 points with some padding
    all_x = [coord[0] for coord in coords_2d]
    all_y = [coord[1] for coord in coords_2d]
    
    range_x = max(all_x) - min(all_x)
    range_y = max(all_y) - min(all_y)
    
    # Add padding (50% larger than the range of points)
    padding_factor = 0.5
    x_range = range_x * (1 + padding_factor)
    y_range = range_y * (1 + padding_factor)
    
    # Create grid
    x_vals = np.linspace(center_x - x_range/2, center_x + x_range/2, grid_size)
    y_vals = np.linspace(center_y - y_range/2, center_y + y_range/2, grid_size)
    
    # Save original model state
    original_state = deepcopy(model.state_dict())
    
    try:
        # Evaluate loss on grid
        loss_grid = np.zeros((grid_size, grid_size))
        
        # Track the best (minimum) loss point
        best_loss = float('inf')
        best_coords = (0, 0)
        
        print(f"Computing checkpoint landscape on {grid_size}x{grid_size} grid...")
        
        for i, x in enumerate(x_vals):
            for j, y in enumerate(y_vals):
                # Convert 2D coordinates back to parameter space
                params_2d = p1 + x * u1 + y * u2
                
                # Reshape back to original parameter shapes and load into model
                param_idx = 0
                new_state_dict = {}
                
                for key, original_param in last_3_checkpoints[0]['model_state_dict'].items():
                    param_size = original_param.numel()
                    param_data = params_2d[param_idx:param_idx + param_size]
                    new_state_dict[key] = torch.from_numpy(param_data).reshape(original_param.shape).to(device)
                    param_idx += param_size
                
                # Load parameters into model and evaluate
                model.load_state_dict(new_state_dict)
                
                current_loss = estimate_loss(model, batch, step, val_steps)

                dist.all_reduce(current_loss, op=dist.ReduceOp.AVG)
                
                loss_grid[j, i] = current_loss
                
                # Track best loss point
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_coords = (x, y)
        # Create visualization if matplotlib available
        try:
            create_landscape_plot(loss_grid, x_vals, y_vals, coords_2d, triple, step, 
                                        best_loss, best_coords)
        except Exception as e:
            print(f"Failed to create landscape plot: {e}")
        
        # return trajectory_loss, recent_checkpoint_loss
        
    finally:
        # Restore original model state
        model.load_state_dict(original_state)


def create_landscape_plot(loss_grid, x_vals, y_vals, coords_2d, triple, step, best_loss, best_coords):
    """Create and save the landscape plot."""
    plt.figure(figsize=(10, 8))
        
                # Create contour plot
    X, Y = np.meshgrid(x_vals, y_vals)
    contour = plt.contourf(X, Y, loss_grid, levels=20, cmap='viridis_r')
    plt.colorbar(contour, label='Loss')
        
    # Add contour lines
    plt.contour(X, Y, loss_grid, levels=10, colors='white', alpha=0.3, linewidths=0.5)

    
        
    # Mark the 3 checkpoints
    colors = ['red', 'blue', 'green']
    for i, ((x, y), color) in enumerate(zip(coords_2d, colors)):
        plt.scatter(x, y, c=color, s=100, marker='o', edgecolors='white', linewidth=2, zorder=5)
        plt.annotate(f'CP{triple[i]["step"]}', 
                    (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, color='white', weight='bold')


    # Mark the best loss point found in the grid
    plt.scatter(best_coords[0], best_coords[1], c='red', s=150, marker='*', 
               edgecolors='white', linewidth=2, zorder=6)
    plt.annotate(f'Grid Min\n({best_loss:.4f})', 
                best_coords, xytext=(5, -15), textcoords='offset points', 
                fontsize=9, color='white', weight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
    
    # create the string for titling the image
    string_triple = ','.join(str(checkpoint['step']) for checkpoint in triple)

    plt.xlabel('Plane Coordinate 1')
    plt.ylabel('Plane Coordinate 2')
    plt.title(f'Checkpoint Landscape from Steps {string_triple}\n')
    plt.grid(True, alpha=0.3)
        # Log to wandb if available

    
    try:
        if wandb.run is not None:
            wandb.log({f"checkpoint_landscape_steps_{string_triple}": wandb.Image(plt.gcf())}, step=step)
    except Exception as e:
        print(f"Failed to log checkpoint landscape to wandb: {e}")
        
    plt.close()  # Close to free memory
