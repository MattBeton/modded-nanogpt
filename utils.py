import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Set, Tuple
import torch
import torch.distributed as dist
from copy import deepcopy
import copy
import wandb
from shared import get_window_size_blocks
import math

def estimate_loss(model, batch, step, val_steps):
    loss = 0
    for [inputs, targets] in batch:
        loss += model(inputs, targets, get_window_size_blocks(step))
    assert len(batch) == val_steps
    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
    return loss / len(batch)

def build_param_owner_maps(model: torch.nn.Module, optimizers) -> Tuple[Dict[str, Optional[int]], Dict[int, Optional[int]]]:
    """
    Returns:
      name_to_opt: mapping of model.state_dict() *parameter names* -> optimizer index (or None if not found)
      id_to_opt:   mapping of id(parameter tensor object)         -> optimizer index (or None if not found)
    Buffers (e.g., RoPE cos/sin) are not owned by any optimizer and will map to None.
    """
    id_to_opt: Dict[int, Optional[int]] = {}
    for opt_idx, opt in enumerate(optimizers):
        for group in opt.param_groups:
            for p in group["params"]:
                id_to_opt[id(p)] = opt_idx
    name_to_opt: Dict[str, Optional[int]] = {}
    for name, p in model.named_parameters():
        name_to_opt[name] = id_to_opt.get(id(p), None)
    return name_to_opt, id_to_opt

@torch.no_grad()
def averaged_state_dict(
    checkpoints: list,
    include_optimizers: Optional[Set[int]] = None,
    name_to_opt: Optional[Dict[str, Optional[int]]] = None,
) -> dict[str, torch.Tensor]:
    assert len(checkpoints) > 0

    keys = list(checkpoints[0]['model_state_dict'].keys())
    avg: dict[str, torch.Tensor] = {}

    for k in keys:
        # If filtering by optimizer ownership, skip keys not owned by requested optimizers.
        if include_optimizers is not None and name_to_opt is not None:
            owner = name_to_opt.get(k, None)
            if owner not in include_optimizers:
                # skip averaging for this key
                continue
        acc = None
        for ckpt in checkpoints:
            t = ckpt["model_state_dict"][k]
            if t.device.type != "cpu":
                t = t.detach().cpu()
            t = t.to(torch.float32)
            acc = t.clone() if acc is None else acc.add_(t)
        avg[k] = acc.div_(len(checkpoints))
    return avg

@torch.no_grad()
def load_state_dict_inplace(model: torch.nn.Module, avg_state_cpu: dict[str, torch.Tensor]) -> None:
    """
    Copies `avg_state_cpu` into `model` *in place*, preserving parameter objects.
    Casts/dtypes are preserved per-parameter.
    """
    msd = model.state_dict()
    for k, dest in msd.items():
        src = avg_state_cpu.get(k, None)
        if src is None:
            continue  # allow partial updates
        src = src.to(device=dest.device, dtype=dest.dtype, non_blocking=True)
        dest.copy_(src, non_blocking=True)

def average_optimizer_states(
    optimizers,
    checkpoints: list,
    only_optimizers: Optional[Set[int]] = None,
):
    """
    In-place average of optimizer states across `checkpoints` for the selected optimizers.
    Robustly aligns per-parameter state by (group_idx, param_idx) position rather than integer keys.
    This avoids mismatches across independent state_dict enumerations.
    """
    num_ckpts = len(checkpoints)
    if num_ckpts == 0:
        return optimizers

    for opt_idx, optimizer in enumerate(optimizers):
        if only_optimizers is not None and opt_idx not in only_optimizers:
            continue  # don't break; skip this optimizer only

        base_sd = optimizer.state_dict()
        base_groups = base_sd["param_groups"]
        base_state = base_sd["state"]

        # Build traversal of (group_idx, param_idx, base_key)
        idx_triplets = []
        for gi, g in enumerate(base_groups):
            for pi, base_key in enumerate(g["params"]):
                idx_triplets.append((gi, pi, base_key))
                # Ensure entry exists
                if base_key not in base_state:
                    base_state[base_key] = {}

        # Accumulators: per-parameter dict of state_name -> accumulated tensor/number
        accum: Dict[int, Dict[str, object]] = {bk: {} for _, _, bk in idx_triplets}

        for ckpt in checkpoints:
            ck_opt_sd = ckpt["optimizer_state"][opt_idx]
            ck_groups = ck_opt_sd["param_groups"]
            ck_state = ck_opt_sd["state"]

            for gi, pi, base_key in idx_triplets:
                if gi >= len(ck_groups) or pi >= len(ck_groups[gi]["params"]):
                    continue
                ckey = ck_groups[gi]["params"][pi]
                if ckey not in ck_state:
                    continue
                c_entry = ck_state[ckey]
                for sname, sval in c_entry.items():
                    # We handle tensors (float/int) and python numbers (step counters).
                    if torch.is_tensor(sval):
                        if torch.is_floating_point(sval):
                            val = sval.detach().to(torch.float32, copy=False).cpu()
                        else:
                            val = sval.detach().to(torch.int64, copy=False).cpu()
                        prev = accum[base_key].get(sname)
                        accum[base_key][sname] = (val.clone() if prev is None else prev + val)
                    elif isinstance(sval, (int, float)):
                        prev = accum[base_key].get(sname)
                        accum[base_key][sname] = (sval if prev is None else prev + sval)
                    # else: ignore non-numeric state types

        # Write back averaged states, preserving dtype/device of existing base state entries when possible.
        for _, _, base_key in idx_triplets:
            if not accum[base_key]:
                continue
            dest = base_state.get(base_key, {})
            for sname, acc_val in accum[base_key].items():
                if torch.is_tensor(acc_val):
                    # Choose target dtype/device from existing dest if present
                    if sname in dest and torch.is_tensor(dest[sname]):
                        target = dest[sname]
                        if torch.is_floating_point(target):
                            avg_val = (acc_val / num_ckpts).to(dtype=target.dtype, device=target.device)
                        else:
                            avg_val = torch.div(acc_val, num_ckpts, rounding_mode="floor").to(dtype=target.dtype, device=target.device)
                    else:
                        # Default to float average on CPU if we have no hint
                        avg_val = (acc_val / num_ckpts)
                    dest[sname] = avg_val
                else:
                    # Python numbers: simple average and cast back to int for counters like "step"
                    mean_val = acc_val / num_ckpts
                    if sname == "step":
                        mean_val = int(mean_val)
                    dest[sname] = mean_val
            base_state[base_key] = dest

        # Load averaged state back into the *same* optimizer (in-place)
        optimizer.load_state_dict(base_sd)

    return optimizers

@torch.no_grad()
def set_muon_velocity_from_diff(
    muon_optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    diff_state_cpu: Dict[str, torch.Tensor],
    *,
    scale: float,
    world_size: int,
    rank: int,
    param_to_name: Dict[int, str],
) -> float:
    """
    For each Muon-optimized parameter owned by this RANK, set the momentum_buffer
    (Muon’s velocity term) to `scale * (avg_param - current_param)`.

    Args:
      muon_optimizer: the Muon optimizer (optimizer2 in your script)
      model: the compiled model (only used to pick a device for the reduction)
      diff_state_cpu: dict[name -> Tensor(cpu, fp32)] with (avg - current) per parameter
      scale: alpha, e.g. 0.1
      world_size, rank: distributed ownership info (param idx % world_size == rank)
      param_to_name: mapping id(param) -> "module.parameter" string name

    Returns:
      Global L2 norm of the pseudo-velocity (for logging).
    """
    device = next(model.parameters()).device
    total_norm_sq = torch.tensor(0.0, device=device)

    for group in muon_optimizer.param_groups:
        params = group["params"]
        for pi, p in enumerate(params):
            # Update only on the owner rank for this parameter index
            if (pi % world_size) != rank:
                continue

            name = param_to_name.get(id(p), None)
            if name is None:
                continue
            diff_cpu = diff_state_cpu.get(name, None)
            if diff_cpu is None:
                continue

            # Ensure state entry exists and is on the right device/dtype
            st = muon_optimizer.state[p]
            mb = st.get("momentum_buffer", None)
            if mb is None or mb.shape != p.shape or mb.dtype != p.dtype or mb.device != p.device:
                st["momentum_buffer"] = torch.zeros_like(p)
                mb = st["momentum_buffer"]

            pseudo = diff_cpu.to(device=p.device, dtype=p.dtype, non_blocking=True).mul_(scale)
            mb.copy_(pseudo)
            total_norm_sq += pseudo.float().pow(2).sum()

    # Aggregate across ranks for logging
    dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
    return float(total_norm_sq.sqrt().item())

@torch.no_grad()
def average_models(
    model,
    checkpoints: list,
    include_optimizers: Optional[Set[int]] = None,
    name_to_opt: Optional[Dict[str, Optional[int]]] = None,
):
    avg_state_cpu = averaged_state_dict(
        checkpoints,
        include_optimizers=include_optimizers,
        name_to_opt=name_to_opt,
    )  # CPU, fp32 (possibly partial)    averaged_model = copy.deepcopy(model)             # keep compiled wrapper semantics same as before
    load_state_dict_inplace(averaged_model, avg_state_cpu)
    return averaged_model

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
