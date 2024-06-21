import time

from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import git

from helper_functions import plot_field
from mode_functions import warp_func, compute_non_orthogonality, compute_similarity
from tilt_optim_functions import compute_tilt_mode, compute_gram_tilt


# ====== Settings ====== #
prefer_gpu = False  # Use cuda-GPU if it is available

if prefer_gpu and torch.cuda.is_available():
    torch.set_default_device('cuda')

do_plot = True
plot_per_its = 10  # Plot every this many iterations
do_save_plot = False
save_path_plot = 'C:/LocalData/mode_optimization_frames'
save_path_coeffs = 'C:/LocalData'  # Where to save output

# Gaussian shape parameters
NA = 0.8  # Numerical Aperture
f_obj1_m = 12.5e-3  # Objective focal length in m
waist_m = 2 * 5.9e-3  # Fit beam profile gaussian width in m

# Mode settings
shape = (100, 100)
k_max = 2

# Coefficients
N = 3  # Sqrt of number of polynomial terms

# Optimization parameters
learning_rate = 2.5e-2
iterations = 500
tilt_weight = 0.1


# ====== Initializations ====== #
# Scaling factor for Gaussian envelope on back pupil plane
r_factor = NA * f_obj1_m / waist_m

# Initialize coefficient arrays
ax = torch.zeros((N, N)).view(1, 1, N, N)
ay = torch.zeros((N, N)).view(1, 1, N, N)
ax[0, 0, 0, 0] = 1
ay[0, 0, 0, 0] = 1
ax.requires_grad = True
ay.requires_grad = True

# Define optimizer
optimizer = torch.optim.Adam([
    {'lr': learning_rate, 'params': [ax, ay]},
], lr=learning_rate, amsgrad=True)

# Initialize arrays for error function values and terms
errors = [np.nan] * iterations
non_orthogonalities = [np.nan] * iterations
tilt_similarities = [np.nan] * iterations
progress_bar = tqdm(total=iterations)

# Initialize plot figure
if do_plot:
    plt.figure(figsize=(16, 10), dpi=90)
    plt.tight_layout()

# Compute starting conditions, for comparison
overlaps_original, tilt_corrs_original = compute_gram_tilt(r_factor, ax.detach(), ay.detach(), shape=shape,
                                                           k_min=-k_max, k_max=k_max)
non_orthogonality_original = compute_non_orthogonality(overlaps_original)
tilt_similarity_original = compute_similarity(tilt_corrs_original)


# ====== Gradient descent ====== #
for it in range(iterations):
    overlaps, tilt_corrs = compute_gram_tilt(r_factor, ax, ay, shape=shape, k_min=-k_max, k_max=k_max)

    # Compute error
    non_orthogonality = compute_non_orthogonality(overlaps)
    tilt_similarity = compute_similarity(tilt_corrs)
    error = non_orthogonality - tilt_weight * tilt_similarity

    # Save error and terms
    errors[it] = error.detach()
    non_orthogonalities[it] = non_orthogonality.detach()
    tilt_similarities[it] = tilt_similarity.detach()

    if it % plot_per_its == 0 and do_plot:
        # Original Gram matrix
        plt.subplot(2, 4, 1)
        plt.cla()
        plt.imshow(overlaps_original.detach().abs())
        plt.xlabel('$k_1$ linear index')
        plt.ylabel('$k_2$ linear index')
        plt.title(f'Original Gram matrix (normalized)\nnon-orthogonality = {non_orthogonality_original:.4f}')

        # New Gram matrix
        plt.subplot(2, 4, 2)
        plt.cla()
        plt.imshow(overlaps.detach().abs())
        plt.xlabel('$k_1$ linear index')
        plt.ylabel('$k_2$ linear index')
        plt.title(f'Gram matrix (normalized), it {it}\nnon-orthogonality = {non_orthogonality:.4f}')

        # Error convergence
        plt.subplot(2, 4, 3)
        plt.cla()
        plt.plot(errors, 'r', label='Error function')
        plt.xlim((0, iterations))
        plt.xlabel('Iteration')
        plt.ylim((-0.07, 0.26))
        plt.legend()
        plt.title('Error convergence')

        # Error term evolution
        plt.subplot(2, 4, 4)
        plt.cla()
        plt.plot(non_orthogonalities, label='non-orthogonality')
        plt.plot(tilt_similarities, label='tilt similarity')
        plt.xlim((0, iterations))
        plt.xlabel('Iteration')
        plt.ylim((0, 1))
        plt.legend()
        plt.title('Error terms')

        # Example mode 1
        plt.subplot(2, 4, 5)
        plt.cla()
        k1 = (2, 2)
        mode1 = compute_tilt_mode(shape, k1, r_factor, ax, ay).detach().squeeze()
        plot_field(mode1, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Half pupil field, k={k1}')

        # Example mode 2
        plt.subplot(2, 4, 6)
        plt.cla()
        k2 = (-3, 2)
        mode2 = compute_tilt_mode(shape, k2, r_factor, ax, ay).detach().squeeze()
        plot_field(mode2, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Half pupil field, k={k2}')

        # Example mode 3
        plt.subplot(2, 4, 7)
        plt.cla()
        k3 = (4, 0)
        mode3 = compute_tilt_mode(shape, k3, r_factor, ax, ay).detach().squeeze()
        plot_field(mode3, 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Half pupil field, k={k3}')

        # Show warp function as grid
        plt.subplot(2, 4, 8)
        plt.cla()
        # Warped grid
        x_grid = torch.linspace(-1, 0, 11).view(1, -1, 1, 1)  # Normalized x coords
        y_grid = torch.linspace(-1, 1, 21).view(-1, 1, 1, 1)  # Normalized y coords
        r_mask = x_grid * x_grid + y_grid * y_grid > 1.01
        wx_grid, wy_grid = warp_func(x_grid, y_grid, ax.detach(), ay.detach())
        wx_grid[r_mask] = np.nan
        wy_grid[r_mask] = np.nan
        # Warped arc
        phi_arc = torch.linspace(np.pi/2, 3*np.pi/2, 80)
        x_arc = torch.cos(phi_arc).view(-1, 1, 1, 1)
        y_arc = torch.sin(phi_arc).view(-1, 1, 1, 1)
        wx_arc, wy_arc = warp_func(x_arc, y_arc, ax.detach(), ay.detach())
        # Plot
        plt.plot(wx_arc.squeeze(), wy_arc.squeeze(), '-', linewidth=1)
        plt.plot(wx_grid.squeeze(), wy_grid.squeeze(), '-k', linewidth=1)
        plt.plot(wx_grid.squeeze().T, wy_grid.squeeze().T, '-k', linewidth=1)
        plt.plot()
        plt.xlim((-1.25, 0.1))
        plt.ylim((-1.25, 1.25))
        plt.gca().set_aspect(1)
        plt.xlabel('warped x')
        plt.ylabel('warped y')
        plt.title('Warped pupil coords')
        plt.pause(1e-2)

        if do_save_plot:
            plt.savefig(f'{save_path_plot}/mode_optimization_it{it:04d}.png')

    # Gradient descent step
    error.backward()
    optimizer.step()
    optimizer.zero_grad()

    progress_bar.update()


# ====== Show and save ====== #
overlaps_big, tilt_corrs_big = compute_gram_tilt(r_factor, ax.detach(), ay.detach(), shape=shape, k_min=-6, k_max=6)
non_orthogonality_big = compute_non_orthogonality(overlaps_big)
tilt_similarity_big = compute_similarity(tilt_corrs_big)
print('\nax:\n', ax.detach())
print('\nay:\n', ay.detach())
print('\nnon-orthogonality:', non_orthogonality_big)
print('\ntilt similarity:', tilt_similarity_big)
repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha

np.savez(f'{save_path_coeffs}/optimized-warp-coeffs_t{round(time.time())}',
         axy=[ax.detach(), ay.detach()],
         dimension_labels=['short', 'long'],
         non_orthogonality=non_orthogonality_big.numpy(),
         tilt_similarity=tilt_similarity_big.numpy(),
         non_orthogonality_original=non_orthogonality_original.numpy(),
         tilt_similarity_original=tilt_similarity_original.numpy(),
         NA=NA,
         f_obj1_m=f_obj1_m,
         waist_m=waist_m,
         r_factor=r_factor,
         shape=shape,
         k_max=k_max,
         N=N,
         learning_rate=learning_rate,
         iterations=iterations,
         tilt_weight=tilt_weight,
         time=time.time(),
         git_sha=git_sha)


# ====== Plot final Gram matrix ====== #
if do_plot:
    plt.figure()
    plt.imshow(overlaps_big.abs())
    plt.xlabel('$k_1$ linear index')
    plt.ylabel('$k_2$ linear index')
    plt.title(f'Gram matrix (normalized)\nnon-orthogonality = {non_orthogonality_big:.4f}')
    plt.show()

pass
