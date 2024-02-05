from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from roma.utils.utils import tensor_to_pil
import cv2
import matplotlib.pyplot as plt


from roma import roma_outdoor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_and_save_matches(im1_path, im2_path, kptsA, kptsB, mask, save_path):
    """
    Plot and save the matches between two images highlighting inliers.

    Parameters:
    - im1_path: Path to the first image.
    - im2_path: Path to the second image.
    - kptsA: Keypoints in the first image (tensor on CUDA or CPU).
    - kptsB: Keypoints in the second image (tensor on CUDA or CPU).
    - mask: Mask indicating inliers (value=1) and outliers (value=0), can be a tensor on CUDA or CPU, or a numpy array.
    - save_path: Path where the plot will be saved.
    """
    # Load images
    im1 = cv2.cvtColor(cv2.imread(im1_path), cv2.COLOR_BGR2RGB)
    im2 = cv2.cvtColor(cv2.imread(im2_path), cv2.COLOR_BGR2RGB)

    # Move keypoints tensors to CPU and convert to numpy if they are PyTorch tensors
    if isinstance(kptsA, torch.Tensor):
        kptsA = kptsA.cpu().numpy()
    if isinstance(kptsB, torch.Tensor):
        kptsB = kptsB.cpu().numpy()
    
    # Check if mask is a PyTorch tensor and convert to numpy if necessary
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy().flatten().astype(bool)
    elif isinstance(mask, np.ndarray):  # mask is already a numpy array
        mask = mask.flatten().astype(bool)  # Ensure it's flattened and boolean

    # Create a plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    ax.axis('off')

    # Draw matches
    plt.imshow(np.concatenate((im1, im2), axis=1))

    for (x1, y1), (x2, y2) in zip(kptsA[mask], kptsB[mask]):
        plt.plot([x1, x2 + im1.shape[1]], [y1, y2], color='y', linestyle='-', linewidth=1)
        plt.scatter([x1, x2 + im1.shape[1]], [y1, y2], color='y', s=3)

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()





def metrics_calculate(mask, certainty=None):
    """
    Calculate metrics for image matching including the number of inlier matches,
    inlier ratio, and optionally the average score of inliers if certainty scores are provided.

    Parameters:
    - mask: A numpy array or tensor indicating inliers (1) and outliers (0) from geometric verification.
    - certainty: An optional numpy array or tensor of certainty scores corresponding to the matches.
                 If provided, the average score of inliers is calculated.

    Returns:
    - num_inliers: The number of inlier matches.
    - inlier_ratio: The ratio of inliers to the total number of matches.
    - avg_score: The average score of inliers, or None if certainty scores are not provided.
    """
    # Ensure mask is a numpy array for compatibility
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()

    num_inliers = mask.sum()
    total_matches = mask.size
    inlier_ratio = num_inliers / total_matches

    # if certainty is not None:
    #     # Ensure certainty is a numpy array for compatibility
    #     if not isinstance(certainty, np.ndarray):
    #         certainty = certainty.cpu().numpy()
    #     inlier_scores = certainty[mask.astype(bool)]
    #     avg_score = inlier_scores.mean()
    # else:
    #     avg_score = None
    avg_score = None

    return num_inliers, inlier_ratio, avg_score


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/toronto_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/toronto_B.jpg", type=str)
    #parser.add_argument("--save_path", default="demo/roma_warp_toronto.jpg", type=str)
    parser.add_argument("--save_path", default="save/roma_warp_toronto.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    save_path = args.save_path

    # Create model
    roma_model = roma_outdoor(device=device, coarse_res=560, upsample_res=(864, 1152))

    H, W = roma_model.get_output_resolution()

    im1 = Image.open(im1_path).resize((W, H))
    im2 = Image.open(im2_path).resize((W, H))

    # Match
    warp, certainty = roma_model.match(im1_path, im2_path, device=device)
    # Custom code start 
    matches, certainty = roma_model.sample(warp, certainty)
    print("Need to check H W properly here")
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H, W, H, W)
    F, mask = cv2.findFundamentalMat(kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000)

    num_inliers, inlier_ratio, avg_score = metrics_calculate(mask, certainty)
    print("num_inliers, inlier_ratio, avg_score :", num_inliers, inlier_ratio, avg_score)
    plot_and_save_matches(im1_path, im2_path, kptsA, kptsB, mask, save_path)
    # Custom code end
    # Sampling not needed, but can be done with model.sample(warp, certainty)
    # x1 = (torch.tensor(np.array(im1)) / 255).to(device).permute(2, 0, 1)
    # x2 = (torch.tensor(np.array(im2)) / 255).to(device).permute(2, 0, 1)

    # im2_transfer_rgb = F.grid_sample(
    # x2[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
    # )[0]
    # im1_transfer_rgb = F.grid_sample(
    # x1[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
    # )[0]
    # warp_im = torch.cat((im2_transfer_rgb,im1_transfer_rgb),dim=2)
    # white_im = torch.ones((H,2*W),device=device)
    # vis_im = certainty * warp_im + (1 - certainty) * white_im
    # tensor_to_pil(vis_im, unnormalize=False).save(save_path)
    # print("SUCCESS")

