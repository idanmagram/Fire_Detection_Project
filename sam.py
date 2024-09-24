import numpy as np
from PIL import Image
import matplotlib
from segment_anything.segment_anything import sam_model_registry, SamPredictor
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def creating_mask(input_array, sliding_windows):
    """
       Create a mask based on fire detection in the input array using sliding windows.

       Args:
           input_array (numpy.ndarray): 2D array representing the image where fire is detected (1) or not (0).
           sliding_windows (list): List of sliding window coordinates (each window is tuple with 4
           coordinates that represent rectangle).

       Returns:
           tuple: Coordinates of detected fire points and their corresponding labels.
       """
    input_labels = np.array([])
    input_points = np.array([]).reshape(0, 2)

    i = 0
    j = 0
    found_fire = False

    for window in sliding_windows:
        if j < len(input_array[0]) - 1:
            j += 1
        elif i < len(input_array) - 1:
            i += 1
            j = 0
        x1, y1, x2, y2 = window
        middle_x = (x1 + x2) // 2
        middle_y = (y1 + y2) // 2
        if input_array[i][j] == 1:
            found_fire = True
            input_points = np.vstack([input_points, [middle_x, middle_y]])
            input_labels = np.append(input_labels, 1)
        # elif input_array[i][j] == 0:
        #     input_points = np.vstack([input_points, [middle_x, middle_y]])
        #     input_labels = np.append(input_labels, 0)
    if found_fire == False:
        print("Not detected fire")
        exit()
    return input_points, input_labels


def show_mask(mask, ax, random_color=False):
    """
           Display a mask on the provided axes.

           Args:
               mask (numpy.ndarray): The mask to display.
               ax (matplotlib.axes.Axes): The axes to plot the mask on.
               random_color (bool): If True, use a random color for the mask.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
       Display points on the provided axes.

       Args:
           coords (numpy.ndarray): Coordinates of the points to display.
           labels (numpy.ndarray): Labels corresponding to the points.
           ax (matplotlib.axes.Axes): The axes to plot the points on.
           marker_size (int): Size of the marker.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def visualize_tsne(masks, scores, image):
    """
       Visualize the segmentation masks using t-SNE.

       Args:
           masks (numpy.ndarray): Array of masks.
           scores (numpy.ndarray): Corresponding scores for each mask.
           image (numpy.ndarray): Original image to display alongside the t-SNE results.

       Returns:
           None
       """
    num_masks, H, W = masks.shape
    masks_reshaped = masks.reshape(num_masks, -1)

    # Determine a suitable perplexity
    perplexity = min(5, num_masks - 1)  # Set to 5 or fewer than num_masks

    # Apply t-SNE with the adjusted perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    masks_tsne = tsne.fit_transform(masks_reshaped)
    plt.figure(figsize=(12, 6))

    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image', fontsize=16)
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(masks_tsne[:, 0], masks_tsne[:, 1], c=scores, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Score')
    plt.title('t-SNE Visualization of Segmentation Masks', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def segment(fire_vector, image, windows):
    """
      Perform segmentation on the image based on fire detection.

      Args:
          fire_vector (numpy.ndarray): Array indicating detected fire areas.
          image (numpy.ndarray): Original image array.
          windows (list): Sliding windows for processing.

      Returns:
          None
      """
    input_points, input_labels = creating_mask(fire_vector, windows)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    image_np = np.array(image)

    predictor.set_image(image_np)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_points, input_labels, plt.gca())
    plt.axis('on')
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    masks.shape  # (number_of_masks) x H x W
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_points, input_labels, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    visualize_tsne(masks, scores, image)
