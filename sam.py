import cv2
import numpy as np
from PIL import Image
import matplotlib
from segment_anything.segment_anything import sam_model_registry, SamPredictor

matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, jaccard_score
import numpy as np
import matplotlib.pyplot as plt


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
    print(logits)
    plot_roc_curve_for_image(logits[2])

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


def sigmoid(x):
    """Apply sigmoid function to logits to get probabilities."""
    return 1 / (1 + np.exp(-x))


def plot_roc_curve_for_image(logits):
    """
    Plot ROC curve based on logits and ground truth masks.

    Args:
    - logits (numpy.ndarray): The raw output from the model for each pixel.
    """
    # Convert logits to probabilities using sigmoid
    probabilities = sigmoid(logits)
    ground_truth_mask_path = "ground_truth_masks\\ground_truth_mask_1351.png"
    ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)
    ground_truth_mask_float = ground_truth_mask.astype(np.float32)

    # Normalize the mask to be in the range [0, 1]
    ground_truth_mask = ground_truth_mask_float / 255.0

    # Ensure that aspect ratios are maintained before resizing
    logits_aspect_ratio = logits.shape[1] / logits.shape[0]
    ground_truth_aspect_ratio = ground_truth_mask.shape[1] / ground_truth_mask.shape[0]

    if abs(logits_aspect_ratio - ground_truth_aspect_ratio) > 0.01:  # Threshold for mismatch
        print("Aspect ratio mismatch detected. Consider padding instead of resizing.")

    # Now resize if necessary
    if ground_truth_mask.shape != logits.shape[:2]:
        # Only resize if aspect ratio is the same, otherwise consider alternative solutions
        ground_truth_mask = cv2.resize(ground_truth_mask, (logits.shape[0], logits.shape[1]),
                                       interpolation=cv2.INTER_NEAREST)

    # Flatten the probabilities and ground truth for ROC curve computation
    probabilities_flat = probabilities.flatten()
    ground_truth_flat = ground_truth_mask.flatten()

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(ground_truth_flat, probabilities_flat)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold using Youden's J statistic
    best_threshold = find_best_threshold_youden(fpr, tpr, thresholds)
    print(f"Best threshold (Youden's J): {best_threshold}")

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.scatter(fpr[np.argmax(tpr - fpr)], tpr[np.argmax(tpr - fpr)], color='red',
                label=f'Best Threshold: {best_threshold:.2f}')
    plt.legend(loc="lower right")
    plt.savefig('ground_truth_masks\\roc_curve_1044.png', format='png', dpi=300)
    plt.show()

    # Binarize the predicted probabilities using the best threshold
    predicted_binary_mask = (probabilities > best_threshold).astype(np.uint8)

    # Flatten the binarized predictions and compute IoU
    ground_truth_binary = ground_truth_mask.astype(np.uint8)
    predicted_binary_flat = predicted_binary_mask.flatten()
    ground_truth_binary_flat = ground_truth_binary.flatten()
    iou_score = jaccard_score(ground_truth_binary_flat, predicted_binary_flat)
    print(f"IoU score: {iou_score:.4f}")

    # Plot predicted binary mask and ground truth binary mask side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the predicted binary mask
    axes[0].imshow(predicted_binary_mask, cmap='gray')
    axes[0].set_title('Predicted Binary Mask')
    axes[0].axis('off')

    # Plot the ground truth binary mask
    axes[1].imshow(ground_truth_binary, cmap='gray')
    axes[1].set_title('Ground Truth Binary Mask')
    axes[1].axis('off')

    # Save the comparison image
    plt.savefig('ground_truth_masks\\comparison_mask_1351.png', format='png', dpi=300)
    plt.show()

    # Plot the probability heat map
    plt.figure()
    plt.title('Probability Heat Map')
    plt.imshow(probabilities, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    plt.savefig('ground_truth_masks\\probability_heatmap_1076.png', format='png', dpi=300)
    plt.show()


def find_best_threshold_youden(fpr, tpr, thresholds):
    """
    Find the best threshold using Youden's J statistic.

    Args:
    - fpr (array): False positive rate.
    - tpr (array): True positive rate.
    - thresholds (array): Thresholds used to compute fpr and tpr.

    Returns:
    - best_threshold (float): The threshold that maximizes Youden's J statistic.
    """
    # Calculate Youden's J statistic
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)  # Index of the maximum J statistic
    best_threshold = thresholds[best_idx]
    return best_threshold
