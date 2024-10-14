from multiprocessing import freeze_support
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from sympy.physics.units import time
from torchvision import datasets, models, transforms
import matplotlib
from PIL import Image, ImageDraw
import sam
import cv2
import numpy as np

matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import time



def imshow(inp, title=None):
    """
    Display an image from a Tensor.

    Parameters:
        inp (Tensor): Input image tensor in shape (C, H, W).
        title (str, optional): Title for the displayed image.
    """
    inp = inp.numpy().transpose((1, 2, 0))  # Convert Tensor to NumPy array and transpose dimensions
    mean = np.array([0.485, 0.456, 0.406])  # Mean for normalization
    std = np.array([0.229, 0.224, 0.225])  # Standard deviation for normalization
    inp = std * inp + mean  # Normalize image
    inp = np.clip(inp, 0, 1)  # Clip values to [0, 1]
    plt.imshow(inp)  # Display image
    if title is not None:
        plt.title(title)  # Set title if provided
    plt.pause(0.001)  # Pause for a brief moment to update display


def predict_image(model, device, class_names, img):
    """
    Predict the class of a given image using the model (Fire or Non_Fire).

    Parameters:
        model (nn.Module): The trained model for prediction.
        device (torch.device): Device to perform computation (CPU or GPU).
        class_names (list): Fire or Non_Fire.
        img (PIL Image): Input image for prediction.

    Returns:
        str: Predicted class name.
    """
    model.eval()  # Set model to evaluation mode
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    img = data_transforms(img)  # Apply transformations
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        return class_names[preds[0]]


def get_image_dimensions(image_path):
    """
    Get dimensions of an image.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: Width and height of the image.
    """
    with Image.open(image_path) as img:
        width, height = img.size  # Get size
    return width, height


def generate_sliding_windows(image_height, image_width):
    """
    Generate sliding windows for a given image size.

    Parameters:
        image_height (int): Height of the image.
        image_width (int): Width of the image.

    Returns:
        list: List of windows defined by their coordinates.
    """
    window_size_height = int(image_height / 4)  # Set window height
    window_size_width = int(image_width / 4)  # Set window width

    height_offset = int(image_height / 4)  # Set vertical offset
    width_offset = int(image_width / 4)  # Set horizontal offset

    windows = []
    k = 0
    # Create windows based on image dimensions
    for i in range(0, image_width - window_size_width + 1, width_offset):
        for j in range(0, image_height - window_size_height + 1, height_offset):
            window = (i, j, i + window_size_width, j + window_size_height)  # Define window
            windows.append(window)  # Append to list
            k += 1

    return windows


def draw_sliding_windows(sliding_windows, image_path):
    """
    Draw sliding windows on an image.

    Parameters:
        sliding_windows (list): List of windows defined by their coordinates.
        image_path (str): Path to the image file.

    Returns:
        None: Displays the image with drawn windows.
    """
    image = plt.imread(image_path)  # Read image
    plt.imshow(image)  # Display image
    input_points = np.array([]).reshape(0, 2)  # Array for center points

    # Draw each sliding window on the image
    for window in sliding_windows:
        x1, y1, x2, y2 = window
        color = np.random.rand(3)  # Generate random color
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color, linewidth=2, fill=False)  # Draw rectangle

        middle_x = (x1 + x2) // 2  # Calculate middle x
        middle_y = (y1 + y2) // 2  # Calculate middle y
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', linewidth=1, fill=False)
        input_points = np.vstack((input_points, [middle_x, middle_y]))  # Store center points
        plt.gca().add_patch(rect)  # Add rectangle to plot

    #plt.scatter(input_points[:, 0], input_points[:, 1], s=1, c='red')  # Draw points
    #plt.show()  # Display image with windows


def process_matrix(matrix):
    """
        Process a binary matrix to update cells based on the presence of neighbors.

        Parameters:
            matrix (np.ndarray): Input binary matrix where 1s indicate a presence and 0s indicate absence.

        Returns:
            np.ndarray: A new matrix where cells with 1s that have at least one 0 neighbor are set to 2.
        """
    new_matrix = np.zeros_like(matrix)
    new_matrix[:] = matrix

    # Define the eight possible directions around a cell
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Iterate over each cell in the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # If the cell is a 1, check its neighbors
            if matrix[i][j] == 1:
                found_zero_neighbor = False
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    # If any neighbor is out of bounds or a 0, mark that we found a zero neighbor
                    if nx < 0 or nx >= len(matrix) or ny < 0 or ny >= len(matrix[0]) or matrix[nx][ny] == 0:
                        found_zero_neighbor = True
                        break
                # If we found a zero neighbor, set the current cell to 0
                if found_zero_neighbor:
                    new_matrix[i][j] = 2

    return new_matrix


def manager_windows_predictions_and_segmentation(model, device, class_names, horizontal_windows_amount,
                                                 vertical_windows_amount, im, image_path):
    """
    Create sliding windows over an image, predict fire presence and calls segment anything.

    Parameters:
        model (nn.Module): The trained model for prediction.
        device (torch.device): Device to perform computation (CPU or GPU).
        class_names (list): List of class names for the model outputs.
        horizontal_windows_amount (int): Number of horizontal windows.
        vertical_windows_amount (int): Number of vertical windows.
        im (PIL Image): Input image for processing.
        image_path (str): Path to the input image.

    Returns:
        None: Displays the results and processes fire detection.
    """
    image_height, image_width = get_image_dimensions(image_path)
    window_width = int(im.size[0] / horizontal_windows_amount)
    window_height = int(im.size[1] / vertical_windows_amount)

    height_offset = (window_height / 4)
    width_offset = (window_width / 4)

    windows = []
    sliding_horiz_windows = 0
    sliding_vert_windows = 0

    # Create sliding windows over the image
    for i in np.arange(0, image_height - window_height + 5, height_offset):
        for j in np.arange(0, image_width - window_width + 5, width_offset):
            window = (i, j, i + window_height, j + window_width)  # Define window coordinates
            windows.append(window)
            sliding_horiz_windows += 1
        sliding_vert_windows += 1

    draw_sliding_windows(windows, image_path)
    im.show()


    fire_vector = np.zeros((sliding_vert_windows, int(sliding_horiz_windows / sliding_vert_windows)))
    i = 0
    j = 0
    found_fire = False
    # Iterate through sliding windows for predictions
    print(windows)
    for window in windows:
        cropped_image = im.crop(window)  # Crop the window from the image
        if j < ((sliding_horiz_windows / sliding_vert_windows) - 1):
            j += 1
        elif i < (sliding_vert_windows - 1):
            i += 1
            j = 0

        pred = predict_image(model, device, class_names, cropped_image)
        # In case you want to use the EfficientNet model change pred == 'Fire' to pred != 'Fire'
        if pred == 'Fire':
            fire_vector[i][j] = 1
            found_fire = True
            draw = ImageDraw.Draw(im)

    if not found_fire:
        print("No fire detected")  # Message if no fire detected
        exit()
    fire_vector_new = process_matrix(fire_vector)  # Process the fire vector

    sam.segment(fire_vector_new, im, windows)  # Call segmentation method
    return

    # Define a logarithmic function with interpolation for (0,0)
    def log_func_constrained(x, a):
        return np.where(x == 0, 0, a * np.log(x + 1e-6) + (1 - a * np.log(1 + 1e-6)))

    popt, _ = curve_fit(log_func_constrained, fpr_sorted[1:],
                        tpr_sorted[1:])  # Exclude the (0,0) point for log fitting
    # Calculate the TPR for the original FPR values using the logarithmic fit
    tpr_log_fit = log_func_constrained(fpr_sorted, *popt)

    # Calculate the area under the curve (AUC) using the trapezoidal rule
    auc = np.trapz(tpr_log_fit, fpr_sorted)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sorted, tpr_log_fit, color='r', linestyle='-',
             label='ROC curve')
    plt.scatter(fpr_sorted, tpr_sorted, color='blue', label='Original Data Points')
    plt.scatter([0, 1], [0, 1], color='g', zorder=5)
    plt.text(0.6, 0.2, f'AUC = {auc:.4f}', fontsize=12, color='blue', bbox=dict(facecolor='white', alpha=0.8))
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def main():
    """
    Main function to set up the model and start fire detection.

    Returns:
        None: Executes the fire detection process.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(weights=None)  # Load the model
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # Adjust the fully connected layer for 2 output classes
    model_ft.load_state_dict(torch.load('final_params_resnet.pt', map_location=torch.device('cpu')))
    model_ft = model_ft.to(device)

    horizontal_windows_amount = 6
    vertical_windows_amount = 4
    image_path = "fire_data\\train\\Fire\\F_1351.jpg"  # Path to the image

    im = Image.open(image_path)
    class_names = ["Fire", "Non_Fire"]
    manager_windows_predictions_and_segmentation(model_ft, device, class_names, horizontal_windows_amount,
                                                 vertical_windows_amount, im, image_path)


# Global variables
drawing = False  # True if the mouse is pressed
mode = True  # True for rectangle mode, False for brush mode
ix, iy = -1, -1  # Initial coordinates
img = None  # Initialize img globally
ground_truth_mask = None  # Initialize ground_truth_mask globally


def draw(event, x, y, flags, param):
    global ix, iy, drawing, mode, img, ground_truth_mask

    if img is None:  # Ensure img is not None before proceeding
        print("Image not loaded correctly.")
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                img_copy = img.copy()  # Safely create a copy if img is valid
                cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 255, 255), -1)
                cv2.imshow('image', img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv2.rectangle(img, (ix, iy), (x, y), (255, 255, 255), -1)
            cv2.rectangle(ground_truth_mask, (ix, iy), (x, y), 1, -1)  # Draw in mask (use 1 instead of 255)


def create_ground_truth_mask():
    global img, ground_truth_mask  # Declare them as global to use them inside draw()

    image_path = "fire_data\\train\\Fire\\F_1351.jpg"  # Change to your image path
    img = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    # Create an empty mask
    ground_truth_mask = np.zeros((height, width), dtype=np.uint8)

    # Create a window and bind the function to window events
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw)

    print("Draw rectangles over the fire areas. Press 'q' to quit.")

    while True:
        cv2.imshow('image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Save the mask
    mask_output_path = "ground_truth_masks\\ground_truth_mask_1351.png"
    cv2.imwrite(mask_output_path, ground_truth_mask * 255)  # Save as binary mask image

    # Clean up
    cv2.destroyAllWindows()


def main():
    """
    Main function to set up the model and start fire detection.

    Returns:
        None: Executes the fire detection process.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = models.resnet18(weights=None)  # Load the model
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)  # Adjust the fully connected layer for 2 output classes
    model_ft.load_state_dict(torch.load('final_params_resnet.pt', map_location=torch.device('cpu')))
    model_ft = model_ft.to(device)

    horizontal_windows_amount = 6
    vertical_windows_amount = 4
    image_path = "fire_data\\train\\Fire\\F_1351.jpg"  # Path to the image

    im = Image.open(image_path)
    class_names = ["Fire", "Non_Fire"]
    manager_windows_predictions_and_segmentation(model_ft, device, class_names, horizontal_windows_amount,
                                                 vertical_windows_amount, im, image_path)


if __name__ == '__main__':
    create_ground_truth_mask()  # First create the ground truth mask
    freeze_support()
    start_time = time.time()
    main()  # Call the main function for fire detection
    end_time = time.time()

    duration = end_time - start_time  # Calculate the duration
    print(f"Duration: {duration} seconds")
