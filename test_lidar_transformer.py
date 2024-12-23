import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from train_LidarTransformerModel import LidarTransformerModel  # Adjust the import path as needed

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters (ensure these match the values used during training)
H_prev = 20   # Number of previous scans used as input
H_post = 10   # Number of future scans to predict
input_dim = 720  # Number of distance measurements per LIDAR scan
d_model = 512     # Embedding dimension
nhead = 8         # Number of attention heads
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1

xrange_plot = [-7, +7]
yrange_plot = [-7, +7]

# Initialize the model
model = LidarTransformerModel(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout
)

# Load the trained model weights
model_file = 'trained_models/lidar_transformer_model.pth'
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()  # Set model to evaluation mode

# Move model to the appropriate device
model.to(device)

# Load the test data
test_sequence = np.load('./data/dists/dist_0.npy')  # Shape: (T_test, 720)
T_test, input_dim = test_sequence.shape

# Ensure that the test sequence is long enough
if T_test < H_prev + H_post:
    raise ValueError(f"Test sequence is too short. It must be at least {H_prev + H_post} frames long.")

# Maximum starting index for the slider
max_start_idx = T_test - H_prev - H_post

# Define the theta range corresponding to the LIDAR measurements
theta_range = np.linspace(-np.pi, np.pi, input_dim, endpoint=False)  # Shape: (720,)

def visualize_lidar_predictions(model, test_sequence, H_prev, H_post):
    T_test, input_dim = test_sequence.shape

    # Initialize variables in the enclosing scope
    start_idx = 0
    input_frame_idx = H_prev - 1  # Start with the last input frame
    pred_frame_idx = 0  # Start with the first predicted frame
    x_input = None
    y_input = None
    x_target = None
    y_target = None
    x_predicted = None
    y_predicted = None

    # Prepare the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_input, ax_target, ax_predicted = axes

    # Adjust layout to make room for the sliders and buttons
    plt.subplots_adjust(bottom=0.25)  # Adjusted bottom margin to fit widgets below the plots

    # Create start index slider just below the plots
    ax_slider_start_idx = plt.axes([0.4, 0.15, 0.2, 0.03])
    slider_start_idx = plt.Slider(ax_slider_start_idx, 'Start Index', 0, max_start_idx, valinit=start_idx, valstep=1)

    # Function to update data and visualization
    def update(val):
        nonlocal start_idx, input_frame_idx, pred_frame_idx
        nonlocal x_input, y_input, x_target, y_target, x_predicted, y_predicted
        start_idx = int(slider_start_idx.val)

        # Prepare the input tensor
        x_test = test_sequence[start_idx:start_idx + H_prev]  # Shape: (H_prev, 720)
        y_test = test_sequence[start_idx + H_prev:start_idx + H_prev + H_post]  # Shape: (H_post, 720)

        # Convert to tensors
        x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(0).to(device)  # Shape: (1, H_prev, 720)
        y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(0).to(device)  # Shape: (1, H_post, 720)

        # Prepare input and target sequences
        y_input = y_test_tensor[:, :-1, :]  # Exclude last time step for decoder input
        y_target_tensor = y_test_tensor[:, 1:, :]  # Exclude first time step for target

        # Run inference
        with torch.no_grad():
            output = model(x_test_tensor, y_input)  # Output shape: (1, H_post - 1, 720)

        # Move tensors to CPU
        x_input_seq = x_test_tensor.cpu().numpy()  # Shape: (1, H_prev, 720)
        y_target_seq = y_target_tensor.cpu().numpy()      # Shape: (1, H_post - 1, 720)
        output_seq = output.cpu().numpy()          # Shape: (1, H_post - 1, 720)

        # Reset frame indices
        input_frame_idx = H_prev - 1  # Start with the last input frame
        pred_frame_idx = 0  # Start with the first predicted frame

        # Convert the selected frames to (x, y) coordinates
        def convert_to_xy(distances):
            x_coords = distances * np.cos(theta_range)
            y_coords = distances * np.sin(theta_range)
            return x_coords, y_coords

        # Precompute coordinates for input frames
        x_input_frames = []
        y_input_frames = []
        for frame in x_input_seq[0]:
            x_coords, y_coords = convert_to_xy(frame)
            x_input_frames.append(x_coords)
            y_input_frames.append(y_coords)

        # Precompute coordinates for target frames
        x_target_frames = []
        y_target_frames = []
        for frame in y_target_seq[0]:
            x_coords, y_coords = convert_to_xy(frame)
            x_target_frames.append(x_coords)
            y_target_frames.append(y_coords)

        # Precompute coordinates for predicted frames
        x_predicted_frames = []
        y_predicted_frames = []
        for frame in output_seq[0]:
            x_coords, y_coords = convert_to_xy(frame)
            x_predicted_frames.append(x_coords)
            y_predicted_frames.append(y_coords)

        x_input = x_input_frames
        y_input = y_input_frames
        x_target = x_target_frames
        y_target = y_target_frames
        x_predicted = x_predicted_frames
        y_predicted = y_predicted_frames

        # Update the visualization
        update_visualization()

    # Initialize the visualization
    def update_visualization():
        nonlocal input_frame_idx, pred_frame_idx, start_idx

        # Input frame
        ax_input.clear()
        ax_input.set_xlim(xrange_plot[0], xrange_plot[1])
        ax_input.set_ylim(yrange_plot[0], yrange_plot[1])
        ax_input.set_autoscale_on(False)
        ax_input.scatter(x_input[input_frame_idx], y_input[input_frame_idx], s=1)
        ax_input.set_title(f'Input Scan {start_idx + input_frame_idx + 1}')
        ax_input.set_xlabel('X')
        ax_input.set_ylabel('Y')
        ax_input.set_aspect('equal', adjustable='box')

        # Target frame
        ax_target.clear()
        ax_target.set_xlim(xrange_plot[0], xrange_plot[1])
        ax_target.set_ylim(yrange_plot[0], yrange_plot[1])
        ax_target.set_autoscale_on(False)
        ax_target.scatter(x_target[pred_frame_idx], y_target[pred_frame_idx], s=1, color='green')
        ax_target.set_title(f'Target Scan {start_idx + H_prev + pred_frame_idx + 1}')
        ax_target.set_xlabel('X')
        ax_target.set_ylabel('Y')
        ax_target.set_aspect('equal', adjustable='box')

        # Predicted frame
        ax_predicted.clear()
        ax_predicted.set_xlim(xrange_plot[0], xrange_plot[1])
        ax_predicted.set_ylim(yrange_plot[0], yrange_plot[1])
        ax_predicted.set_autoscale_on(False)
        ax_predicted.scatter(x_predicted[pred_frame_idx], y_predicted[pred_frame_idx], s=1, color='red')
        ax_predicted.set_title(f'Predicted Scan {start_idx + H_prev + pred_frame_idx + 1}')
        ax_predicted.set_xlabel('X')
        ax_predicted.set_ylabel('Y')
        ax_predicted.set_aspect('equal', adjustable='box')

        fig.canvas.draw_idle()

    # Button click event handlers for input frames
    def next_input_frame(event):
        nonlocal input_frame_idx
        if input_frame_idx < H_prev - 1:
            input_frame_idx += 1
            update_visualization()

    def prev_input_frame(event):
        nonlocal input_frame_idx
        if input_frame_idx > 0:
            input_frame_idx -= 1
            update_visualization()

    # Button click event handlers for predicted frames
    def next_pred_frame(event):
        nonlocal pred_frame_idx
        if pred_frame_idx < len(x_predicted) - 1:
            pred_frame_idx += 1
            update_visualization()

    def prev_pred_frame(event):
        nonlocal pred_frame_idx
        if pred_frame_idx > 0:
            pred_frame_idx -= 1
            update_visualization()

    # Create buttons for input frames
    ax_button_prev_input = plt.axes([0.1, 0.1, 0.1, 0.04])
    button_prev_input = Button(ax_button_prev_input, 'Prev Input Frame')
    button_prev_input.on_clicked(prev_input_frame)

    ax_button_next_input = plt.axes([0.22, 0.1, 0.1, 0.04])
    button_next_input = Button(ax_button_next_input, 'Next Input Frame')
    button_next_input.on_clicked(next_input_frame)

    # Label for input frame buttons
    ax_label_input = plt.axes([0.1, 0.05, 0.22, 0.03])
    ax_label_input.axis('off')
    ax_label_input.text(0.5, 0.5, 'Input Frame', ha='center', va='center')

    # Create buttons for predicted frames
    ax_button_prev_pred = plt.axes([0.68, 0.1, 0.1, 0.04])
    button_prev_pred = Button(ax_button_prev_pred, 'Prev Pred Frame')
    button_prev_pred.on_clicked(prev_pred_frame)

    ax_button_next_pred = plt.axes([0.8, 0.1, 0.1, 0.04])
    button_next_pred = Button(ax_button_next_pred, 'Next Pred Frame')
    button_next_pred.on_clicked(next_pred_frame)

    # Label for predicted frame buttons
    ax_label_pred = plt.axes([0.68, 0.05, 0.22, 0.03])
    ax_label_pred.axis('off')
    ax_label_pred.text(0.5, 0.5, 'Predicted Frame', ha='center', va='center')

    # Connect the update function to the start index slider
    slider_start_idx.on_changed(update)

    # Initial update
    update(0)

    plt.show()

# Call the visualization function
visualize_lidar_predictions(model, test_sequence, H_prev, H_post)
