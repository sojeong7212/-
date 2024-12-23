import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# from train_EncoderDecoderConvLSTM import *  # Import your model and related classes
from train_EncoderDecoderConvLSTMWithAttention import *  # Import your model and related classes


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters (ensure these match the values used during training)
H_prev = 20  # Number of previous frames used as input
H_post = 10   # Number of future frames to predict
input_dim = 1  # Number of input channels (e.g., grayscale images)
hidden_dim = 64  # Hidden dimension used in the model
kernel_size = 3  # Kernel size for convolution
num_layers = 1   # Number of layers in ConvLSTM

# Load the trained model
# model = EncoderDecoderConvLSTM(
#     input_dim=input_dim,
#     hidden_dim=hidden_dim,
#     kernel_size=kernel_size,
#     num_layers=num_layers
# )

model = EncoderDecoderConvLSTMWithAttention(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=kernel_size,
    num_layers=num_layers
)

model_file = 'trained_models/conv_lstm_attention_model.pth'
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()  # Set model to evaluation mode

# Move model to the appropriate device
model.to(device)

# Load the test data
test_sequence = np.load('./data/imgs/img_0.npy')  # Shape: (T_test, H, W)
T_test, H, W = test_sequence.shape

# Ensure that the test sequence is long enough
if T_test < H_prev + H_post:
    raise ValueError(f"Test sequence is too short. It must be at least {H_prev + H_post} frames long.")

# Maximum starting index for the slider
max_start_idx = T_test - H_prev - H_post

def visualize_with_start_idx_slider(model, test_sequence, H_prev, H_post):
    T_test, H, W = test_sequence.shape

    # Initialize variables in the enclosing scope
    start_idx = 0
    input_frames_np = None
    target_frames_np = None
    predicted_frames_np = None
    input_frame_idx = H_prev - 1  # Start with the last input frame
    pred_frame_idx = 0  # Start with the first predicted frame

    # Prepare the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_input, ax_target, ax_predicted = axes

    # Adjust layout to make room for the sliders and buttons
    plt.subplots_adjust(bottom=0.25)  # Adjusted bottom margin to fit widgets below the images

    # Create start index slider just below the images
    ax_slider_start_idx = plt.axes([0.2, 0.15, 0.6, 0.03])
    slider_start_idx = Slider(ax_slider_start_idx, 'Start Index', 0, max_start_idx, valinit=start_idx, valstep=1)

    # Function to update data and visualization
    def update(val):
        nonlocal input_frames_np, target_frames_np, predicted_frames_np
        nonlocal input_frame_idx, pred_frame_idx, start_idx

        # Update start_idx
        start_idx = int(val)

        # Prepare the input tensor
        x_test = test_sequence[start_idx:start_idx + H_prev]  # Shape: (H_prev, H, W)
        y_test = test_sequence[start_idx + H_prev:start_idx + H_prev + H_post]  # Shape: (H_post, H, W)

        # Convert to tensors and add channel dimension
        x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(1)  # Shape: (H_prev, 1, H, W)
        y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1)  # Shape: (H_post, 1, H, W)

        # Add batch dimension
        x_test_tensor = x_test_tensor.unsqueeze(0)  # Shape: (1, H_prev, 1, H, W)
        y_test_tensor = y_test_tensor.unsqueeze(0)  # Shape: (1, H_post, 1, H, W)

        # Move tensors to the appropriate device
        x_test_tensor = x_test_tensor.to(device)
        y_test_tensor = y_test_tensor.to(device)

        # Run inference
        with torch.no_grad():
            output = model(x_test_tensor, target_len=H_post)  # Output shape: (1, H_post, 1, H, W)

        # Move output to CPU and adjust shapes
        output = output.cpu()  # Shape: (1, H_post, 1, H, W)

        # Convert tensors to numpy arrays
        input_frames_np = x_test_tensor.cpu().numpy()        # Shape: (1, H_prev, 1, H, W)
        target_frames_np = y_test_tensor.cpu().numpy()       # Shape: (1, H_post, 1, H, W)
        predicted_frames_np = output.numpy().squeeze(0).squeeze(1)   # Shape: (H_post, H, W)

        # Reset frame indices
        input_frame_idx = H_prev - 1  # Reset to last input frame
        pred_frame_idx = 0  # Reset to first predicted frame

        # Update the visualization
        update_visualization()

    # Initialize the visualization
    def update_visualization():
        nonlocal input_frame_idx, pred_frame_idx, start_idx
        # Display images based on current frame indices
        # Input frame
        input_frame = input_frames_np[0, input_frame_idx].squeeze()  # Shape: (H, W)
        ax_input.clear()
        ax_input.imshow(input_frame, cmap='gray')
        ax_input.set_title(f'Input Frame {start_idx + input_frame_idx + 1}')
        ax_input.axis('off')

        # Target frame
        target_frame = target_frames_np[0, pred_frame_idx].squeeze()
        ax_target.clear()
        ax_target.imshow(target_frame, cmap='gray')
        ax_target.set_title(f'Target Frame {start_idx + H_prev + pred_frame_idx + 1}')
        ax_target.axis('off')

        # Predicted frame
        predicted_frame = predicted_frames_np[pred_frame_idx]
        ax_predicted.clear()
        ax_predicted.imshow(predicted_frame, cmap='gray')
        ax_predicted.set_title(f'Predicted Frame {start_idx + H_prev + pred_frame_idx + 1}')
        ax_predicted.axis('off')

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
        if pred_frame_idx < predicted_frames_np.shape[0] - 1:
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

    # Initial update to display the initial frames
    update(start_idx)

    plt.show()

# Call the visualization function
visualize_with_start_idx_slider(model, test_sequence, H_prev, H_post)
