import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

from train_EncoderDecoderConvLSTM import *  # Import your model and related classes

# Parameters (ensure these match the values used during training)
H_prev = 20  # Number of previous frames used as input
H_post = 5   # Number of future frames to predict
input_dim = 1  # Number of input channels (e.g., grayscale images)
hidden_dim = 64  # Hidden dimension used in the model
kernel_size = 3  # Kernel size for convolution
num_layers = 1   # Number of layers in ConvLSTM

# Load the trained model
model = EncoderDecoderConvLSTM(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    kernel_size=kernel_size,
    num_layers=num_layers
)

model_file = 'trained_models/EncoderDecoderConvLSTM_model.pth'
model.load_state_dict(torch.load(model_file))
model.eval()  # Set model to evaluation mode

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the test data
test_sequence = np.load('./data/imgs/img_40.npy')  # Shape: (T_test, H, W)
T_test, H, W = test_sequence.shape

# Ensure that the test sequence is long enough
if T_test < H_prev + H_post:
    raise ValueError(f"Test sequence is too short. It must be at least {H_prev + H_post} frames long.")

# Maximum starting index for the slider
max_start_idx = T_test - H_prev - H_post

# Animator class to manage animations and updates
class Animator:
    def __init__(self, fig, axes, model, test_sequence, H_prev, H_post, device):
        self.fig = fig
        self.ax_input, self.ax_target, self.ax_predicted = axes
        self.model = model
        self.test_sequence = test_sequence
        self.H_prev = H_prev
        self.H_post = H_post
        self.device = device

        self.T_test = test_sequence.shape[0]
        self.max_start_idx = self.T_test - H_prev - H_post

        # Initial start_idx
        self.start_idx = 0

        # Prepare initial data
        self.update_data(self.start_idx)

        # Initialize images
        self.input_im = self.ax_input.imshow(self.input_frames_norm[0], cmap='gray', vmin=0, vmax=1)
        self.ax_input.set_title(f'Input Frame {self.start_idx + 1}')
        self.ax_input.axis('off')

        self.target_im = self.ax_target.imshow(self.target_frames_norm[0], cmap='gray', vmin=0, vmax=1)
        self.ax_target.set_title(f'Target Frame {self.start_idx + H_prev + 1}')
        self.ax_target.axis('off')

        self.predicted_im = self.ax_predicted.imshow(self.predicted_frames_norm[0], cmap='gray', vmin=0, vmax=1)
        self.ax_predicted.set_title(f'Predicted Frame {self.start_idx + H_prev + 1}')
        self.ax_predicted.axis('off')

        # Create the animation
        self.anim = FuncAnimation(self.fig, self.update_animation, frames=self.total_frames, interval=500, blit=False)

        # Create the slider
        self.ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(self.ax_slider, 'Start Index', 0, self.max_start_idx, valinit=self.start_idx, valstep=1)
        self.slider.on_changed(self.on_slider_change)

    def normalize_frames(self, frames):
        min_val = frames.min()
        max_val = frames.max()
        frames_norm = (frames - min_val) / (max_val - min_val + 1e-8)
        return frames_norm

    def update_data(self, start_idx):
        self.start_idx = start_idx

        # Prepare x_test and y_test
        self.x_test = self.test_sequence[start_idx:start_idx + self.H_prev]  # Shape: (H_prev, H, W)
        self.y_test = self.test_sequence[start_idx + self.H_prev:start_idx + self.H_prev + self.H_post]  # Shape: (H_post, H, W)

        # Convert to tensors and move to device
        x_test_tensor = torch.from_numpy(self.x_test).float().unsqueeze(1)  # Shape: (H_prev, 1, H, W)
        x_test_tensor = x_test_tensor.unsqueeze(0).to(self.device)  # Shape: (1, H_prev, 1, H, W)

        # Run inference
        with torch.no_grad():
            output = self.model(x_test_tensor, target_len=self.H_post)  # Output shape: (1, H_post, 1, H, W)
        self.predicted_frames = output.cpu().numpy().squeeze(0).squeeze(1)  # Shape: (H_post, H, W)

        # Normalize frames for display
        self.input_frames_norm = self.normalize_frames(self.x_test)
        self.target_frames_norm = self.normalize_frames(self.y_test)
        self.predicted_frames_norm = self.normalize_frames(self.predicted_frames)

        # Total number of frames for the animation
        self.total_frames = max(self.H_prev, self.H_post)

    def update_animation(self, frame):
        # Update input frames
        if frame < self.H_prev:
            self.input_im.set_data(self.input_frames_norm[frame])
            self.ax_input.set_title(f'Input Frame {self.start_idx + frame + 1}')
        else:
            # After H_prev frames, keep displaying the last input frame
            self.input_im.set_data(self.input_frames_norm[-1])
            self.ax_input.set_title(f'Input Frame {self.start_idx + self.H_prev}')

        # Update target frames
        if frame < self.H_post:
            self.target_im.set_data(self.target_frames_norm[frame])
            self.ax_target.set_title(f'Target Frame {self.start_idx + self.H_prev + frame + 1}')
        else:
            # After H_post frames, hold the last frame
            self.target_im.set_data(self.target_frames_norm[-1])
            self.ax_target.set_title(f'Target Frame {self.start_idx + self.H_prev + self.H_post}')

        # Update predicted frames
        if frame < self.H_post:
            self.predicted_im.set_data(self.predicted_frames_norm[frame])
            self.ax_predicted.set_title(f'Predicted Frame {self.start_idx + self.H_prev + frame + 1}')
        else:
            # After H_post frames, hold the last frame
            self.predicted_im.set_data(self.predicted_frames_norm[-1])
            self.ax_predicted.set_title(f'Predicted Frame {self.start_idx + self.H_prev + self.H_post}')

        return self.input_im, self.target_im, self.predicted_im

    def on_slider_change(self, val):
        # Get the new start index from the slider
        new_start_idx = int(self.slider.val)
        # Update data with the new start index
        self.update_data(new_start_idx)
        # Update the images with the new data
        self.input_im.set_data(self.input_frames_norm[0])
        self.ax_input.set_title(f'Input Frame {self.start_idx + 1}')

        self.target_im.set_data(self.target_frames_norm[0])
        self.ax_target.set_title(f'Target Frame {self.start_idx + self.H_prev + 1}')

        self.predicted_im.set_data(self.predicted_frames_norm[0])
        self.ax_predicted.set_title(f'Predicted Frame {self.start_idx + self.H_prev + 1}')

        # Reset the animation
        self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, self.update_animation, frames=self.total_frames, interval=500, blit=False)
        self.anim.event_source.start()

# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
ax_input, ax_target, ax_predicted = axes

# Create the Animator
animator = Animator(fig, axes, model, test_sequence, H_prev, H_post, device)

plt.show()
