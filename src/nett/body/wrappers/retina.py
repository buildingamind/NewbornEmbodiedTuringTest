"""
Artificial retina transformation for gym environments.
"""

from typing import Optional
import gymnasium as gym
import numpy as np
import logging
import torch
import torchvision as tv
import cv2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Retina(gym.ObservationWrapper):
    """
    A gym observation wrapper that applies an artificial retina transformation on the environment observations.

    Args:
        env (gym.Env): The environment to wrap.
        change_threshold (int): The threshold value for detecting changes in pixel intensity.
        kernel_size (tuple): The size of the Gaussian kernel used for blurring.
        sigma (float): The standard deviation of the Gaussian kernel.
        is_color (bool): Whether the observation is in color or grayscale.

    Attributes:
        change_threshold (int): The threshold value for detecting changes in pixel intensity.
        kernel_size (tuple): The size of the Gaussian kernel used for blurring.
        sigma (float): The standard deviation of the Gaussian kernel.
        num_stack (int): The number of frames to stack.
        env (gym.Env): The wrapped environment.
        stack (collections.deque): A deque to store the stacked frames.
        shape (tuple): The shape of the observation space.
        observation_space (gym.spaces.Box): The modified observation space.
        is_color (bool): Whether the observation is in color or grayscale.

    Methods:
        create_grayscale(image): Converts an image to grayscale.
        gaussianDiff(previous, current): Computes the difference between two images using Gaussian blur.
        observation(obs): Performs the DVS transformation on the observation.
        threshold(change): Applies a threshold to the change map.
        reset(**kwargs): Resets the environment and returns the initial observation.

    """

    def __init__(self, env, device: int, *args, **kwargs):
        super().__init__(env)

        try:
            if isinstance(self.env.observation_space, gym.spaces.Dict):
                key = list(self.env.observation_space.spaces.keys())[0]
                shape = self.env.observation_space.spaces[key].shape
                if len(shape) == 4:
                    _, channels, width, height = shape
                elif len(shape) == 3:
                    channels, width, height = shape
                    self.env = gym.wrappers.FrameStackObservation(env, stack_size=2)
                else:
                    raise ValueError(
                        "Unsupported observation space shape: {}".format(shape)
                    )
            else:
                shape = self.env.observation_space.shape
                if len(shape) == 4:
                    _, channels, width, height = shape
                elif len(shape) == 3:
                    channels, width, height = shape
                    self.env = gym.wrappers.FrameStackObservation(env, 2)
                else:
                    raise ValueError(
                        "Unsupported observation space shape: {}".format(shape)
                    )

            self.retina = ArtificialRetina(
                P=width,
                fovea_center=(height // 2, width // 2),
                fovea_radius=height // 4,
                device=device,
            )
            self.shape = (channels, width, height)
            self.observation_space = gym.spaces.Box(
                shape=self.shape, low=0, high=255, dtype=np.uint8
            )
        except Exception as e:
            raise e

    def observation(self, obs):
        """
        Performs the retinal transformation on the observation.

        Args:
            obs (list): The list of stacked frames.

        Returns:
            numpy.ndarray: The transformed observation.

        """

        if isinstance(self.env.observation_space, gym.spaces.Dict):
            out = {}
            for key in obs.keys():
                prev: np.ndarray = obs[key][-2]
                current: np.ndarray = obs[key][-1]
                # prev = np.transpose(obs[key][-2], (1, 2, 0))  # move channels to last dimension
                # current = np.transpose(obs[key][-1], (1, 2, 0))  # move channels to last dimension
                out[key] = self.retina.process(image=prev, next_image=current).astype(
                    np.uint8
                )
            return out
        else:
            # grab the last 2 images from the stack
            prev: np.ndarray = obs[-2]  # move channels to last dimension
            current: np.ndarray = obs[-1]  # move channels to last dimension
            # prev = np.transpose(obs[-2], (1, 2, 0))  # move channels to last dimension
            # current = np.transpose(obs[-1], (1, 2, 0))  # move channels to last dimension
            out = self.retina.process(image=prev, next_image=current)

            # change to channel first, w, h
            # out = np.transpose(out, (2, 0, 1))

            return out.astype(np.uint8)

    def reset(self, **kwargs):
        """
        Resets the environment and returns the initial observation.

        Args:
            **kwargs: Additional arguments for the reset method.

        Returns:
            numpy.ndarray: The initial observation.
        """
        initial_obs, initial_info = self.env.reset(**kwargs)
        return self.observation(initial_obs), initial_info


class ArtificialRetina:
    """
    [args]:
    image - Input RGB image path,
    P - size of the image,
    fovea_center - (x,y) coordinates of the fovea,
    fovea_radius - radius r of the fovea,
    peripheral_active_cones - x% of active cones (color cells) on the peripheral region,
    fovea_active_rods - x% of active rods (non-color cells) on the fovea,
    peripheral_gaussianBlur - enable/disable Gaussian Blur on the peripheral region,
    peripheral_gaussianBlur_kernel - Gaussian Blur kernal size,
    grad_blur - (91, 91),  # (121,121),
    visual_clutter - True,
    clutter_intensity - 0.5,
    peripheral_grayscale - apply grayscale on the peripheral region if True,
    cortical_magnifi - False,
    magnifi_strength - 0.5,
    magnifi_radius - 0.3,
    """

    def __init__(
        self,
        image=None,
        P=64,  # image size PxP # self.env.observation_space.shape
        foveation_type="dynamic",  # ['dynamic', 'static']
        dynamic_foveation_grid_size=(10, 10),
        fovea_center=(8, 8),  # x,y coordinates
        fovea_radius=16,  # must be >0
        peripheral_active_cones=0,
        fovea_active_rods=0,
        peripheral_gaussianBlur=True,  # always keep this True
        peripheral_gaussianBlur_kernel=(21, 21),
        grad_blur=(91, 91),  # (121,121),
        visual_clutter=True,
        clutter_intensity=0.5,
        peripheral_grayscale=True,
        cortical_magnifi=False,
        magnifi_strength=0.5,
        magnifi_radius=0.3,
        device: int = 0,  # GPU device ID
    ):
        self.image = image
        self.P = P
        self.foveation_type = foveation_type
        self.dynamic_foveation_grid_size = dynamic_foveation_grid_size
        self.fovea_center = fovea_center
        self.fovea_radius = fovea_radius
        self.peripheral_active_cones = peripheral_active_cones
        self.fovea_active_rods = fovea_active_rods
        self.peripheral_gaussianBlur = peripheral_gaussianBlur
        self.peripheral_gaussianBlur_kernel = peripheral_gaussianBlur_kernel
        self.grad_blur = grad_blur
        self.visual_clutter = visual_clutter
        self.clutter_intensity = clutter_intensity
        self.peripheral_grayscale = peripheral_grayscale
        self.cortical_magnifi = cortical_magnifi
        self.magnifi_strength = magnifi_strength
        self.magnifi_radius = magnifi_radius

        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )

    def process(self, image: np.ndarray, next_image: np.ndarray):
        # prev_tensor = torch.from_numpy(image).to(self.device)
        current_tensor = torch.from_numpy(next_image).to(self.device)

        # dynamically adjust the fovea location based on optic flow magnitude
        if self.foveation_type == "dynamic":
            # pass t and t+1 frames to get coordinates for dynamic foveation
            self.fovea_center = self.dynamic_fovea(
                prev_frame=image,  # prev_tensor,
                current_frame=next_image,  # current_tensor,
                grid_size=self.dynamic_foveation_grid_size,
            )

        # create retina_filter and generate parts of the retina
        self.fovea, self.peripheral_mask = self.create_retina_filter()
        # apply retinal filter on image
        retina_image = self.apply_retina_filter(current_tensor)

        # activate cones and rods in peripheral and fovea respectively
        # randomly select x% of pixels in the fovea and make them grayscale
        self.fovea_selected_indices = self.__select_random_pixels(
            percentage=self.fovea_active_rods, mask=self.fovea
        )

        retina_image = self.__apply_random_pixel_effect(
            retina_image=retina_image,
            selected_indices=self.fovea_selected_indices,
            effect="grayscale",
        )

        # randomly select y% of pixels in the peripheral and remove grayscale effect
        self.peripheral_selected_indices = self.__select_random_pixels(
            percentage=self.peripheral_active_cones,
            mask=self.peripheral_mask,
        )

        retina_image = self.__apply_random_pixel_effect(
            retina_image=retina_image,
            selected_indices=self.peripheral_selected_indices,
            effect="color",
            original_image=current_tensor,
        )

        if self.cortical_magnifi == True:
            retina_image = self.cortical_magnification(
                image=retina_image,
                strength=self.magnifi_strength,
                radius=self.magnifi_radius,
            )
        # convert to numpy
        return retina_image.cpu().numpy()

    def create_retina_filter(self):
        x: torch.Tensor
        y: torch.Tensor
        # Create a grid of coordinates
        x, y = torch.meshgrid(
            torch.arange(self.P, device=self.device),
            torch.arange(self.P, device=self.device),
            indexing="xy",
        )

        # Get fovea center and radius
        center_x, center_y = self.fovea_center
        radius: int = self.fovea_radius

        # Calculate the distance from the center for the circular fovea region
        dist_sq: torch.Tensor = (x - center_x) ** 2 + (y - center_y) ** 2

        # Create the fovea mask
        fovea: torch.Tensor = (dist_sq <= radius**2).int()

        # create mask for the peripheral region of the retina
        peripheral_mask: torch.Tensor = 1 - fovea

        return fovea, peripheral_mask

    def apply_retina_filter(self, preprocessed_image: torch.Tensor) -> torch.Tensor:
        # Initialize `img` with the original image
        img = preprocessed_image.clone()

        # define kernel
        ker = self.grad_blur if self.peripheral_gaussianBlur else (1, 1)

        # Initialize the mask with the original fovea
        mask: torch.Tensor = self.fovea.unsqueeze(0).float()  # Add channel dim
        mask = tv.transforms.functional.gaussian_blur(mask, ker)
        mask = mask.repeat(3, 1, 1)  # Repeat for 3 channels

        # Apply Gaussian blur to the entire image if enabled
        if self.peripheral_gaussianBlur:
            img = tv.transforms.functional.gaussian_blur(
                img, self.peripheral_gaussianBlur_kernel
            )

        # apply visual clutter to the entire image
        if self.visual_clutter == True:
            img = self.radial_pixel_distortion(
                image=img, distortion_intensity=self.clutter_intensity
            )

        # Convert the entire image to grayscale if enabled
        if self.peripheral_grayscale:
            img = tv.transforms.functional.rgb_to_grayscale(img, num_output_channels=3)

        # Combine the foveal and peripheral regions
        combined_image = preprocessed_image * mask + img * (1 - mask)

        return combined_image

    def radial_pixel_distortion(
        self, image: torch.Tensor, max_distortion=10, distortion_intensity=1.0
    ):
        _, rows, cols = image.shape
        distorted_image = image.clone()

        adjusted_max_distortion = max_distortion * distortion_intensity

        # Generate random radii and angles for all pixels at once
        radius = torch.rand(rows, cols, device=self.device) * adjusted_max_distortion
        angle = torch.rand(rows, cols, device=self.device) * 2 * torch.pi

        # Convert polar to Cartesian
        dx = (radius * torch.cos(angle)).long()
        dy = (radius * torch.sin(angle)).long()

        # Create coordinate grids
        x = torch.arange(cols, device=self.device)
        y = torch.arange(rows, device=self.device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

        # Calculate new pixel locations
        x_new = torch.clamp(x_grid + dx, 0, cols - 1)
        y_new = torch.clamp(y_grid + dy, 0, rows - 1)

        # Apply distortion
        distorted_image[:, y_grid, x_grid] = image[:, y_new, x_new]

        return distorted_image

    # Function to calculate optical flow and dynamically determine new fovea position
    def dynamic_fovea(
        self, prev_frame: np.ndarray, current_frame: np.ndarray, grid_size=(10, 10)
    ) -> tuple[int, int]:
        # Convert to grayscale
        prev_gray = cv2.cvtColor(
            np.transpose(prev_frame, (1, 2, 0)), cv2.COLOR_RGB2GRAY
        )
        current_gray = cv2.cvtColor(
            np.transpose(current_frame, (1, 2, 0)), cv2.COLOR_RGB2GRAY
        )

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        flow = torch.from_numpy(flow).to(self.device).permute(2, 0, 1)
        flow = flow.squeeze(0)  # Remove batch dimension

        # Calculate magnitude of 2D vectors (flow vector in this case)
        mag = torch.norm(flow, dim=0)

        # Initialize grid for average magnitude calculation
        h, w = mag.shape
        grid_h, grid_w = grid_size

        # Calculate average magnitude for each grid cell
        mag_unfolded = mag.unfold(0, h // grid_h, h // grid_h).unfold(
            1, w // grid_w, w // grid_w
        )
        # after knowing what pixels are in each cell, we take the average of those pixels
        avg_magnitude = mag_unfolded.mean(dim=[2, 3])

        # Find the index of the maximum average magnitude
        max_idx_flat = torch.argmax(avg_magnitude)
        max_idx = torch.unravel_index(max_idx_flat, avg_magnitude.shape)

        # Calculate fovea coordinates
        fovea_y: torch.Tensor = max_idx[0] * (h // grid_h) + (h // (2 * grid_h))
        fovea_x: torch.Tensor = max_idx[1] * (w // grid_w) + (w // (2 * grid_w))

        return fovea_x.item(), fovea_y.item()

    # private function to randomly select x% of cones and rods cells
    def __select_random_pixels(
        self, percentage: int, mask: torch.Tensor
    ) -> torch.Tensor:
        # determine the number of pixels to select based on the percentage
        num_pixels = int(percentage / 100 * torch.count_nonzero(mask))

        # get the indices of non-zero pixels in the image mask
        nonzero_indices = torch.nonzero(mask, as_tuple=False)

        # randomly select pixel coordinates
        if len(nonzero_indices) > 0:
            random_indices = torch.randperm(len(nonzero_indices))[:num_pixels]
            selected_indices = nonzero_indices[random_indices]
        else:
            selected_indices = torch.empty(0, 2, dtype=torch.long, device=self.device)

        return selected_indices

    # private function to activate rods and cones at specified coordinates
    def __apply_random_pixel_effect(
        self,
        retina_image: torch.Tensor,
        selected_indices: torch.Tensor,
        effect: str,
        original_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # apply the specified effect to the randomly selected pixels
        # check if length is none
        if selected_indices.shape[0] == 0:
            return retina_image

        y, x = selected_indices[:, 0], selected_indices[:, 1]

        if effect == "grayscale":
            # Calculate mean over the channel dimension
            retina_image[:, y, x] = (
                torch.mean(retina_image[:, y, x], dim=0).unsqueeze(0).expand(3)
            )
        elif effect == "color":
            if original_image is None:
                raise ValueError("Original image must be provided for color effect.")
            retina_image[:, y, x] = original_image[:, y, x]
        else:
            raise ValueError(
                "Unsupported effect type. Supported types are 'grayscale' and 'color'."
            )
        return retina_image

    def cortical_magnification(
        self,
        image: torch.Tensor,
        strength: float = 0.5,
        radius: float = 0.3,
    ) -> torch.Tensor:
        _, height, width = image.shape

        # Normalize coordinates to [-1, 1] space
        x = torch.linspace(-1, 1, width, device=self.device)
        y = torch.linspace(-1, 1, height, device=self.device)
        xv, yv = torch.meshgrid(x, y, indexing="xy")

        # Normalize the focal center to [-1, 1]
        center_x = (self.fovea_center[0] / width) * 2 - 1
        center_y = (self.fovea_center[1] / height) * 2 - 1

        # Shift grid based on the focal point
        xv = xv - center_x
        yv = yv - center_y

        # Calculate distance from the center
        distance = torch.sqrt(xv**2 + yv**2)
        distance = torch.clamp(distance, 1e-6, 1.0)

        # Define outward magnification using a smooth falloff function
        falloff = torch.exp(-((distance / radius) ** 2))
        magnification = 1 + strength * falloff

        # Invert the distortion effect (scale outward)
        xv = xv / magnification
        yv = yv / magnification

        # Create the grid for remap
        grid = torch.stack([xv, yv], dim=-1).unsqueeze(
            0
        )  # Add batch dimension for grid_sample

        # Add center back to grid
        grid += torch.tensor([center_x, center_y], device=self.device)

        # Remap image using the distortion map
        magnified_image_tensor = torch.nn.functional.grid_sample(
            image.unsqueeze(0),  # (Batch, C, H, W)
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(
            0
        )  # remove batch dimension

        return magnified_image_tensor
