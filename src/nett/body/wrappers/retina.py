"""
Dynamic Vision Sensor (DVS) transformation for gym environments.
"""

import collections
import gymnasium as gym
import numpy as np
import cv2
import logging
from stable_baselines3.common.vec_env import VecFrameStack

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Retina(gym.ObservationWrapper):
    """
    A gym observation wrapper that performs Dynamic Vision Sensor (DVS) transformation on the environment observations.

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

    def __init__(self, env):
        super().__init__(env)

        # self.num_stack = 2  ## default
        # self.env = gym.wrappers.FrameStackObservation(env, self.num_stack)
        # self.stack = collections.deque(maxlen=self.num_stack)

        try:
            _, channels, width, height = self.env.observation_space.shape

            # if stack < 2:
            # self.env = gym.wrappers.FrameStackObservation(env, 2)
            # self.env = VecFrameStack(self.env, 2)

            self.retina = ArtificialRetina(
                P=width,
                fovea_center=(height // 2, width // 2),
                fovea_radius=height // 4,
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

        if len(obs) > 1:
            prev = np.transpose(obs[0], (1, 2, 0))
            current = np.transpose(obs[1], (1, 2, 0))
            out = self.retina.process(image=prev, next_image=current)

        else:
            obs = np.transpose(obs, (1, 2, 0))

            out = self.retina.process(image=obs)

        # change to channel first, w, h
        out = np.transpose(out, (2, 0, 1))

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
    peripheral_grayscale - apply grayscale on the peripheral region if True,
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

    def process(self, image, next_image=None):

        # open and pre-process RGB image
        preprocessed_image = image

        # dynamically adjust the fovea location based on optic flow magnitude
        if self.foveation_type == "dynamic":
            # pass t and t+1 frames to get coordinates for dynamic foveation
            fovea_x, fovea_y = self.dynamic_fovea(
                prev_frame=preprocessed_image,
                current_frame=next_image,
                grid_size=self.dynamic_foveation_grid_size,
            )

            # update self.center
            self.fovea_center = (fovea_x, fovea_y)

        # create retina_filter and generate parts of the retina
        self.fovea, self.peripheral_mask = self.create_retina_filter()
        # apply retinal filter on image
        retina_image = self.apply_retina_filter(preprocessed_image)

        # activate cones and rods in peripheral and fovea respectively
        # randomly select x% of pixels in the fovea and make them grayscale
        self.fovea_selected_indices = self.__select_random_pixels(
            percentage=self.fovea_active_rods, mask=self.fovea
        )

        self.__apply_random_pixel_effect(
            retina_image=retina_image,
            selected_indices=self.fovea_selected_indices,
            effect="grayscale",
        )

        # randomly select y% of pixels in the peripheral and remove grayscale effect
        self.peripheral_selected_indices = self.__select_random_pixels(
            percentage=self.peripheral_active_cones, mask=self.peripheral_mask
        )

        self.__apply_random_pixel_effect(
            retina_image=retina_image,
            selected_indices=self.peripheral_selected_indices,
            effect="color",
        )

        if self.cortical_magnifi == True:
            retina_image = self.cortical_magnification(
                image=retina_image,
                center=self.fovea_center,
                strength=self.magnifi_strength,
                radius=self.magnifi_radius,
            )
        return retina_image

    def create_retina_filter(self):
        # create a 2D mask for the circular fovea region
        mask = np.zeros(
            (self.P, self.P), dtype=np.float32
        )  # changed from Uint8 for smooth gradient effect

        # plot the fovea on the 2D mask
        """
        args:
        mask - background on which the circle will be created
        center - coordinates for the circle
        radius - radius of the circle
        (1,1,1) - value inside the circle
        -1 - outline of the circle, -1 means no outline
        """

        fovea = cv2.circle(mask, self.fovea_center, self.fovea_radius, (1, 1, 1), -1)

        # create mask for the peripheral region of the retina
        peripheral_mask = cv2.bitwise_not(fovea)

        return fovea, peripheral_mask

    def apply_retina_filter(self, preprocessed_image):

        # Initialize `img` with the original image
        img = preprocessed_image.copy()

        # define kernel
        ker = self.grad_blur if self.peripheral_gaussianBlur else (1, 1)

        # Initialize the mask with the original fovea
        mask = cv2.GaussianBlur(self.fovea, ker, 0)
        mask = np.dstack([mask] * 3)

        # Apply Gaussian blur to the entire image if enabled
        if self.peripheral_gaussianBlur:
            img = cv2.GaussianBlur(img, self.peripheral_gaussianBlur_kernel, 0)

        # apply visual clutter to the entire image
        if self.visual_clutter == True:
            img = self.radial_pixel_distortion(
                image=img, distortion_intensity=self.clutter_intensity
            )

        # Convert the entire image to grayscale if enabled
        if self.peripheral_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.merge([img] * 3)  # Convert to 3-channel grayscale

        # Combine the foveal and peripheral regions
        combined_image = preprocessed_image * mask + img * (1 - mask)

        return combined_image

    def radial_pixel_distortion(
        self, image, max_distortion=10, distortion_intensity=1.0
    ):
        rows, cols, _ = image.shape
        distorted_image = np.copy(image)

        adjusted_max_distortion = max_distortion * distortion_intensity

        for y in range(rows):
            for x in range(cols):
                # Generate a random radius and angle for radial distortion
                radius = np.random.uniform(0, adjusted_max_distortion)
                angle = np.random.uniform(0, 2 * np.pi)

                # Convert polar to Cartesian
                dx = int(radius * np.cos(angle))
                dy = int(radius * np.sin(angle))

                # Calculate new pixel location
                x_new = np.clip(x + dx, 0, cols - 1)
                y_new = np.clip(y + dy, 0, rows - 1)

                # Set the new pixel value
                distorted_image[y, x] = image[y_new, x_new]

        return distorted_image

    # Function to calculate optical flow and dynamically determine new fovea position
    def dynamic_fovea(self, prev_frame=None, current_frame=None, grid_size=(10, 10)):
        # Convert frames to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        # Calculate optical flow (only accepts single channel images) at timestamps t and t+1
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Calculate magnitude and angle of 2D vectors (flow vector in this case)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Initialize grid for average magnitude calculation
        h, w = mag.shape
        grid_h, grid_w = grid_size
        avg_magnitude = np.zeros((grid_h, grid_w))

        # Calculate average magnitude for each grid cell
        for i in range(grid_h):
            for j in range(grid_w):
                y0, y1 = i * h // grid_h, (i + 1) * h // grid_h
                x0, x1 = j * w // grid_w, (j + 1) * w // grid_w

                # after knowing what pixels are in each cell, we take the average of those pixels
                avg_magnitude[i, j] = np.mean(mag[y0:y1, x0:x1])

        max_idx = np.unravel_index(np.argmax(avg_magnitude), avg_magnitude.shape)
        fovea_y, fovea_x = max_idx[0] * h // grid_h + h // (2 * grid_h), max_idx[
            1
        ] * w // grid_w + w // (2 * grid_w)

        return fovea_x, fovea_y

    # private function to randomly select x% of cones and rods cells
    def __select_random_pixels(self, percentage, mask):
        # determine the number of pixels to select based on the percentage
        num_pixels = int(
            percentage / 100 * np.count_nonzero(mask)
        )  # total pixels = HxW

        # get the indices of non-zero pixels in the image mask
        nonzero_indices = np.transpose(np.nonzero(mask))

        # randomly select pixel coordinates
        random_indices = np.random.choice(
            len(nonzero_indices), num_pixels, replace=False
        )
        selected_indices = nonzero_indices[random_indices]

        return selected_indices

    # private function to activate rods and cones at specified coordinates
    def __apply_random_pixel_effect(self, retina_image, selected_indices, effect):
        # apply the specified effect to the randomly selected pixels
        for y, x in selected_indices:
            if effect == "grayscale":
                retina_image[y, x] = np.mean(retina_image[y, x])
            elif effect == "color":
                retina_image[y, x] = self.preprocessed_image[y, x]
            else:
                raise ValueError(
                    "Unsupported effect type. Supported types are 'grayscale' and 'color'."
                )

    def cortical_magnification(
        self, image, center, strength: float = 0.5, radius: float = 0.3
    ):

        height, width = image.shape[:2]
        min_dim = min(height, width)

        # Normalize coordinates to [-1, 1] space
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        xv, yv = np.meshgrid(x, y)

        # Normalize the focal center to [-1, 1]
        center_x = (center[0] / width) * 2 - 1
        center_y = (center[1] / height) * 2 - 1

        # Shift grid based on the focal point
        xv -= center_x
        yv -= center_y

        # Calculate distance from the center
        distance = np.sqrt(xv**2 + yv**2)
        distance = np.clip(distance, 1e-6, 1.0)

        # Define outward magnification using a smooth falloff function
        falloff = np.exp(-((distance / radius) ** 2))
        magnification = 1 + strength * falloff

        # Invert the distortion effect (scale outward)
        xv = xv / magnification + center_x
        yv = yv / magnification + center_y

        # Map back to pixel coordinates
        map_x = ((xv + 1) * 0.5 * width).astype(np.float32)
        map_y = ((yv + 1) * 0.5 * height).astype(np.float32)

        # Remap image using the distortion map
        magnified_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

        return magnified_image
