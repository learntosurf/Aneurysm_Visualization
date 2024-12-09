import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
from numpy.linalg import norm

class SliceVisualizer:
    """
    Visualizes all slices of a 3D data array, with an option to overlay a corresponding mask.
    
    Args:
        data (numpy.ndarray): A 3D array where each slice along the first axis is a 2D image.
        mask (numpy.ndarray): A 3D array of the same shape as data, representing a mask for each slice.
    """
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def show_slices(self, with_overlay=False):
        """
        Visualizes each slice of the data. Optionally overlays the mask on the data.
        
        Args:
            with_overlay (bool): If True, the mask is overlaid on each slice.
        """
        num_slices = self.data.shape[0]
        cols = 8
        rows = num_slices // cols + int(num_slices % cols != 0)

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()

        for i in range(num_slices):
            axes[i].imshow(self.data[i, :, :], cmap='gray', interpolation='none')
            if with_overlay:
                axes[i].imshow(self.mask[i, :, :], cmap='hot', alpha=0.5, interpolation='none')
            axes[i].set_title(f'Slice {i}')
            axes[i].axis('off')

        for i in range(num_slices, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()    
    
class OverlayVisualizer:
    """
    Creates overlay images from given data and mask slices, with an option to average the overlays.
    
    Args:
        data (numpy.ndarray): The 3D array of data slices.
        mask (numpy.ndarray): The 3D array of mask slices.
    """
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def overlay_slices(self, process_all_slices=True, average_slices=False):
        """
        Creates overlay slices from the data and mask.
        
        Args:
            process_all_slices (bool): If True, processes all slices; otherwise, processes only where mask is non-zero.
            average_slices (bool): If True, averages the overlaid data and mask.
        
        Returns:
            tuple: Overlayed data, overlayed mask, and overlayed data on mask. Averages if specified.
        """
        if process_all_slices:
            overlayed_data = np.sum(self.data, axis=0)
            overlayed_mask = np.sum(self.mask, axis=0)
        else:
            valid_slices = self.mask.any(axis=(1, 2))
            overlayed_data = np.sum(self.data[valid_slices], axis=0)
            overlayed_mask = np.sum(self.mask[valid_slices], axis=0)

        if average_slices:
            num_slices = len(self.data) if process_all_slices else valid_slices.sum()
            overlayed_data /= num_slices
            overlayed_mask /= num_slices

        overlayed_data_on_mask = np.where(overlayed_mask > 0, overlayed_data, 0)

        return overlayed_data, overlayed_mask, overlayed_data_on_mask

    def visualize_overlay(self):
        """
        Visualizes the overlay of data and mask.
        """
        overlayed_data, overlayed_mask, overlayed_data_on_mask = self.overlay_slices()
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
        axes[0].imshow(overlayed_data, cmap='gray')
        axes[0].set_title("Overlayed Data")

        axes[1].imshow(overlayed_mask, cmap='gray')
        axes[1].set_title("Overlayed Mask")

        axes[2].imshow(overlayed_data, cmap='gray', interpolation='none')
        axes[2].imshow(overlayed_mask, cmap='hot', alpha=0.5, interpolation='none')
        axes[2].set_title("Overlayed Data on Mask")

        plt.show()    
        
class OptimizedVesselView:
    """
    Optimizes the view of vessels in the data by rotating the data and mask based on a given direction.
    
    Args:
        data (numpy.ndarray): The 3D data array.
        mask (numpy.ndarray): The 3D mask array.
        direction (numpy.ndarray): Array representing the direction for optimal viewing.
    """
    def __init__(self, data, mask, direction):
        self.data = data
        self.mask = mask
        self.direction = direction

    def rotate_data(self, angle, axes=(0, 1)):
        """
        Rotates the data and mask by a given angle around specified axes.
        
        Args:
            angle (float): The angle of rotation in degrees.
            axes (tuple): The axes around which to rotate the data and mask.
        """
        self.rotated_data = rotate(self.data, angle, axes=axes, reshape=False, order=1)
        self.rotated_mask = rotate(self.mask, angle, axes=axes, reshape=False, order=1)

        rotation_matrix_z = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])

        self.rotated_direction = self.direction.copy()
        for i in range(self.direction.shape[0]):
            direction_subset = self.direction[[1, 4, 7]]
            rotated_direction_subset_z = np.dot(rotation_matrix_z, direction_subset)
            self.rotated_direction[[1, 4, 7]] = rotated_direction_subset_z

        
class AneurysmCentroid:
    """
    Calculates and visualizes the centroid of an aneurysm in rotated data.
    
    Args:
        rotated_data (numpy.ndarray): The rotated data array.
        rotated_mask (numpy.ndarray): The rotated mask array.
    """
    def __init__(self, rotated_data, rotated_mask):
        self.rotated_data = rotated_data
        self.rotated_mask = rotated_mask
        
    def calculate_centroid(self):
        """
        Calculates the centroid of the aneurysm based on the mask.
        
        Returns:
            numpy.ndarray: The coordinates of the centroid.
        """
        coords = np.argwhere(self.rotated_mask == 1)
        centroid = np.mean(coords, axis=0)
        return centroid

    def show_slices_with_centroid(self, centroid):
        """
        Visualizes slices of the data with the aneurysm centroid marked.
        
        Args:
            centroid (numpy.ndarray): The coordinates of the centroid to be marked.
        """
        num_slices = self.rotated_data.shape[0]
        cols = 8
        rows = num_slices // cols + int(num_slices % cols != 0)

        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
        axes = axes.flatten()

        for i in range(num_slices):
            axes[i].imshow(self.rotated_data[i, :, :], cmap='gray')
            if np.any(self.rotated_mask[i, :, :]):
                axes[i].imshow(self.rotated_mask[i, :, :], cmap='hot', alpha=0.5)
                if int(centroid[0]) == i:
                    axes[i].plot(centroid[2], centroid[1], 'go')

            axes[i].set_title(f'Slice {i}')
            axes[i].axis('off')

        for i in range(num_slices, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
        
class SliceExtractor:
    """
    Extracts and combines slices from the rotated data based on the vessel's direction and centroid.
    
    This class is used to extract specific plane slices from a 3D image data array. It calculates the normal to the plane
    based on the vessel's direction and the centroid of the region of interest. It then extracts slices perpendicular
    to this plane and combines them to create a comprehensive view of the target region.
    
    Args:
        rotated_data (numpy.ndarray): The rotated data array, representing 3D image data.
        rotated_mask (numpy.ndarray): The rotated mask array, used for identifying regions of interest.
        direction (numpy.ndarray): An array representing the vessel's direction in the image data.
        centroid_coords (numpy.ndarray): The coordinates of the centroid of the region of interest.
    """
    def __init__(self, rotated_data, rotated_mask, direction, centroid_coords):
        self.data = rotated_data
        self.mask = rotated_mask
        self.direction = direction
        self.centroid = centroid_coords

    def _calculate_plane_normal(self):
        """
        Calculates the normal vector to the plane of interest in the image data.
        It uses the vessel's direction and the centroid to calculate a vector that is perpendicular to the direction of the vessel at the centroid.

        Returns:
            numpy.ndarray: A normalized vector representing the normal to the plane.
        """
        vessel_direction = self.direction[[1, 4, 7]]
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])

        perpendicular_vector = np.cross(vessel_direction, x_axis)
        if norm(perpendicular_vector) == 0:
            perpendicular_vector = np.cross(vessel_direction, y_axis)

        point_on_plane = self.centroid + perpendicular_vector / norm(perpendicular_vector)
        perpendicular_vector_through_centroid = (point_on_plane - self.centroid) / norm(point_on_plane - self.centroid)

        return np.cross(vessel_direction, perpendicular_vector_through_centroid) / norm(np.cross(vessel_direction, perpendicular_vector_through_centroid))

    def extract_plane_slice(self, threshold=0.5):
        """
        Extracts a plane slice from the rotated data based on the calculated plane normal and centroid.
        
        Args:
            threshold (float): Threshold value for extracting the slice.
        
        Returns:
            numpy.ndarray: The extracted plane slice.
        """
        plane_normal = self._calculate_plane_normal()
        z, y, x = self.data.shape
        extracted_slice = np.zeros((y, x))

        for i in range(z):
            for j in range(y):
                for k in range(x):
                    point = np.array([i, j, k])
                    distance = np.abs(np.dot(plane_normal, point - self.centroid))
                    if distance <= threshold:
                        extracted_slice[j, k] = self.data[i, j, k]

        return extracted_slice

    def combine_slices(self, slices):
        """
        Combines multiple extracted slices into a single 2D image.
        
        Args:
            slices (list of numpy.ndarray): List of extracted slices.
        
        Returns:
            numpy.ndarray: The combined image.
        """
        combined_image = np.mean(slices, axis=0)
        return combined_image

    def create_and_combine_slices(self, num_slices, slice_thickness):
        """
        Creates and combines multiple slices from the rotated data.
        
        Args:
            num_slices (int): Number of slices to create and combine.
            slice_thickness (float): Thickness of each slice.
        
        Returns:
            numpy.ndarray: The combined image from multiple slices.
        """
        slices = []
        for i in range(-num_slices // 2, num_slices // 2):
            plane_offset = i * slice_thickness
            slice = self.extract_plane_slice(threshold=0.5)
            slices.append(slice)

        combined_image = self.combine_slices(slices)
        return combined_image

    def visualize_image(self, image, title="Image"):
        """
        Visualizes a single image.
        
        Args:
            image (numpy.ndarray): The image to visualize.
            title (str): Title for the visualization.
        """
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.show()
