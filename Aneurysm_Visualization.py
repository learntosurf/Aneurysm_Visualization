import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import affine_transform
from scipy.ndimage import rotate
from numpy.linalg import norm


class SliceVisualizer:
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def show_slices(self, with_overlay=False):
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
    def __init__(self, data, mask):
        self.data = data
        self.mask = mask

    def overlay_slices(self, process_all_slices=True, average_slices=False):
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
    def __init__(self, data, mask, direction):
        self.data = data
        self.mask = mask
        self.direction = direction

    def rotate_data(self, angle):
        r = R.from_euler('z', angle, degrees=True)
        self.rotated_data = r.apply(self.data, axes=(0, 1))
        self.rotated_mask = r.apply(self.mask, axes=(0, 1))
        
        rotated_direction_subset_z = r.apply(self.direction[[1, 4, 7]])
        self.rotated_direction = self.direction.copy()
        self.rotated_direction[[1, 4, 7]] = rotated_direction_subset_z

        
class AneurysmCentroid:
    def __init__(self, rotated_data, rotated_mask):
        self.rotated_data = rotated_data
        self.rotated_mask = rotated_mask
        
    def calculate_centroid(self):
        coords = np.argwhere(self.rotated_mask == 1)
        centroid = np.mean(coords, axis=0)
        return centroid

    def show_slices_with_centroid(self, centroid):
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
        
class Rotation:
    def __init__(self, rotated_data, rotated_mask, direction, centroid_coords):
        self.data = rotated_data
        self.mask = rotated_mask
        self.direction = direction
        self.centroid = centroid_coords

    def calculate_rotation(self):
        plane_normal = self._calculate_plane_normal()
        rotation_angle = self._calculate_rotation_angle(plane_normal)
        self._apply_rotation(rotation_angle, plane_normal)

    def _calculate_plane_normal(self):
        vessel_direction = self.direction[[1, 4, 7]]
        x_axis = np.array([1, 0, 0])
        y_axis = np.array([0, 1, 0])

        perpendicular_vector = np.cross(vessel_direction, x_axis)
        if norm(perpendicular_vector) == 0:
            perpendicular_vector = np.cross(vessel_direction, y_axis)

        point_on_plane = self.centroid + perpendicular_vector / norm(perpendicular_vector)
        perpendicular_vector_through_centroid = (point_on_plane - self.centroid) / norm(point_on_plane - self.centroid)

        return np.cross(vessel_direction, perpendicular_vector_through_centroid) / norm(np.cross(vessel_direction, perpendicular_vector_through_centroid))

    def _calculate_rotation_angle(self, plane_normal):
        rotation_axis = np.array([0, 0, 1])
        cos_angle = np.dot(plane_normal, rotation_axis) / (norm(plane_normal) * norm(rotation_axis))
        return np.arccos(cos_angle)

    def _apply_rotation(self, rotation_angle, plane_normal):
        rotation_vector = np.cross(plane_normal, np.array([0, 0, 1]))
        rotation = R.from_rotvec(rotation_vector * rotation_angle)

        self.rotated_data = rotate(self.data, angle=np.degrees(rotation_angle), axes=(1, 2), reshape=False)
        self.rotated_mask = rotate(self.mask, angle=np.degrees(rotation_angle), axes=(1, 2), reshape=False)