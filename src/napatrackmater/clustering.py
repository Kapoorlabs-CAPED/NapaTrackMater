from kapoorlabs_lightning.lightning_trainer import AutoLightningModel
import numpy as np
import concurrent
import os
from skimage.measure import regionprops, marching_cubes
from pyntcloud import PyntCloud
import pandas as pd
import trimesh
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import tempfile
from scipy.spatial import ConvexHull
from lightning import Trainer
from typing import List


class PointCloudDataset(Dataset):
    def __init__(self, clouds: PyntCloud, center=True, scale_z=1.0, scale_xy=1.0):
        self.clouds = clouds
        self.center = center
        self.scale_z = scale_z
        self.scale_xy = scale_xy

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        # read the image
        point_cloud = self.clouds[idx]
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)

        if self.center:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale_z, self.scale_xy, self.scale_xy]])
        point_cloud = (point_cloud - mean) / scale

        return point_cloud


class Clustering:
    def __init__(
        self,
        accelerator: str,
        devices: List[int],
        label_image: np.ndarray,
        axes,
        num_points: int,
        model: AutoLightningModel,
        key=0,
        min_size: tuple = (2, 2, 2),
        progress_bar=None,
        batch_size=1,
        scale_z=1.0,
        scale_xy=1.0,
        center=True,
    ):

        self.accelerator = accelerator
        self.devices = devices
        self.label_image = label_image
        self.model = model
        self.axes = axes
        self.num_points = num_points
        self.min_size = min_size
        self.scale_z = scale_z
        self.scale_xy = scale_xy
        self.center = center
        self.progress_bar = progress_bar
        self.key = key
        self.batch_size = batch_size
        self.timed_cluster_label = {}
        self.count = 0

    def _create_cluster_labels(self):

        ndim = len(self.label_image.shape)

        # YX image
        if ndim == 2:

            labels, centroids, clouds = _label_cluster(
                self.label_image, self.num_points, self.min_size, ndim
            )

            (
                output_labels,
                output_cluster_centroid,
                output_cloud_eccentricity,
                output_largest_eigenvector,
                output_largest_eigenvalue,
                output_dimensions,
                output_cloud_surface_area,
            ) = _model_output(
                self.model,
                self.accelerator,
                self.devices,
                clouds,
                labels,
                centroids,
                self.batch_size,
                self.scale_z,
                self.scale_xy,
            )
            self.timed_cluster_label[str(self.key)] = [
                output_labels,
                output_cluster_centroid,
                output_cloud_eccentricity,
                output_largest_eigenvector,
                output_largest_eigenvalue,
                output_dimensions,
                output_cloud_surface_area,
            ]

        # ZYX image
        if ndim == 3 and "T" not in self.axes:

            labels, centroids, clouds = _label_cluster(
                self.label_image, self.num_points, self.min_size, ndim
            )
            if len(labels) > 1:

                (
                    output_labels,
                    output_cluster_centroid,
                    output_cloud_eccentricity,
                    output_largest_eigenvector,
                    output_largest_eigenvalue,
                    output_dimensions,
                    output_cloud_surface_area,
                ) = _model_output(
                    self.model,
                    self.accelerator,
                    self.devices,
                    clouds,
                    labels,
                    centroids,
                    self.batch_size,
                    self.scale_z,
                    self.scale_xy,
                )
                self.timed_cluster_label[str(self.key)] = [
                    output_labels,
                    output_cluster_centroid,
                    output_cloud_eccentricity,
                    output_largest_eigenvector,
                    output_largest_eigenvalue,
                    output_dimensions,
                    output_cloud_surface_area,
                ]

        # TYX
        if ndim == 3 and "T" in self.axes:

            for i in range(self.label_image.shape[0]):
                self.count = self.count + 1
                (
                    output_labels,
                    output_cluster_centroid,
                    output_cloud_eccentricity,
                    output_largest_eigenvector,
                    output_largest_eigenvalue,
                    output_dimensions,
                    output_cloud_surface_area,
                ) = self._label_computer(i, ndim - 1)
                self.timed_cluster_label[str(i)] = [
                    output_labels,
                    output_cluster_centroid,
                    output_cloud_eccentricity,
                    output_largest_eigenvector,
                    output_largest_eigenvalue,
                    output_dimensions,
                    output_cloud_surface_area,
                ]

        # TZYX image
        if ndim == 4:

            for i in range(self.label_image.shape[0]):
                self.count = self.count + 1
                (
                    output_labels,
                    output_cluster_centroid,
                    output_cloud_eccentricity,
                    output_largest_eigenvector,
                    output_largest_eigenvalue,
                    output_dimensions,
                    output_cloud_surface_area,
                ) = self._label_computer(i, ndim)
                self.timed_cluster_label[str(i)] = [
                    output_labels,
                    output_cluster_centroid,
                    output_cloud_eccentricity,
                    output_largest_eigenvector,
                    output_largest_eigenvalue,
                    output_dimensions,
                    output_cloud_surface_area,
                ]

    def _label_computer(self, i, dim):

        xyz_label_image = self.label_image[i, :]
        labels, centroids, clouds = _label_cluster(
            xyz_label_image, self.num_points, self.min_size, dim
        )
        if len(labels) > 1:

            (
                output_labels,
                output_cluster_centroid,
                output_cloud_eccentricity,
                output_largest_eigenvector,
                output_largest_eigenvalue,
                output_dimensions,
                output_cloud_surface_area,
            ) = _model_output(
                self.model,
                self.accelerator,
                self.devices,
                clouds,
                labels,
                centroids,
                self.batch_size,
                self.scale_z,
                self.scale_xy,
            )

            return (
                output_labels,
                output_cluster_centroid,
                output_cloud_eccentricity,
                output_largest_eigenvector,
                output_largest_eigenvalue,
                output_dimensions,
                output_cloud_surface_area,
            )


def _model_output(
    model: torch.nn.Module,
    accelerator: str,
    devices: List[int],
    clouds,
    labels,
    centroids,
    batch_size: int,
    scale_z: float = 1.0,
    scale_xy: float = 1.0,
):

    output_labels = []
    output_cluster_centroid = []
    output_cloud_eccentricity = []
    output_cloud_surface_area = []
    output_largest_eigenvector = []
    output_largest_eigenvalue = []
    output_dimensions = []
    dataset = PointCloudDataset(clouds, scale_z=scale_z, scale_xy=scale_xy)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    print(f"Predicting {len(dataset)} clouds..., {len(centroids)} centroids...")
    pretrainer = Trainer(accelerator=accelerator, devices=devices)
    outputs_list = pretrainer.predict(model=model, dataloaders=dataloader)
    output_cluster_centroid = output_cluster_centroid + [
        tuple(centroid_input) for centroid_input in centroids
    ]
    output_labels = output_labels + [int(float(label_input)) for label_input in labels]
    for outputs in outputs_list:
        output_cloud_eccentricity = output_cloud_eccentricity + [
            tuple(get_eccentricity(cloud_input.detach().cpu().numpy()))[0]
            for cloud_input in outputs
        ]
        output_largest_eigenvector = output_largest_eigenvector + [
            get_eccentricity(cloud_input.detach().cpu().numpy())[1]
            for cloud_input in outputs
        ]
        output_largest_eigenvalue = output_largest_eigenvalue + [
            get_eccentricity(cloud_input.detach().cpu().numpy())[2]
            for cloud_input in outputs
        ]
        output_dimensions = output_dimensions + [
            get_eccentricity(cloud_input.detach().cpu().numpy())[3]
            for cloud_input in outputs
        ]
        output_cloud_surface_area = output_cloud_surface_area + [
            float(get_surface_area(cloud_input.detach().cpu().numpy()))
            for cloud_input in outputs
        ]

    return (
        output_labels,
        output_cluster_centroid,
        output_cloud_eccentricity,
        output_largest_eigenvector,
        output_largest_eigenvalue,
        output_dimensions,
        output_cloud_surface_area,
    )


def _label_cluster(label_image, num_points, min_size, ndim):

    labels = []
    centroids = []
    clouds = []
    nthreads = os.cpu_count()
    properties = regionprops(label_image)
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        for prop in properties:
            futures.append(executor.submit(get_current_label_binary, prop))
        for r in concurrent.futures.as_completed(futures):
            binary_image, label, centroid = r.result()
            results = get_label_centroid_cloud(
                binary_image, num_points, ndim, label, centroid, min_size
            )

            if results is not None:
                label, centroid, cloud = results
                clouds.append(cloud)
                labels.append(label)
                centroids.append(centroid)

    return labels, centroids, clouds


def get_label_centroid_cloud(binary_image, num_points, ndim, label, centroid, min_size):

    valid = []

    if min_size is not None:
        for j in range(len(min_size)):
            if binary_image.shape[j] >= min_size[j]:
                valid.append(True)
            else:
                valid.append(False)
    else:
        for j in range(len(binary_image.shape)):
            valid.append(True)

    if False not in valid:
        # Apply the model prediction for getting clusters
        try:
            vertices, faces, normals, values = marching_cubes(binary_image)
        except RuntimeError:
            print("Marching cubes failed for label: ", label)
            vertices = None
        if vertices is not None:
            mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            data = {"pos": mesh_obj.vertices, "face": mesh_obj.faces}
            if data["pos"].shape[1] == 3 and data["face"].shape[0] == 3:
                points = sample_points(data=data, num=num_points).numpy()
                if ndim == 2:
                    cloud = get_panda_cloud_xy(points)
                if ndim == 3:
                    cloud = get_panda_cloud_xyz(points)
                else:
                    cloud = get_panda_cloud_xyz(points)

                return label, centroid, cloud


def get_panda_cloud_xy(points):

    cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y"]))

    return cloud


def get_panda_cloud_xyz(points):

    cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y", "z"]))

    return cloud


def get_eccentricity(point_cloud):

    # Compute the covariance matrix of the point cloud
    cov_mat = np.cov(point_cloud, rowvar=False)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

    # Sort the eigenvectors in descending order of their corresponding eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]
    largest_eigen_vector = eigenvectors[:, idx[0]]
    largest_eigen_value = eigenvalues[idx[0]]
    # Compute the eccentricity along each principal axis
    eccentricities = np.sqrt(eigenvalues / eigenvalues.min())
    dimensions = idx

    return eccentricities, largest_eigen_vector, largest_eigen_value, dimensions


def get_surface_area(point_cloud):
    # Compute the convex hull of the point cloud
    hull = ConvexHull(point_cloud)

    # Compute the areas of the triangles in the convex hull
    areas = np.zeros(hull.simplices.shape[0])
    for i, simplex in enumerate(hull.simplices):
        a = point_cloud[simplex[0]]
        b = point_cloud[simplex[1]]
        c = point_cloud[simplex[2]]
        ab = b - a
        ac = c - a
        areas[i] = 0.5 * np.linalg.norm(np.cross(ab, ac))

    # Compute the total surface area
    surface_area = areas.sum()

    return surface_area


def get_current_label_binary(prop: regionprops):

    binary_image = prop.image
    label = prop.label
    centroid = np.asarray(prop.centroid)

    return binary_image, label, centroid

def sample_points(data, num):
    pos, face = data["pos"], data["face"]
    assert pos.size(1) == 3 and face.size(0) == 3

    pos_max = pos.abs().max()
    pos = pos / pos_max

    area = (pos[face[1]] - pos[face[0]]).cross(pos[face[2]] - pos[face[0]])
    area = area.norm(p=2, dim=1).abs() / 2

    prob = area / area.sum()
    sample = torch.multinomial(prob, num, replacement=True)
    face = face[:, sample]

    frac = torch.rand(num, 2)
    mask = frac.sum(dim=-1) > 1
    frac[mask] = 1 - frac[mask]

    vec1 = pos[face[1]] - pos[face[0]]
    vec2 = pos[face[2]] - pos[face[0]]

    pos_sampled = pos[face[0]]
    pos_sampled += frac[:, :1] * vec1
    pos_sampled += frac[:, 1:] * vec2

    pos_sampled = pos_sampled * pos_max

    return pos_sampled