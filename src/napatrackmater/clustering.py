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
from scipy.spatial.qhull import QhullError
from lightning import Trainer
from typing import List
from tqdm import tqdm


class PointCloudDataset(Dataset):
    def __init__(self, clouds: List[PyntCloud], center=True, scale_z=1.0, scale_xy=1.0):
        self.clouds = clouds
        self.center = center
        self.scale_z = scale_z
        self.scale_xy = scale_xy

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        point_cloud = self.clouds[idx]
        mean = 0.0  
        point_cloud = torch.tensor(point_cloud.points.values).float()

        if self.center:
            mean = torch.mean(point_cloud, 0).float()  

        scale = torch.tensor([[self.scale_z, self.scale_xy, self.scale_xy]]).float()
        point_cloud = (point_cloud - mean) / scale

        return point_cloud



class Clustering:
    def __init__(
        self,
        pretrainer: Trainer,
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
        compute_with_autoencoder=True,
    ):

        self.pretrainer = pretrainer
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
        self.compute_with_autoencoder = compute_with_autoencoder
        self.timed_cluster_label = {}
        self.timed_latent_features = {}
        self.count = 0

    def _compute_latent_features(self):

        ndim = len(self.label_image.shape)

        if ndim == 2:

            labels, centroids, clouds, marching_cube_points = _label_cluster(
                self.label_image,
                self.num_points,
                self.min_size,
                ndim,
                self.compute_with_autoencoder,
            )
            (
                latent_features,
                cluster_centroids,
                output_largest_eigenvalues,
            ) = _extract_latent_features(
                self.model,
                self.accelerator,
                clouds,
                marching_cube_points,
                centroids,
                self.batch_size,
                self.scale_z,
                self.scale_xy,
            )

            self.timed_latent_features[str(self.key)] = (
                latent_features,
                cluster_centroids,
                output_largest_eigenvalues,
            )

        # ZYX image
        if ndim == 3 and "T" not in self.axes:

            labels, centroids, clouds, marching_cube_points = _label_cluster(
                self.label_image,
                self.num_points,
                self.min_size,
                ndim,
                self.compute_with_autoencoder,
            )
            if len(labels) > 1:

                (
                    latent_features,
                    cluster_centroids,
                    output_largest_eigenvalues,
                ) = _extract_latent_features(
                    self.model,
                    self.accelerator,
                    clouds,
                    marching_cube_points,
                    centroids,
                    self.batch_size,
                    self.scale_z,
                    self.scale_xy,
                )

                self.timed_latent_features[str(self.key)] = (
                    latent_features,
                    cluster_centroids,
                    output_largest_eigenvalues,
                )

        # TYX
        if ndim == 3 and "T" in self.axes:

            for i in range(self.label_image.shape[0]):
                (
                    latent_features,
                    cluster_centroids,
                    output_largest_eigenvalues,
                ) = self._latent_computer(i, ndim - 1)
                self.timed_latent_features[str(i)] = (
                    latent_features,
                    cluster_centroids,
                    output_largest_eigenvalues,
                )

        # TZYX image
        if ndim == 4:

            for i in range(self.label_image.shape[0]):
                (
                    latent_features,
                    cluster_centroids,
                    output_largest_eigenvalues,
                ) = self._latent_computer(i, ndim)
                self.timed_latent_features[str(i)] = (
                    latent_features,
                    cluster_centroids,
                    output_largest_eigenvalues,
                )

    def _latent_computer(self, i, dim):

        xyz_label_image = self.label_image[i, :]
        labels, centroids, clouds, marching_cube_points = _label_cluster(
            xyz_label_image,
            self.num_points,
            self.min_size,
            dim,
            self.compute_with_autoencoder,
        )
        if len(labels) > 1:

            (
                latent_features,
                cluster_centroids,
                output_largest_eigenvalues,
            ) = _extract_latent_features(
                self.model,
                self.accelerator,
                clouds,
                marching_cube_points,
                centroids,
                self.batch_size,
                self.scale_z,
                self.scale_xy,
            )
            return latent_features, cluster_centroids, output_largest_eigenvalues

    def _create_cluster_labels(self):

        ndim = len(self.label_image.shape)
        if ndim == 2:

            labels, centroids, clouds, marching_cube_points = _label_cluster(
                self.label_image,
                self.num_points,
                self.min_size,
                ndim,
                self.compute_with_autoencoder,
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
                self.pretrainer,
                marching_cube_points,
                clouds,
                labels,
                centroids,
                self.compute_with_autoencoder,
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

            labels, centroids, clouds, marching_cube_points = _label_cluster(
                self.label_image,
                self.num_points,
                self.min_size,
                ndim,
                self.compute_with_autoencoder,
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
                    self.pretrainer,
                    marching_cube_points,
                    clouds,
                    labels,
                    centroids,
                    self.compute_with_autoencoder,
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
        labels, centroids, clouds, marching_cube_points = _label_cluster(
            xyz_label_image,
            self.num_points,
            self.min_size,
            dim,
            self.compute_with_autoencoder,
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
                self.pretrainer,
                marching_cube_points,
                clouds,
                labels,
                centroids,
                self.compute_with_autoencoder,
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


def _extract_latent_features(
    model: AutoLightningModel,
    accelerator: str,
    clouds,
    marching_cube_points,
    centroids,
    batch_size: int,
    scale_z: float = 1.0,
    scale_xy: float = 1.0,
):
    dataset = PointCloudDataset(clouds, scale_z=scale_z, scale_xy=scale_xy)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    output_cluster_centroids = []
    output_cluster_centroids = output_cluster_centroids + [
        tuple(centroid_input) for centroid_input in centroids
    ]

    torch_model = model.network
    torch_model.eval()
    device = accelerator
    torch_model.to(device)
    latent_features = []
    output_largest_eigenvalue = [
        get_eccentricity(cloud_input)[2]
        if get_eccentricity(cloud_input) is not None
        else -1
        for cloud_input in marching_cube_points
    ]

    for batch in dataloader:

        with torch.no_grad():
            batch = batch.to(device).float()
            latent_representation_list = torch_model.encoder(batch)
            for latent_representation in latent_representation_list:
                latent_features.append(latent_representation.cpu().numpy())

    return latent_features, output_cluster_centroids, output_largest_eigenvalue


def _model_output(
    model: AutoLightningModel,
    pretrainer: Trainer,
    marching_cube_points,
    clouds,
    labels,
    centroids,
    compute_with_autoencoder,
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

    output_cluster_centroid = output_cluster_centroid + [
        tuple(centroid_input) for centroid_input in centroids
    ]
    output_labels = output_labels + [int(float(label_input)) for label_input in labels]

    if compute_with_autoencoder:

        model.eval()

        outputs_list = pretrainer.predict(model=model, dataloaders=dataloader)

        for outputs in tqdm(outputs_list, desc="Autoencoder model", unit="batch"):
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

    else:

        for cloud_input in tqdm(
            marching_cube_points, desc="Marching cubes", unit="cloud_input"
        ):
            try:
                ConvexHull(cloud_input)

                output_cloud_eccentricity.append(
                    tuple(get_eccentricity(cloud_input))[0]
                )

                output_largest_eigenvector.append(get_eccentricity(cloud_input)[1])
                output_largest_eigenvalue.append(get_eccentricity(cloud_input)[2])
                output_dimensions.append(get_eccentricity(cloud_input)[3])
                output_cloud_surface_area.append(float(get_surface_area(cloud_input)))
            except QhullError:
                output_cloud_eccentricity.append(-1)
                output_largest_eigenvector.append(-1)
                output_largest_eigenvalue.append(-1)
                output_dimensions.append(-1)
                output_cloud_surface_area.append(-1)
    return (
        output_labels,
        output_cluster_centroid,
        output_cloud_eccentricity,
        output_largest_eigenvector,
        output_largest_eigenvalue,
        output_dimensions,
        output_cloud_surface_area,
    )


def _label_cluster(label_image, num_points, min_size, ndim, compute_with_autoencoder):

    labels = []
    centroids = []
    clouds = []
    nthreads = os.cpu_count()
    properties = regionprops(label_image)
    futures = []
    marching_cube_points = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=nthreads) as executor:
        for prop in properties:
            futures.append(executor.submit(get_current_label_binary, prop))
        for r in concurrent.futures.as_completed(futures):
            binary_image, label, centroid = r.result()
            results = get_label_centroid_cloud(
                binary_image,
                num_points,
                ndim,
                label,
                centroid,
                min_size,
                compute_with_autoencoder,
            )

            if results is not None:
                label, centroid, cloud, sample_points = results
                clouds.append(cloud)
                labels.append(label)
                centroids.append(centroid)
                marching_cube_points.append(sample_points)

    return labels, centroids, clouds, marching_cube_points


def get_label_centroid_cloud(
    binary_image, num_points, ndim, label, centroid, min_size, compute_with_autoencoder, padding_size = 3
):

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

        try:
            
            binary_image_padded = np.pad(binary_image, padding_size, mode='constant', constant_values=0)
            vertices, faces, normals, values = marching_cubes(binary_image_padded)
        except Exception as e:
            print(f'Zero padding not possible {e}')    
            try:
                vertices, faces, normals, values = marching_cubes(binary_image)
            except RuntimeError:
                vertices = None

        
        if vertices is not None:
            mesh_obj = trimesh.Trimesh(vertices=vertices, faces=faces)
            simple_clouds = np.asarray(mesh_obj.sample(num_points).data)

            if compute_with_autoencoder:
                mesh_obj = trimesh.Trimesh(
                    vertices=vertices, faces=faces, process=False
                )

                mesh_file = str(label)

                with tempfile.TemporaryDirectory() as mesh_dir:
                    save_mesh_file = os.path.join(mesh_dir, mesh_file) + ".off"
                    mesh_obj.export(save_mesh_file)
                    data = read_off(save_mesh_file)
                pos, face = data["pos"], data["face"]
                if pos.size(1) == 3 and face.size(0) == 3:
                    points = sample_points(data=data, num=num_points).numpy()
                    if ndim == 2:
                        cloud = get_panda_cloud_xy(points)
                    if ndim == 3:
                        cloud = get_panda_cloud_xyz(points)
                    else:
                        cloud = get_panda_cloud_xyz(points)
                else:
                    if ndim == 2:
                        cloud = get_panda_cloud_xy(simple_clouds)
                    elif ndim == 3:
                        cloud = get_panda_cloud_xyz(simple_clouds)
                    else:
                        cloud = get_panda_cloud_xyz(simple_clouds)
            else:
                if ndim == 2:
                    cloud = get_panda_cloud_xy(simple_clouds)
                elif ndim == 3:
                    cloud = get_panda_cloud_xyz(simple_clouds)
                else:
                    cloud = get_panda_cloud_xyz(simple_clouds)

            return label, centroid, cloud, simple_clouds


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
    if np.any(eigenvalues < 0):
        return None
    eccentricities = np.sqrt(eigenvalues)
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


def read_off(path):
    r"""Reads an OFF (Object File Format) file, returning both the position of
    nodes and their connectivity in a :class:`torch_geometric.data.Data`
    object.
    Args:
        path (str): The path to the file.
    """
    with open(path) as f:
        src = f.read().split("\n")[:-1]
    return parse_off(src)


def parse_off(src):
    # Some files may contain a bug and do not have a carriage return after OFF.
    if src[0] == "OFF":
        src = src[1:]
    else:
        src[0] = src[0][3:]

    num_nodes, num_faces = (int(float(item)) for item in src[0].split()[:2])

    pos = parse_txt_array(src[1 : 1 + num_nodes])
    face = src[1 + num_nodes : 1 + num_nodes + num_faces]
    face = face_to_tri(face)
    data = {"pos": pos, "face": face}

    return data


def parse_txt_array(src, sep=None, start=0, end=None, dtype=None):
    src = [[float(x) for x in line.split(sep)[start:end]] for line in src]
    src = torch.tensor(src, dtype=dtype).squeeze()
    return src


def face_to_tri(face):
    face = [[int(float(x)) for x in line.strip().split()] for line in face]

    triangle = torch.tensor([line[1:] for line in face if line[0] == 3])
    triangle = triangle.to(torch.int64)

    rect = torch.tensor([line[1:] for line in face if line[0] == 4])
    rect = rect.to(torch.int64)

    if rect.numel() > 0:
        first, second = rect[:, [0, 1, 2]], rect[:, [0, 2, 3]]
        return torch.cat([triangle, first, second], dim=0).t().contiguous()
    else:
        return triangle.t().contiguous()


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
