from cellshape_cluster import DeepEmbeddedClustering
from cellshape_cloud import CloudAutoEncoder
from cellshape_helper.vendor.pytorch_geometric_files import read_off, sample_points
import numpy as np
import concurrent 
import os
from pathlib import Path
from skimage.measure import regionprops, marching_cubes
from pyntcloud import PyntCloud
import pandas as pd
import trimesh
import torch
from torch.utils.data import Dataset
from pyntcloud import PyntCloud
from torch.utils.data import DataLoader


class PointCloudDataset(Dataset):
    def __init__(self, clouds, labels, centre=True, scale=20.0):
        self.clouds = clouds
        self.labels = labels 
        self.centre = centre
        self.scale = scale
      
      

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        # read the image
        point_cloud = self.clouds[idx]
        point_label = self.labels[idx]
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
        point_label = torch.mean(point_label,0)
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale
        return point_cloud, point_label

class Clustering:

    def __init__(self, label_image: np.ndarray, axes, mesh_dir: str, num_points: int, model: DeepEmbeddedClustering, min_size:tuple = (2,2,2)):

        self.label_image = label_image 
        self.model = model
        self.axes = axes
        self.num_points = num_points
        self.mesh_dir = mesh_dir 
        self.min_size = min_size
        self.timed_cluster_label = {}
        Path(self.mesh_dir).mkdir(exist_ok=True)

    def _create_cluster_labels(self):

        ndim = len(self.label_image.shape)

        #YX image  
        if ndim == 2:
           
           labels, clouds = label_cluster(self.label_image, self.model, self.mesh_dir, self.num_points, self.min_size, ndim)
           dataset = PointCloudDataset(clouds, labels)
           dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
           input_labels = []
           cluster_labels = []
           for data in dataloader:
                 inputs = data[0]
                 label_inputs = data[1]
                 output, features, clusters = self.model(inputs)
                 
                 input_labels.append(label_inputs)
                 cluster_labels.append(clusters)

           self.timed_cluster_label[str(0)] = [input_labels, cluster_labels]     
 
        #ZYX image
        if ndim == 3 and 'T' not in self.axes:
               
           labels, clouds = label_cluster(self.label_image,  self.mesh_dir, self.num_points, self.min_size, ndim)
           if len(labels) > 1:
                dataset = PointCloudDataset(clouds, labels)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                input_labels = []
                cluster_labels = []
                for data in dataloader:
                        inputs = data[0]
                        label_inputs = data[1]
                        output, features, clusters = self.model(inputs)
                        
                        input_labels.append(label_inputs)
                        cluster_labels.append(clusters)


        #TYX
        if ndim == 3 and 'T' in self.axes:
               for i in range(self.label_image.shape[0]):
                      xy_label_image = self.label_image[i,:]
                      labels, clouds = label_cluster(xy_label_image, self.mesh_dir, self.num_points, self.min_size, ndim - 1)
                      if len(labels) > 1:
                            dataset = PointCloudDataset(clouds, labels)
                            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                            input_labels = []
                            cluster_labels = []
                            for data in dataloader:
                                    inputs = data[0]
                                    label_inputs = data[1]
                                    output, features, clusters = self.model(inputs)
                                    
                                    input_labels.append(label_inputs)
                                    cluster_labels.append(clusters)
                            self.timed_cluster_label[str(i)] = [input_labels, cluster_labels]
        #TZYX image        
        if ndim == 4:
               for i in range(self.label_image.shape[0]):
                      xyz_label_image = self.label_image[i,:]
                      labels, clouds = label_cluster(xyz_label_image,  self.mesh_dir, self.num_points, self.min_size, ndim)
                      if len(labels) > 1:
                            dataset = PointCloudDataset(clouds, labels)
                            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                            input_labels = []
                            cluster_labels = []
                            for data in dataloader:
                                    inputs = data[0]
                                    label_inputs = data[1]
                                    output, features, clusters = self.model(inputs)
                                    
                                    input_labels.append(label_inputs)
                                    cluster_labels.append(clusters)
                            self.timed_cluster_label[str(i)] = [input_labels, cluster_labels]


def label_cluster(label_image,  mesh_dir, num_points, min_size, ndim):
       
       labels = []
       clouds = []
       nthreads = os.cpu_count() - 1
       properties = regionprops(label_image)
       with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
                    futures = []
                    for prop in properties:
                            futures.append(executor.submit(get_current_label_binary, prop))
                    for future in concurrent.futures.as_completed(futures):
                            binary_image, label = future.result()
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
                                    #Apply the model prediction for getting clusters
                                    vertices, faces, normals, values = marching_cubes(binary_image)
                                    mesh_obj = trimesh.Trimesh(
                                        vertices=vertices, faces=faces, process=False
                                    )
                                    mesh_file = str(label) 
                                    
                                    save_mesh_file = os.path.join(mesh_dir, mesh_file) + ".off"
                                    mesh_obj.export(save_mesh_file) 
                                    data = read_off(os.path.join(mesh_dir, save_mesh_file))
                                    points = sample_points(data=data, num=num_points).numpy()
                                    if ndim == 2:
                                      cloud = get_panda_cloud_xy(points)
                                    if ndim == 3:
                                      cloud = get_panda_cloud_xyz(points)  

                                    clouds.append(cloud)  
                                    labels.append(labels)       

       return labels, clouds

def get_panda_cloud_xy(points):
        
        cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y"]))

        return cloud

def get_panda_cloud_xyz(points):
        
        cloud = PyntCloud(pd.DataFrame(data=points, columns=["x", "y", "z"]))

        return cloud

        


def get_current_label_binary(prop):
                      
                binary_image = prop.image
                label = prop.label  

                return binary_image , label