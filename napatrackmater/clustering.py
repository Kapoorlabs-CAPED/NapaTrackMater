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
class Clustering:

    def __init__(self, label_image: np.ndarray, axes, mesh_dir: str, num_points: int, model: DeepEmbeddedClustering):

        self.label_image = label_image 
        self.model = model
        self.axes = axes
        self.num_points = num_points
        self.mesh_dir = mesh_dir 
        self.timed_cluster_label = {}
        Path(self.mesh_dir).mkdir(exist_ok=True)

    def _create_cluster_labels(self):

        ndim = len(self.label_image.shape)

        #YX image  
        if ndim == 2:
           
           labels, cluster_labels = label_cluster(self.label_image, self.model, self.mesh_dir, self.num_points, ndim)
           self.timed_cluster_label[str(0)] = [labels, cluster_labels]     
 
        #ZYX image
        if ndim == 3 and 'T' not in self.axes:
               
           labels, cluster_labels = label_cluster(self.label_image, self.model, self.mesh_dir, self.num_points, ndim)
           self.timed_cluster_label[str(0)] = [labels, cluster_labels]

        #TYX
        if ndim == 3 and 'T' in self.axes:
               for i in range(self.label_image.shape[0]):
                      xy_label_image = self.label_image[i,:]
                      labels, cluster_labels = label_cluster(xy_label_image, self.model, self.mesh_dir, self.num_points, ndim - 1)
                      self.timed_cluster_label[str(i)] = [labels, cluster_labels]
        #TZYX image        
        if ndim == 4:
               for i in range(self.label_image.shape[0]):
                      xyz_label_image = self.label_image[i,:]
                      labels, cluster_labels = label_cluster(xyz_label_image, self.model, self.mesh_dir, self.num_points, ndim)
                      self.timed_cluster_label[str(i)] = [labels, cluster_labels]


def label_cluster(label_image, model, mesh_dir, num_points, ndim):
       
       labels = []
       cluster_labels = []
       nthreads = os.cpu_count() - 1
       properties = regionprops(label_image)
       with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
                    futures = []
                    for prop in properties:
                            futures.append(executor.submit(get_current_label_binary, prop))
                    for future in concurrent.futures.as_completed(futures):
                            binary_image, label = future.result()
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
                            output, features, clusters = model(cloud)
                            labels.append(label)
                            cluster_labels.append(clusters)

       return labels, cluster_labels

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