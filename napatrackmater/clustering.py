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
from tqdm import tqdm
import tempfile 
from scipy.spatial import ConvexHull


class PointCloudDataset(Dataset):
    def __init__(self, clouds, labels, centroids, centre=True, scale=20.0):
        self.clouds = clouds
        self.labels = labels 
        self.centroids = centroids 
        self.centre = centre
        self.scale = scale
      
      

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        # read the image
        point_cloud = self.clouds[idx]
        point_label = self.labels[idx]
        point_centroid = self.centroids[idx]
        mean = 0
        point_cloud = torch.tensor(point_cloud.points.values)
       
        if self.centre:
            mean = torch.mean(point_cloud, 0)

        scale = torch.tensor([[self.scale, self.scale, self.scale]])
        point_cloud = (point_cloud - mean) / scale
        return point_cloud, point_label, point_centroid 

class Clustering:

    def __init__(self, label_image: np.ndarray, axes,  num_points: int, model: DeepEmbeddedClustering, key = 0,  min_size:tuple = (2,2,2), progress_bar = None, batch_size = 1):

        self.label_image = label_image 
        self.model = model
        self.axes = axes
        self.num_points = num_points
        self.min_size = min_size
       
        self.progress_bar = progress_bar
        self.key = key
        self.batch_size = batch_size
        self.timed_cluster_label = {}
        self.count = 0

    def _create_cluster_labels(self):

        ndim = len(self.label_image.shape)

        #YX image  
        if ndim == 2:
           
           labels, centroids, clouds = _label_cluster(self.label_image, self.model, self.num_points, self.min_size, ndim)
           
           output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area = _model_output(self.model, clouds, labels, centroids, self.batch_size)
           self.timed_cluster_label[str(self.key)] = [output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area]     
 
        #ZYX image
        if ndim == 3 and 'T' not in self.axes:
               
           labels, centroids, clouds = _label_cluster(self.label_image,   self.num_points, self.min_size, ndim)
           if len(labels) > 1:
                
                output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector,output_largest_eigenvalue, output_cloud_surface_area = _model_output(self.model, clouds, labels, centroids, self.batch_size)
                self.timed_cluster_label[str(self.key)] = [output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area]


        #TYX
        if ndim == 3 and 'T' in self.axes:
             

               for i in range(self.label_image.shape[0]):
                        self.count = self.count + 1
                        output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area  = self._label_computer(i, ndim - 1)
                        self.timed_cluster_label[str(i)] = [output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area]
                
        #TZYX image        
        if ndim == 4:
               

                for i in range(self.label_image.shape[0]):
                        self.count = self.count + 1
                        output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area  = self._label_computer(i, ndim)
                        self.timed_cluster_label[str(i)] = [output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area]
                
                        
                            

    def _label_computer(self, i, dim):
        
            xyz_label_image = self.label_image[i,:]
            labels, centroids, clouds = _label_cluster(xyz_label_image,   self.num_points, self.min_size, dim)
            if len(labels) > 1:
                
                output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area = _model_output(self.model, clouds, labels, centroids, self.batch_size)
            
                return  output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area

def _model_output(model, clouds, labels, centroids, batch_size):
       
        output_labels = []
        output_cluster_score = []
        output_cluster_class = []
        output_cluster_centroid = []
        output_cloud_eccentricity = [] 
        output_cloud_surface_area = []
        output_largest_eigenvector = []

        dataset = PointCloudDataset(clouds, labels, centroids)
        dataloader = DataLoader(dataset, batch_size = batch_size)
        model.eval()
       
        for data in dataloader:
                cloud_inputs, label_inputs, centroid_inputs = data
                try:
                        output, features, clusters = model(cloud_inputs.cuda())
                except ValueError:
                        output, features, clusters = model(cloud_inputs.cpu())      
                                
                output_cluster_score = output_cluster_score + [max(torch.squeeze(cluster).detach().cpu().numpy()) for cluster in clusters]
                output_cluster_centroid = output_cluster_centroid +  [tuple(torch.squeeze(centroid_input).detach().cpu().numpy()) for centroid_input in centroid_inputs]
                output_labels = output_labels + [int(float(torch.squeeze(label_input).detach().cpu().numpy())) for label_input in label_inputs]
                output_cluster_class = output_cluster_class + [np.argmax(torch.squeeze(cluster).detach().cpu().numpy()) for cluster in clusters]
                output_cloud_eccentricity = output_cloud_eccentricity +  [tuple(get_eccentricity(cloud_input)[0]) for cloud_input in cloud_inputs]
                output_largest_eigenvector = output_largest_eigenvector + [get_eccentricity(cloud_input)[1] for cloud_input in cloud_inputs]
                output_largest_eigenvalue = output_largest_eigenvalue + [get_eccentricity(cloud_input)[2] for cloud_input in cloud_inputs]

                output_cloud_surface_area = output_cloud_surface_area + [float(get_surface_area(cloud_input)) for cloud_input in cloud_inputs]
        return output_labels, output_cluster_score, output_cluster_class, output_cluster_centroid, output_cloud_eccentricity, output_largest_eigenvector, output_largest_eigenvalue, output_cloud_surface_area             


       

def _label_cluster(label_image, num_points, min_size, ndim):
       
       labels = []
       centroids = []
       clouds = []
       nthreads = os.cpu_count()
       properties = regionprops(label_image)
       futures = []
       with concurrent.futures.ThreadPoolExecutor(max_workers = nthreads) as executor:
             for prop in properties:
                          futures.append(executor.submit(get_current_label_binary, prop))
             for r in concurrent.futures.as_completed(futures):
                        binary_image, label, centroid = r.result()             
                        label, centroid, cloud = get_label_centroid_cloud(binary_image,  num_points, ndim, label, centroid,  min_size)
                        clouds.append(cloud)  
                        labels.append(label)   
                        centroids.append(centroid)

       return labels, centroids, clouds

def get_label_centroid_cloud(binary_image,  num_points, ndim, label, centroid, min_size):
                            
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
                                    
                                    with tempfile.TemporaryDirectory() as mesh_dir:
                                                save_mesh_file = os.path.join(mesh_dir, mesh_file) + ".off"
                                                mesh_obj.export(save_mesh_file) 
                                                data = read_off(save_mesh_file)
                                    
                                    points = sample_points(data=data, num=num_points).numpy()
                                    if ndim == 2:
                                      cloud = get_panda_cloud_xy(points)
                                    if ndim == 3:
                                      cloud = get_panda_cloud_xyz(points)  
                                    else:
                                      cloud = get_panda_cloud_xyz(points)    

                                     
                                      

                                     
                            return  label, centroid, cloud       
       

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
    
        return eccentricities, largest_eigen_vector, largest_eigen_value

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

def get_current_label_binary(prop):
                      
                binary_image = prop.image
                label = prop.label 
                centroid = np.asarray(prop.centroid) 


                return binary_image , label, centroid