import pandas as pd
import numpy as np


class Ergodicity:

    def __init__(
         self,    
         cell_type_dataframe: pd.DataFrame,
         features = [
            "Radius",
            "Eccentricity_Comp_First",
            "Eccentricity_Comp_Second",
            "Eccentricity_Comp_Third",
            "Surface_Area",
        ], 
         time_delta: int = 25,
         cell_type_str = 'Cell_Type',
         class_map_gbr = {
                0: "Basal",
                1: "Radial",
                2: "Goblet"
            }
    ):
        
        self.cell_type_dataframe = cell_type_dataframe 
        self.features = features
        self.time_delta = time_delta 
        self.cell_type_str = cell_type_str
        self.class_map_gbr = class_map_gbr
        self.unique_time_points = self.cell_type_dataframe['t'].unique()
        
    def _get_temporal_average(self):
        
        self.temporal_average_dict = {cell_type: {} for cell_type in self.class_map_gbr.values()}
        unique_t = np.sort(self.unique_time_points)
        max_t = unique_t.max() - self.time_delta 


        for time_point in unique_t:
            if time_point > max_t:
                break
            
            start_time = time_point 
            end_time = start_time + self.time_delta
            interval_data = self.cell_type_dataframe[(self.cell_type_dataframe['t'] >= start_time) & (self.cell_type_dataframe['t'] < end_time)]
            for cell_type in self.class_map_gbr.values():
               
               cell_type_data = interval_data[interval_data[self.cell_type_str] == cell_type]
               
               if not cell_type_data.empty:
                   if end_time not in self.temporal_average_dict[cell_type]:
                       self.temporal_average_dict[cell_type][end_time] = {}

                   track_averages = []  
                   for trackmate_id in cell_type_data['TrackMate Track ID'].unique():
                        current_trackmate_id_data = cell_type_data[cell_type_data['TrackMate Track ID'] == trackmate_id]
                        for track_id in current_trackmate_id_data['Track ID'].unique():
                            track_features = current_trackmate_id_data[current_trackmate_id_data['Track ID'] == track_id][self.features].to_numpy()
                            track_averages.append(track_features.mean(axis=0))    
                   if track_averages:
                       self.temporal_average_dict[cell_type][end_time] = np.vstack(track_averages)

    def _get_statial_average(self):

        self.spatial_average_dict = {cell_type: {} for cell_type in self.class_map_gbr.values()}

        for time_point in self.unique_time_points:
           
           time_data = self.cell_type_dataframe[self.cell_type_dataframe['t'] == time_point]
           
           for cell_type in self.class_map_gbr.values():
               
               cell_type_data = time_data[time_data[self.cell_type_str] == cell_type]

               if not cell_type_data.empty:
                   if time_point not in self.spatial_average_dict[cell_type]:
                       self.spatial_average_dict[cell_type][time_point] = {}
                   
                   mean_vals = np.mean(
                        cell_type_data[self.features].values,
                        axis=0
                    )
                   ensemble_average = {
                        feat: mean_vals[i]
                        for i, feat in enumerate(self.features)
                    }
                   self.spatial_average_dict[cell_type][time_point] = ensemble_average

    def ergodicity_test(self):
        """
        For each cell_type and each valid interval end_time,
        compute per-feature squared error between:
        - spatial average at end_time
        - temporal average (across tracks) at end_time
        Returns:
            dict[cell_type] → pd.DataFrame with columns:
                ['end_time'] + self.features
        """
        # rebuild your averages
        self._get_statial_average()
        self._get_temporal_average()

        results = {}
        for cell_type in self.class_map_gbr.values():
            rows = []
            for start in np.sort(self.unique_time_points):
                end_time = start + self.time_delta

                if (end_time not in self.spatial_average_dict[cell_type] or
                    end_time not in self.temporal_average_dict[cell_type]):
                    continue

                
                spatial_vec = np.array([
                    self.spatial_average_dict[cell_type][end_time][feat]
                    for feat in self.features
                ])
                temp_array   = self.temporal_average_dict[cell_type][end_time]
                temporal_vec = temp_array.mean(axis=0)

               
                se = spatial_vec - temporal_vec

               
                row = {'end_time': end_time}
                for idx, feat in enumerate(self.features):
                    row[feat] = se[idx]
                rows.append(row)

            results[cell_type] = pd.DataFrame(rows)

        return results              