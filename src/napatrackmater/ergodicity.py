import pandas as pd
import numpy as np


class Ergodicity:

    def __init__(
         self,    
         cell_type_dataframe: pd.DataFrame,
         features: list = [
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
        
    def _get_spatial_temporal_average(self):
        
        self.temporal_average_dict = {cell_type: {} for cell_type in self.class_map_gbr.values()}
        self.spatial_average_dict = {cell_type: {} for cell_type in self.class_map_gbr.values()}

        unique_t = np.sort(self.unique_time_points)
        max_t = unique_t.max() - self.time_delta 


        for time_point in unique_t:
            if time_point > max_t:
                break
            
            start_time = time_point 
            end_time = start_time + self.time_delta
            interval_data = self.cell_type_dataframe[(self.cell_type_dataframe['t'] >= start_time) & (self.cell_type_dataframe['t'] <= end_time)]
            time_data = self.cell_type_dataframe[self.cell_type_dataframe['t'] == end_time]
            for cell_type in self.class_map_gbr.values():
               
               cell_type_data = interval_data[interval_data[self.cell_type_str] == cell_type]
               end_time_cell_type_data = time_data[time_data[self.cell_type_str] == cell_type]

               if not end_time_cell_type_data.empty:
                    if time_point not in self.spatial_average_dict[cell_type]:
                            self.spatial_average_dict[cell_type][time_point] = {}
                    ensemble_averages = []  
                    for trackmate_id in end_time_cell_type_data['TrackMate Track ID'].unique():
                        current_trackmate_id_data = end_time_cell_type_data[end_time_cell_type_data['TrackMate Track ID'] == trackmate_id]
                        for track_id in current_trackmate_id_data['Track ID'].unique():
                            track_features = current_trackmate_id_data[current_trackmate_id_data['Track ID'] == track_id][self.features].to_numpy()
                            ensemble_averages.append(track_features.mean(axis=0))    
                    if ensemble_averages:
                       self.spatial_average_dict[cell_type][end_time] = np.mean(ensemble_averages, axis=0)

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
                       self.temporal_average_dict[cell_type][end_time] = np.mean(np.vstack(track_averages), axis=0)



    def ergodicity_test(self):
        """
        For each cell_type and each valid interval end_time,
        compute for each feature i:
        - mean_k[Δ_i,k(t)]   and
        - std_k[Δ_i,k(t)]
        where Δ_i,k = ensemble_avg_i - track_k_time_avg_i.
        Returns:
            dict[cell_type] → pd.DataFrame with columns:
            ['end_time'] + [f"{feat}_mean", f"{feat}_std" for feat in self.features]
        """
        # rebuild your averages
        self._get_spatial_temporal_average()

        results = {}
        for cell_type in self.class_map_gbr.values():
            rows = []
            for start in np.sort(self.unique_time_points):
                end_time = start + self.time_delta

                if (end_time not in self.spatial_average_dict[cell_type]
                    or end_time not in self.temporal_average_dict[cell_type]):
                    continue

                # ensemble average vector
                spatial_vec =self.spatial_average_dict[cell_type][end_time]
                # per-track time-averages
                temp_array = self.temporal_average_dict[cell_type][end_time]
                # compute signed diffs: shape (n_tracks, n_features)
                diffs = np.abs(spatial_vec - temp_array)

                # for each feature compute mean & std over tracks
                row = {'end_time': end_time}
                
                for i, feat in enumerate(self.features):
                   
                    row[f"{feat}"] = diffs[i]
                rows.append(row)

            results[cell_type] = pd.DataFrame(rows)

        return results          