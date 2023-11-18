<!-- MathJax configuration -->
<script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']]
        }
    };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<!-- Your Markdown content -->

# Converting tracks to Feature Matrices

## Structure of Master XML

- AllSpots: <Spot ID (same as TrackMate Spot ID), name, Intensity, t,z,y,x,radius,tissue xyz, distance-cell-mask, track duration, radial angle, motion angle, speed, acceleration, eccentricity 
            components, surface area, cell axis mask, unique Napatrackmater ID, tracklet ID, generation ID>
- AllTracks: <Track TrackID (same as TrackMate Track ID), Track tzyx, Track speed, Track duration, Edge Spot source ID, Spot target ID, Directional rate of change, t,z,y,x,displacement>            

- Filtered track ids: list of selected track ids.

## Functional structure

- xml_content = parseXML(xml path)
- filtered_tracks = xml_content.find("Model").find("FilteredTracks").find("TrackID")
- spotobjects = xml_content.find("Model").find("AllSpots")
- tracks = xml_content.find("Model").find("AllTracks")

## Loops

- for frame in spotobjects.findall("SpotsinFrame"):
               _master_spot_computer(frame)
- _master_spot_computer:
         Most important dictionaries: unique_spot_properties[cell_id] = [cell_id, t,z,y,x,Intensity (mean and total), radius, Volume, distance cell to mask, unique track id, tracklet id, generation id, motion angle, speed, acceleration, radial angle, surface area, eccentricity components, cell axis mask]
- for track in tracks.findall("Track"):
               track_id = track.get(track_id_key)
               _master_track_computer(track, track_id)
- _master_track_computer:
         Most important dictionaries: track_mitosis_label[track_id] = [1, number_dividing] or [0,0] if the trjaectory belongs to a dividing cell or non-dividing cell
                                      unique_spot_centroid[t,z,y,x] = cell_id
                                      unique_track_centroid[t,z,y,x] = track_id
         Lists: AllTrackIds, Dividing TrackIDs, Non-Dividing TrackIDs          

## TrackMate Csvs

- dataframe object: spot_dataset = read(spot_csv_path)
- dataframe object: track_dataset = read(track_csv_path)
- dataframe object: edges_dataset = read(edges_csv_path)
Used in the Napari plugin for coloring the tracks and segmentation labels

# TrackVector

This is a class that converts the data from master xml file and spots, edges, tracks csv to feature matrices. A region of interest in TZYX can be entered and the tracks will only be extracted from the temporal voxel and it extracts the above mentioned dictionaries and lists and constructs new dictionaries:

### Unique Tracks

This dictionary contains for each TrackMate track id an array of tracklets, for a non dividing trajectory there is just one tracklet while for a dividing trajectory it is a union of tracklets of mother and daughter cells which are distinguished by thier unique tracklet id attached to them. The unique tracklet id contains the TrackMate track id and the generation and tracklet id of the corresponding tracklet.
- unique_tracks[track_id] = tracklets = array(unique napatrackmate ID, t, z, y, x)

### Unique Track Properties

Like the dictionary above this dictionary has tracklet properties 

- unique_track_properties[track_id] = tracklet_properties = array(t, unique napatrackmate ID, Shape Features, Dynamic Features, Intensity)

- shape features = [radius, volume in pixels, eccentricity components, surface area]
- dynamic deatures = [speed, motion angle, accceleration, distance, cell mask, radial angle, cell axis mask] 

## Tracklet Dictionaries

For each tracklet of each track we then construct the following dictionaries:

- unique_shape_properties_tracklet[unique_tracklet_id] = [t, z, y, x, radius, volume,...]
- unique_dynamic_properties_tracklet[unique_tracklet_id] = [t. speed, motion angle, acceleration,...]
- unique_dynamic_properties[track_id (TrackMate Track ID)] = {unique_tracklet_id: unique_dynamic_properties_tracklet}
- unique_shape_properties[track_id (TrackMate Track ID)] = {unique_tracklet_id: unique_shape_properties_tracklet}

These dictionaries ensure that we have full information of all the tracklets of a given TrackMate track ID. Then we construct the final data frame object that allows us to create the track matrices:

shape_dynamic_vectors = [] = [array(tracklet_id, t, z, y, x, shape properties (extracted from unique_shape_properties), dynamic properties (extracted from unique_dynamic_properties))]

Then we convert this to a pandas dataframe object and iterate over the tracklet_id to obtain for each tracklet ID the following Track Matrices:



\begin{bmatrix}
\text{time 1} & \text{time 2} & \text{time 3} & \cdots \\
\text{shape features} & \text{shape features} & \text{shape features} & \cdots \\
\text{dynamic features} & \text{dynamic features} & \text{dynamic features} & \cdots \\
\end{bmatrix}


This breaks down each tracklet into a (T, 11) dimensional matrix and we apply machine learning techniques on (K,T_k,11) dimensional tensor as explained in the following sections, K being the number of tracks.

# Machine Learning for cell fate quantification

Having breaken down tracks into tracklets and tracklets into feature matrix of shape (K,T_k,11) 11 being the shape and dynamic features computed, T_k being the timepoints for tracklet number K. For each tracklet we compute the covariance matrix which converts the matric (T_k,11) to (11,11) matrix. For K tracks this gives us (K,11,11) dimensional matrix. To ascertain which features had the most variance we compute an averaged covaraince matrix of shape (11,11) as shown here  

![image](images/track_animation_dividing.gif)