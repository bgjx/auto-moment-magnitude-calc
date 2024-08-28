#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Fri Dec 15 19:32:03 2023
@author : ARHAM ZAKKI EDELO
@contact: edelo.arham@gmail.com
"""
import copy
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from obspy.geodetics import gps2dist_azimuth


def build_raw_model(model_top: List[List], velocity: List) -> List[List]:
    """
    Build a model of layers from the given top and bottom depths and velocities.

    Args:
        model_top (List[List]): List of lists where each sublist contains top and bottom depths for a layer.
        velocity (List): List of layer velocities.

    Returns:
        List[List]: List containing sublists where each sublist represents a layer with:
            - The top depth of the layer (in km).
            - The thickness of the layer (in km).
            - The velocity of the layer (in km/s).
    """
    MAX_DEPTH = 3
    model = list()
    for layer, (top, bottom) in enumerate(model_top):
        thick = (top - bottom)*1000
        model.append([-1000*top, thick, velocity[layer]])
        if  bottom > MAX_DEPTH:
            break
            
    return model



def upward_model (hypo_depth: float, sta_elev: float, model_raw: List[List]) -> List[List]:
    """
    Build a modified model for direct upward-refracted waves from the raw model by evaluating 
    the hypo depth and the station elevation.

    Args:
        hypo_depth(float) : Depth in meter (!!! depth marked with negative sign).
        sta_elev (float): Elevation of station
        model_raw (List[List]): List containing sublist where each sublist represents top depth, thickness, and velocity of each layer.

    Returns:
        List[List] : List containing sublists where each sublist represents a modified layer with:
            - The top depth of the layer (in km).
            - The thickness of the layer (in km).
            - The velocity of the layer (in km/s).
    """
    # correct upper model boundary and last layer thickness
    sta_pos = -1
    hypo_pos = -1
    for layer in model_raw:
        if layer[0] >= max(sta_elev, hypo_depth):
            sta_pos+=1
            hypo_pos+=1
        elif layer[0] >= hypo_depth:
            hypo_pos+=1
        else:
            pass
    modified_model = model_raw[sta_pos:hypo_pos+1]
    modified_model[0][0] = sta_elev
    if len(modified_model) > 1:
        modified_model[0][1] = float(modified_model[1][0]) - sta_elev
        modified_model[-1][1] = hypo_depth - modified_model[-1][0]
    else:
        modified_model[0][1] =  hypo_depth - sta_elev
    upward_model = modified_model
    
    return upward_model
 

 
def downward_model(hypo_depth: float, sta_elev: float, model_raw: List[List]) -> List[List]:
    """
    Build a modified model for downward critically refracted waves from the raw model by evaluating 
    the hypo depth and the station elevation.

    Args:
        hypo_depth(float) : Depth in meter (!!! depth marked with negative sign).
        sta_elev (float): Elevation of station
        model_raw (List[List]): List containing sublist where each sublist represents top depth,
                                thickness, and velocity of each layer.

    Returns:
        List[List] : List containing sublists where each sublist represents a modified layer with:
            - The top depth of the layer (in km).
            - The thickness of the layer (in km).
            - The velocity of the layer (in km/s).
    """
    
    hypo_pos = -1
    for layer in model_raw:
        if layer[0] >= hypo_depth:
            hypo_pos+=1
    modified_model = model_raw[hypo_pos:]
    modified_model[0][0] = hypo_depth
    if len(modified_model) > 1:
        modified_model[0][1] = float(modified_model[1][0]) - hypo_depth
    downward_model = modified_model
    return downward_model
   
   

def up_refract (epi_dist: float, 
                up_model: List[List], 
                angles: np.ndarray
                ) -> Tuple[Dict[str, List], float]:
    """
    Calculate the refracted angle (relative to the normal line), the cumulative distance traveled, 
    and the total travel time for all layers based on the direct upward refracted wave.

    Args:
        epi_dist (float): the Epicenter distance in m.
        up_model (List[List]): List of sublists containing modified raw model results from the 'upward_model' function.
        angles (np.ndarray): A numpy array of pre-defined angles for a grid search.

    Returns:
        Tuple[Dict[str, List], float]:
            - holder_up (Dict[str, List]): A dictionary of all direct upward refracted waves, 
                containing refracted angles, cumulative distances, and travel times for each layer. 
            - final_take_off (float): The final take-off angle of the refracted wave reaches the station.
        
    """
    holder_up = defaultdict(dict)
    final_take_off = str
    for angle in angles:
        holder_up[f"take_off_{angle}"]= {'refract_angle':[], 'distance':[], 'tt':[]}
        start_dist = 0
        angle_emit = angle
        for i,j in zip(range(1, len(up_model)+1), range(2, len(up_model)+2)):
            i=-1*i; j= -1*j           
            dist = np.tan(angle_emit*np.pi/180)*abs(up_model[i][1])  # cumulative distance, in abs since the thickness is in negative
            tt = abs(up_model[i][1])/(np.cos(angle_emit*np.pi/180)*(up_model[i][-1]*1000)) 
            start_dist += dist
            if start_dist > epi_dist:
                break
            holder_up[f"take_off_{angle}"]['refract_angle'].append(angle_emit)
            holder_up[f"take_off_{angle}"]['distance'].append(start_dist)
            holder_up[f"take_off_{angle}"]['tt'].append(tt)
            try:
                angle_emit = 180*(np.arcsin(np.sin(angle_emit*np.pi/180)*up_model[j][-1]/up_model[i][-1]))/np.pi 
            except IndexError:
                break
        if start_dist > epi_dist:
            break
        final_take_off = angle
    return holder_up, final_take_off
      
      
      
def down_refract(epi_dist: float,
                    up_model: List[List],
                    down_model: List[List]
                    ) -> Tuple[Dict[str, List], Dict[str, List]] :
    """
    Calculate the refracted angle (relative to the normal line), the cumulative distance traveled, 
    and the total travel time for all layers based on the downward critically refracted wave.

    Args:
        epi_dist (float): the Epicenter distance in m.
        up_model (List[List]): List of sublists containing modified raw model results from the 'upward_model' function.
        down_model (List[List]): List of sublists containing modified raw model results from the 'downward_model' function.

    Returns:
        Tuple[Dict[str, List], Dict[str, List]]:
            - holder_down (Dict[str, List]): A dictionary of all downward segments of critically refracted waves, 
                containing refracted angles, cumulative distances, and travel times for each layer. 
            - holder_up_seg (Dict[str, List]): A dictionary of all upward segments of downward critically refracted waves, 
                containing refracted angles, cumulative distances, and travel times for each layer. 
    """
    half_dist = epi_dist/2
    # find all the critical angles in the model
    c_angle = list()
    if len(down_model) > 1:
        for i in range(0, len(down_model)):
            j = i+1
            try:
                c_a = 180*(np.arcsin(down_model[i][-1]/down_model[j][-1]))/np.pi
                c_angle.append(c_a)
            except IndexError:
                pass
    
    # find the first take-off angle for every critical angle
    holder_c = defaultdict(list)
    for i in range(0, len(c_angle)):
        if i < 1: 
            holder_c[i].append(c_angle[i])
        v = i + 1
        for k, j in zip (range(-v,0), range(-v+1,0)) :
            k *=-1; j*=-1
            c_angle[i] =  180*(np.arcsin(np.sin(c_angle[i]*np.pi/180)*down_model[j-1][-1]/down_model[k-1][-1]))/np.pi
            holder_c[i].append(c_angle[i])
            holder_c[i].sort()
        holder_c[i].append(c_angle[i])
    angles = [holder_c[i][0] for i in list(holder_c.keys())]
    angles.sort()


    holder_up_seg = defaultdict(dict)
    holder_down = defaultdict(dict)
    for angle in angles:
        start_dist = 0
        holder_down[f"take_off_{angle}"]= {'refract_angle':[], 'distance':[], 'tt':[]}
        angle_emit = angle
        for i,j in zip(range(0, len(down_model)), range(1, len(down_model)+1)):
            dist = np.tan(angle_emit*np.pi/180)*abs(down_model[i][1])  # abs since the thickness is in negative
            tt = abs(down_model[i][1])/(np.cos(angle_emit*np.pi/180)*(down_model[i][-1]*1000)) 
            start_dist += dist
            if start_dist > half_dist:
                break
            holder_down[f"take_off_{angle}"]['refract_angle'].append(angle_emit)
            holder_down[f"take_off_{angle}"]['distance'].append(start_dist)
            holder_down[f"take_off_{angle}"]['tt'].append(tt)
            
            # cek emit angle ray
            emit_deg = np.sin(angle_emit*np.pi/180)*down_model[j][-1]/down_model[i][-1]
            if emit_deg < 1:
                angle_emit = 180*(np.arcsin(np.sin(angle_emit*np.pi/180)*down_model[j][-1]/down_model[i][-1]))/np.pi
                continue
            
            elif emit_deg == 1:
                angle_emit = 90
                start_angle = holder_down[f"take_off_{angle}"]['refract_angle'][0]
                angle_up_segment =  np.array([start_angle])
                ray_up, take_off_ = up_refract(epi_dist, up_model, angle_up_segment)
                holder_up_seg.update(ray_up)
                try:
                    dist_up = ray_up[f'take_off_{start_angle}']['distance'][-1]
                    dist_critical = (2*half_dist) - (2*start_dist) - dist_up   # total flat line length
                except IndexError:
                    continue
                if dist_critical < 0:
                    continue
                tt_critical = (dist_critical / (down_model[j][-1]*1000))
                holder_down[f"take_off_{angle}"]['refract_angle'].append(angle_emit)
                holder_down[f"take_off_{angle}"]['distance'].append(dist_critical + start_dist)
                holder_down[f"take_off_{angle}"]['tt'].append(tt_critical)
                break
                
            else:
                break
    return  holder_down, holder_up_seg



def plot_rays (hypo_depth: float, 
                sta_elev: float,
                velocity: List, 
                base_model: List[List],
                up_model: List[List],
                down_model: List[List],
                reached_up_ref: Dict[str, List],
                c_ref: Dict[str, List],
                down_ref: Dict[str, List],
                down_up_ref: Dict[str, List],
                epi_dist: float
                ) -> None:
    """
    Plot the raw/base model, hypocenter, station, and the relative distance between the hypocenter and station
    and also plot all waves that reach the station.

    Args:
        hypo_depth(float) : Depth in meter (!!! depth marked with negative sign).
        sta_elev (float): Elevation of station
        base_model (List[List]): List containing sublist where each sublist represents top depth, 
            thickness, and velocity of each layer.
        up_model (List[List]): List of sublists containing modified raw model results from the 'upward_model' function.
        down_model (List[List]): List of sublists containing modified raw model results from the 'downward_model' function.
        reached_up_ref (Dict[str, List]): A dictionary of a direct upward refracted wave that reaches the station
            , containing refracted angles, cumulative distances, and travel times for each layer.
        c_ref (Dict[str, List]): A dictionary of a direct upward refracted wave that reaches the station
            , containing refracted angles, cumulative distances, and travel times for each layer.
        
        down_ref (Dict[str, List]): A dictionary of all downward segments of critically refracted waves, 
            containing refracted angles, cumulative distances, and travel times for each layer.
        down_up_ref (Dict[str, List]): A dictionary of all upward segments of downward critically refracted waves, 
            containing refracted angles, cumulative distances, and travel times for each layer. 
        epi_dist (float): the Epicenter distance in m.
        
    Returns:
        None
    """
    
    
    fig, axs = plt.subplots()
    
    # Define colormaps and normalization
    cmap = cm.Oranges
    norm = mcolors.Normalize(vmin=min(velocity), vmax=max(velocity))
    
    # Plot the layers
    max_width = epi_dist + 2000
    for layer in base_model:
        color = cmap(norm(layer[-1]))
        rect = patches.Rectangle((-1000, layer[0]), max_width, layer[1], linewidth=1, edgecolor= 'black', facecolor = color)
        axs.add_patch(rect)
    
    # Plot only the last ray of direct upward wave that reaches the station
    layer_cek = 0
    x1 = 0 
    for layer in reversed(up_model):
        if layer_cek == 0:
            y1 = layer[0] + layer[1]
        x2 = reached_up_ref['distance'][layer_cek]
        y2 = layer[0]
        axs.plot([x1,x2], [y1,y2], color = 'k')
        x1, y1 = x2, y2
        layer_cek+=1

    # Plot downward critically refracted rays and their upward segments that reach the station
    if len(c_ref):
        for take_off in c_ref.keys():
            layer_cek = 0
            x1 =  0
            for layer in down_model:
                if layer_cek == 0:
                    y1 = layer[0]
                x2 = down_ref[take_off]['distance'][layer_cek]
                
                # plot second half
                if down_ref[take_off]['refract_angle'][layer_cek] == 90:
                    y2 = layer[0]
                    axs.plot([x1,x2], [y1,y2], color = 'b')
                    
                    # plot the upward of the second half 
                    up_check = layer_cek
                    d1, r1 = x2, y2
                    for layer in reversed(down_model[:layer_cek]):
                        r2 = layer[0]
                        d2 =  d1 + down_ref[take_off]['distance'][up_check - 1] - down_ref[take_off]['distance'][up_check - 2]
                        if up_check == 1:
                            d2 =  d1 + down_ref[take_off]['distance'][up_check - 1]
                        axs.plot([d1,d2], [r1,r2], color = 'b')
                        d1, r1 = d2, r2
                        up_check -= 1

                    # Plotting upward ray segment from critically refracted wave
                    model_up_cek = 0
                    q1 = d2
                    for layer in reversed(up_model):
                        if model_up_cek == 0:
                            t1 = layer[0] + layer[1]
                        q2 = q1 + down_up_ref[take_off]['distance'][model_up_cek] - down_up_ref[take_off]['distance'][model_up_cek - 1]
                        if model_up_cek == 0:
                            q2 = q1 + down_up_ref[take_off]['distance'][model_up_cek]
                        t2 = layer[0] 
                        axs.plot([q1,q2], [t1,t2], color = 'b')
                        q1, t1 =q2, t2
                        model_up_cek+=1
                    break
                y2 = layer [0] + layer [1]
                axs.plot([x1,x2], [y1,y2], color = 'b')
                x1, y1 = x2, y2
                layer_cek +=1

    #plot the source and the station
    axs.plot(epi_dist, sta_elev, marker = 'v', color = 'black', markersize = 15)
    axs.text(epi_dist, sta_elev + 200 , 'STA', horizontalalignment='right', verticalalignment='center')
    axs.plot(0, hypo_depth, marker = '*', color = 'red', markersize = 12)
    axs.set_xlim(-2000,max_width)
    axs.set_ylim(-3000, 3000)
    axs.set_ylabel('Depth')
    axs.set_xlabel('Distance')
    axs.set_title('Shooting Snell Method')
    plt.show()
    
    return None

def calculate_inc_angle(hypo: List,
                        sta: List,
                        model: List[List],
                        velocity: List, 
                        plot_figure: bool = False
                        ) -> Tuple [float, float, float]:
    """
    Calculate the take-off angle, total travel-time and the incidence angle at the station for 
    refracted angle using Snell's shooting method.

    Args:
        hypo (List): A list containing the latitude, longitude, and depth of the hypocenter (depth in negative notation).
        sta (List): A list containing the latitude, longitude, and elevation of the station.
        model (List[List]): List of list where each sublist contains top and bottom depths for a layer.
        velocity (List): List of layer velocities.
        plot_figure (bool): Whether to generate and save figures (default is False).
        
    Returns:
        Tuple[float, float, float]: take-off angle, total travel time and incidence angle.
    """
    ANGLE_RESOLUTION = np.linspace(0, 90, 1000) # set grid resolution for direct upward refracted wave
    # initialize hypocenter, station, model, and calculate the epicentral distance
    [hypo_lat,hypo_lon, depth] = hypo
    [sta_lat, sta_lon, elev] = sta
    epicentral_distance, azimuth, back_azimuth = gps2dist_azimuth(hypo_lat, hypo_lon, sta_lat, sta_lon)
    
    # build a raw model
    model_raw = build_raw_model(model, velocity)
    
    # deep copy the model (original copy is  needed for plotting)
    model_copied = copy.deepcopy(model_raw)
    model_copied_2 = copy.deepcopy(model_raw)
    up_model = upward_model (depth, elev, model_copied)
    down_model = downward_model(depth, elev, model_copied_2)
    
    #  start calculating all refracted waves for all layers they may propagate through
    up_ref, final_take_off = up_refract(epicentral_distance, up_model, ANGLE_RESOLUTION)
    down_ref, down_up_ref = down_refract(epicentral_distance, up_model, down_model)
    
    # result from direct upward refracted wave only
    last_ray = up_ref[f"take_off_{final_take_off}"]
    take_off_upward_refract = 180 - last_ray['refract_angle'][0]
    upward_refract_tt = np.sum(last_ray['tt'])
    reach_distance = last_ray['distance'][-1] # -1 index since distance is cumulative value
    upward_incidence_angle = last_ray['refract_angle'][-1]

    # result from direct downward critically refracted
    checker_take_off = list()
    for key in down_ref.keys():
        try:
            if down_ref[key]['refract_angle'][-1] == 90:
                first_refract_angle = down_ref[key]['refract_angle'][0]
                checker_take_off.append(first_refract_angle)
        except IndexError:
            pass

    c_refract = defaultdict(dict) # list of downward critically refracted ray (take_off_angle, total_tt, incidence_angle)
    if len(checker_take_off):
        for take_off in checker_take_off:
            if len(down_up_ref[f"take_off_{take_off}"]['distance']) < len(last_ray['distance']):  # only take the reach signal
                continue
            c_refract[f'take_off_{take_off}'] = {'total_tt':[], 'incidence_angle': []}
            tt_downward = np.sum(down_ref[f"take_off_{take_off}"]['tt'])
            tt_upward_sec_half = np.sum(down_ref[f"take_off_{take_off}"]['tt'][:-1])
            tt_refract_upward = np.sum(down_up_ref[f"take_off_{take_off}"]['tt'])
            total_tt = tt_downward + tt_upward_sec_half + tt_refract_upward
            incidence_angle = down_up_ref[f"take_off_{take_off}"]['refract_angle'][-1]
            c_refract[f'take_off_{take_off}']['total_tt'].append(total_tt)
            c_refract[f'take_off_{take_off}']['incidence_angle'].append(incidence_angle)
            
    if len(c_refract): 
        fastest_ray = 9999
        fastest_key = str
        for key in c_refract.keys():
            if c_refract[key]['total_tt'][-1] < fastest_ray:     # find the fastest ray from downward refraction
                fastest_ray = c_refract[key]['total_tt'][-1]
                fastest_key = key

        if c_refract[fastest_key]['total_tt'][-1] > upward_refract_tt:
            takeoff = take_off_upward_refract
            total_travel = upward_refract_tt
            incidence_angle = upward_incidence_angle

        else:
            takeoff = fastest_key
            total_travel = c_refract[fastest_key]['total_tt'][-1]
            incidence_angle = c_refract[fastest_key]['incidence_angle'][-1]
    else:
        takeoff = take_off_upward_refract
        total_travel = upward_refract_tt
        incidence_angle = upward_incidence_angle

    # If plot image statement is true    
    if plot_figure:
        plot_rays(depth, elev, velocity, model_raw, up_model, down_model, last_ray, c_refract, down_ref, down_up_ref, epicentral_distance)

        # print report        
        print("* Direct upward refracted solutions (black line):\n",
                "Take off angle:",take_off_upward_refract,'\n',
                "Incidence angle:", upward_incidence_angle,'\n',
                "Travel time:", upward_refract_tt, '\n',
                "Reached distance:", reach_distance, '\n'
                )                
        
        if len(c_refract): 
            print("\n* List of all critically refracted solutions (blue lines):")
            fastest_ray = 9999
            fastest_key = str
            count = 1
            for key in c_refract.keys():
                print(count,'.', key, ':', c_refract[key])
                if c_refract[key]['total_tt'][-1] < fastest_ray:     # find the fastest ray from downward refraction
                    fastest_ray = c_refract[key]['total_tt'][-1]
                    fastest_key = key
                count +=1
            
            print("\n* Fastest critically refracted solution:",'\n',
                    f"Take off angle : {fastest_key[9:]} \n",
                    f"Total travel time: {c_refract[fastest_key]['total_tt'][-1]}\n",
                    f"Incidence angle: {c_refract[fastest_key]['incidence_angle'][-1]}")
        
            print("\n* Final shooting snell solutions:")
            if c_refract[fastest_key]['total_tt'][-1] > upward_refract_tt:
                print(
                    f"Take off angle: {take_off_upward_refract} \n",
                    f"Total travel time: {upward_refract_tt}\n",
                    f"Incidence angle: {upward_incidence_angle}")
            else:
                print(
                    f"Take off angle: {fastest_key} \n",
                    f"Total travel time: {c_refract[fastest_key]['total_tt'][-1]}\n",
                    f"Incidence angle: {c_refract[fastest_key]['incidence_angle'][-1]}")
        else:
            print("\nNo critically refracted rays reach the station are available....")

    return takeoff, total_travel, incidence_angle

# End of functions 
#============================================================================================

if __name__ == "__main__" :
    # Parameters
    # model parameters (list of list, consisting of top, bottom boundaries, and velocity (P phase in km/s)
    model_top = [
                    [-3.00, -1.90], [-1.90, -0.59], [-0.59, 0.22], [0.22, 2.50], [2.50, 7.00], [7.00, 9.00], [9.00, 15.00], [15.00, 33.00], [33.00, 99999.00] 
    ]
    velocity_P = [2.68, 2.99, 3.95, 4.50, 4.99, 5.60, 5.80, 6.40, 8.00]
    
    # hypo and station
    hypo_test = [37.916973, 126.651613, 200]
    sta = [ 37.916973, 126.700882, 2200]
    
    # call the function to start the calculation
    ray_inc = calculate_inc_angle(hypo_test, sta, model_top, velocity_P, plot_figure = True)