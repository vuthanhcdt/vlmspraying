import numpy as np
import matplotlib.pyplot as plt
from google import genai
from google.genai import types 
import os 
import re 
from scipy.interpolate import interp1d 
from scipy.interpolate import make_interp_spline 
import matplotlib.cm as cm
import argparse 


DEFAULT_NPZ_FILE = 'tree_locations.npz' 
USE_VLM = False 
ROW_NPZ_FILE = 'crop_rows.npz'
ROW_OFFSET_DIST = 1.5
EXTEND_DIST = 1.0      
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')


def plot_plant_locations_simple(file_path):
    """
    Plots the plant locations and saves a temporary image required for the VLM analysis.
    """
    output_filename = 'tree_locations.png' 
    
    try:
        data = np.load(file_path)

        if 'objects_with_id' not in data:
            print(" Error: Key 'objects_with_id' not found in file.")
            print(f"Available keys in file: {list(data.keys())}")
            return None 

        all_data = data['objects_with_id']
        all_data = all_data[:497, :]
        if all_data.ndim != 2 or all_data.shape[1] < 4:
            print(f" Error: Array 'objects_with_id' has shape {all_data.shape}, expected (N, 4).")
            return None
        
        num_plants = all_data.shape[0]
        print(f" Successfully loaded {num_plants} data points.")

        plant_ids = all_data[:, 0]
        X_all = all_data[:, 1]
        Y_all = all_data[:, 2]
        
        plt.figure(figsize=(14, 10))  

        plt.scatter(
            X_all, 
            Y_all,
            color='green',
            s=80,     
            alpha=0.8,
            edgecolors='k',
            linewidths=0.5
        )
        
        for i in range(num_plants):
            plt.text(
                X_all[i],                   
                Y_all[i] + 0.5,             
                str(int(plant_ids[i])),     
                fontsize=12,                
                ha='center',
                va='bottom',                
                alpha=0.9,
                color='black',
            )
        
        plt.xlabel('X Coordinate', fontsize=14)
        plt.ylabel('Y Coordinate', fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.4)
        
        # Save temporarily for the VLM to read
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"\n Temporary plot saved for VLM to: **{output_filename}**")
        plt.close() 
        return output_filename 

    except FileNotFoundError:
        print(f" Error: File not found: {file_path}. Please check the filename.")
        return None
    except Exception as e:
        print(f" An error occurred while processing data: {e}")
        return None
    
def extract_rows_from_vlm_text(vlm_text):
    row_data = {}
    matches = re.findall(r"Row\s*([0-9]+)\s*:\s*([0-9,\s]+)", vlm_text, re.IGNORECASE)
    
    if not matches:
        print(" Format 'Row X: ID_LIST' not found in VLM response.")
        return {}
    
    for row_num_str, id_list_str in matches:
        try:
            row_num = int(row_num_str)
            plant_ids = [
                int(id_str.strip()) 
                for id_str in id_list_str.split(',') 
                if id_str.strip()
            ]
            
            if plant_ids:
                row_data[row_num] = plant_ids
        except ValueError as e:
            print(f" Data conversion error: {e}. Skipping row {row_num_str}.")
            continue
            
    if row_data:
        print(f" Extracted {len(row_data)} rows from VLM response.")
    return row_data

def compute_centerline_between_two_rows(all_data, row1_ids, row2_ids, num_points=100):
    id_to_coords = {int(all_data[i,0]): (all_data[i,1], all_data[i,2]) 
                    for i in range(all_data.shape[0])}
    
    row1_coords = np.array([id_to_coords[i] for i in row1_ids if i in id_to_coords])
    row2_coords = np.array([id_to_coords[i] for i in row2_ids if i in id_to_coords])

    if len(row1_coords) == 0 or len(row2_coords) == 0:
        return None, None

    row1_coords = row1_coords[np.argsort(row1_coords[:,0])]
    row2_coords = row2_coords[np.argsort(row2_coords[:,0])]

    x_min = min(row1_coords[:,0].min(), row2_coords[:,0].min())
    x_max = max(row1_coords[:,0].max(), row2_coords[:,0].max())
    x_center = np.linspace(x_min, x_max, num_points)

    f1 = interp1d(row1_coords[:,0], row1_coords[:,1], kind='linear', fill_value='extrapolate')
    f2 = interp1d(row2_coords[:,0], row2_coords[:,1], kind='linear', fill_value='extrapolate')

    y1 = f1(x_center)
    y2 = f2(x_center)

    y_center = (y1 + y2)/2

    return x_center, y_center

def compute_offset_line_from_row(x_row, y_row, offset_dist=0.5):
    dx = np.gradient(x_row)
    dy = np.gradient(y_row)
    length = np.sqrt(dx**2 + dy**2)
    
    nx = -dy / length
    ny = dx / length

    x_offset = x_row + nx * offset_dist
    y_offset = y_row + ny * offset_dist
    return x_offset, y_offset

def compute_auto_offset_line(x_row, y_row, row_num, row_data, all_data, offset_dist=0.5):
    x_offset_plus, y_offset_plus = compute_offset_line_from_row(x_row, y_row, offset_dist)
    x_offset_minus, y_offset_minus = compute_offset_line_from_row(x_row, y_row, -offset_dist)

    sorted_rows = sorted(row_data.keys())
    
    if len(sorted_rows) < 2:
        return None, None
    
    if row_num == sorted_rows[0]:
        next_row_ids = row_data[sorted_rows[1]]
        next_coords = np.array([all_data[np.where(all_data[:,0]==pid)[0][0],1:3] for pid in next_row_ids])
        next_coords = next_coords[np.argsort(next_coords[:, 0])]
        
        mid_idx = len(x_row) // 2
        start_idx = max(0, mid_idx - 2)
        end_idx = min(len(x_row), mid_idx + 3)
        
        mean_y_plus = y_offset_plus[start_idx:end_idx].mean()
        mean_y_minus = y_offset_minus[start_idx:end_idx].mean()
        
        next_mid_idx = len(next_coords) // 2
        next_start_idx = max(0, next_mid_idx - 2)
        next_end_idx = min(len(next_coords), next_mid_idx + 3)
        mean_y_next = next_coords[next_start_idx:next_end_idx, 1].mean()
        
        if abs(mean_y_plus - mean_y_next) > abs(mean_y_minus - mean_y_next):
            return x_offset_plus, y_offset_plus
        else:
            return x_offset_minus, y_offset_minus
            
    elif row_num == sorted_rows[-1]:
        prev_row_ids = row_data[sorted_rows[-2]]
        prev_coords = np.array([all_data[np.where(all_data[:,0]==pid)[0][0],1:3] for pid in prev_row_ids])
        prev_coords = prev_coords[np.argsort(prev_coords[:, 0])]

        mid_idx = len(x_row) // 2
        start_idx = max(0, mid_idx - 2)
        end_idx = min(len(x_row), mid_idx + 3)

        mean_y_plus = y_offset_plus[start_idx:end_idx].mean()
        mean_y_minus = y_offset_minus[start_idx:end_idx].mean()

        prev_mid_idx = len(prev_coords) // 2
        prev_start_idx = max(0, prev_mid_idx - 2)
        prev_end_idx = min(len(prev_coords), prev_mid_idx + 3)
        mean_y_prev = prev_coords[prev_start_idx:prev_end_idx, 1].mean()
        
        if abs(mean_y_plus - mean_y_prev) > abs(mean_y_minus - mean_y_prev):
            return x_offset_plus, y_offset_plus
        else:
            return x_offset_minus, y_offset_minus
    else:
        return None, None 

def extend_line(x, y, extend_dist=1.0):
    if len(x) < 2:
        return x, y
        
    dx_start = x[1] - x[0]
    dy_start = y[1] - y[0]
    length_start = np.sqrt(dx_start**2 + dy_start**2)
    x0_new = x[0] - dx_start/length_start * extend_dist
    y0_new = y[0] - dy_start/length_start * extend_dist

    dx_end = x[-1] - x[-2]
    dy_end = y[-1] - y[-2]
    length_end = np.sqrt(dx_end**2 + dy_end**2)
    xN_new = x[-1] + dx_end/length_end * extend_dist
    yN_new = y[-1] + dy_end/length_end * extend_dist

    x_extended = np.concatenate([[x0_new], x, [xN_new]])
    y_extended = np.concatenate([[y0_new], y, [yN_new]])
    return x_extended, y_extended

def smooth_path_corners(path_x, path_y, k_degree=3, smooth_factor=200):
    if len(path_x) < 4:
        return path_x, path_y

    points = np.vstack((path_x, path_y)).T
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)

    try:
        spline_2d = make_interp_spline(dist, points, k=k_degree)
    except Exception as e:
        print(f"Error creating B-spline: {e}. Keeping original path.")
        return path_x, path_y

    dist_smooth = np.linspace(dist.min(), dist.max(), smooth_factor)
    
    x_smooth, y_smooth = spline_2d(dist_smooth).T

    return x_smooth, y_smooth

def save_row_data_npz(row_data, filename='tree_rows.npz'):
    try:
        np.savez_compressed(filename, row_data=dict(row_data))
        print(f" Row data saved to {filename}")
    except Exception as e:
        print(f" Failed to save row data: {e}")


def compute_path_line_labels(path_x, path_y, angle_threshold_deg=8.0, min_segment_length=5):
    """Label each path point by line segment number using heading changes."""
    if len(path_x) < 2:
        return np.array([], dtype=np.int32)

    dx = np.diff(path_x)
    dy = np.diff(path_y)
    headings = np.arctan2(dy, dx)
    headings = np.unwrap(headings)

    dhead = np.abs(np.diff(headings))
    dhead = np.minimum(dhead, 2 * np.pi - dhead)

    threshold = np.deg2rad(angle_threshold_deg)
    corners = np.where(dhead > threshold)[0] + 1

    boundaries = [0]
    for idx in corners:
        if idx - boundaries[-1] >= min_segment_length:
            boundaries.append(idx)

    if boundaries[-1] != len(path_x):
        boundaries.append(len(path_x))

    labels = np.zeros(len(path_x), dtype=np.int32)
    for label_num, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), start=1):
        labels[start:end] = label_num

    return labels

def load_row_data_npz(filename='tree_rows.npz'):
    if not os.path.exists(filename):
        print(f" File {filename} does not exist. Must run VLM.")
        return None
    try:
        data = np.load(filename, allow_pickle=True)
        row_data = data['row_data'].item()
        print(f" Row data loaded from {filename}")
        return row_data
    except Exception as e:
        print(f" Failed to load row data: {e}")
        return None

def save_robot_path_to_npz(path_x, path_y, output_filename='global_path.npz', line_labels=None):
    if len(path_x) != len(path_y) or len(path_x) == 0:
        print(" Error: Path data is empty or dimension mismatch.")
        return None

    num_points = len(path_x)
    timestamps = np.arange(num_points)
    z_coords = np.zeros(num_points)
    q_default = np.zeros((num_points, 4))
    q_default[:, 3] = 1.0  # qw = 1
    path_full = np.column_stack((
        timestamps,
        path_x,
        path_y,
        z_coords,
        q_default
    ))

    if line_labels is None:
        line_labels = compute_path_line_labels(path_x, path_y)
    else:
        line_labels = np.asarray(line_labels, dtype=np.int32)
        if line_labels.shape[0] != num_points:
            raise ValueError('line_labels length must match path length')
    try:
        np.savez_compressed(output_filename, path=path_full, line_labels=line_labels)
        print(f"\n Robot path successfully saved to NPZ: **{output_filename}**")
        return output_filename
    except Exception as e:
        print(f" Error saving NPZ file: {e}")
        return None


def build_id_to_coord_map(all_data):
    return {int(row[0]): (float(row[1]), float(row[2])) for row in all_data}


def get_row_coordinates(row_ids, id_map):
    coords = [id_map[pid] for pid in row_ids if pid in id_map]
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    return np.array(coords, dtype=float)


def plot_spraying_strategy(all_data, row_data, path, output_file, line_labels=None):
    id_map = build_id_to_coord_map(all_data)
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = cm.get_cmap('tab20', max(len(row_data), 1))

    sorted_rows = sorted(row_data.keys())
    for idx, row_num in enumerate(sorted_rows):
        coords = get_row_coordinates(row_data[row_num], id_map)
        if coords.size == 0:
            continue

        x_row, y_row = coords[:, 0], coords[:, 1]
        color = cmap(idx)

        ax.scatter(
            x_row,
            y_row,
            color=color,
            s=110,
            alpha=0.9,
            edgecolors='k',
            linewidths=0.5,
            label=f'Row {row_num}'
        )

        for tree_id in row_data[row_num]:
            if tree_id in id_map:
                ax.text(
                    id_map[tree_id][0],
                    id_map[tree_id][1] + 0.35,
                    str(tree_id),
                    fontsize=8,
                    ha='center',
                    va='bottom',
                    color='black'
                )

    path_x = path[:, 1]
    path_y = path[:, 2]

    ax.plot(
        path_x,
        path_y,
        color='blue',
        linewidth=2.5,
        linestyle='--',
        alpha=0.85,
        label='Global Path'
    )

    if line_labels is None:
        try:
            line_labels = compute_path_line_labels(path_x, path_y)
        except Exception:
            line_labels = np.zeros(len(path_x), dtype=np.int32)

    unique_labels = np.unique(line_labels)
    for lab in unique_labels:
        if lab == 0:
            continue
        idxs = np.where(line_labels == lab)[0]
        if idxs.size == 0:
            continue
        mid_idx = idxs[len(idxs) // 2]
        ax.text(
            path_x[mid_idx],
            path_y[mid_idx] + 0.6,
            f'Line {int(lab)}',
            fontsize=10,
            ha='center',
            va='bottom',
            color='black',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
            zorder=8
        )

    if len(path_x) > 0:
        ax.scatter(
            [path_x[0]],
            [path_y[0]],
            color='green',
            s=140,
            marker='o',
            edgecolors='k',
            linewidths=1.0,
            zorder=6,
            label='Start'
        )
        ax.scatter(
            [path_x[-1]],
            [path_y[-1]],
            color='red',
            s=140,
            marker='o',
            edgecolors='k',
            linewidths=1.0,
            zorder=6,
            label='End'
        )

    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')


def create_robot_path_custom_start_end(all_data, row_data, start_row_id, end_row_id, offset_dist=2.0, extend_dist=1.0):
    sorted_rows = sorted(row_data.keys())
    
    if start_row_id not in sorted_rows or end_row_id not in sorted_rows:
        print(f" Error: Start row ID ({start_row_id}) or end row ID ({end_row_id}) is invalid.")
        return np.array([]), np.array([])
        
    start_index = sorted_rows.index(start_row_id)
    end_index = sorted_rows.index(end_row_id)

    if start_index <= end_index:
        sequence_rows = sorted_rows[start_index : end_index + 1]
        is_forward = True
    else:
        sequence_rows = sorted_rows[end_index : start_index + 1]
        sequence_rows.reverse()
        is_forward = False

    print(f" Creating route in sequence: {sequence_rows}. (Forward: {is_forward})")
    path_x = []
    path_y = []
    path_labels = []
    label_counter = 1
    current_row_id = sequence_rows[0]
    next_row_id = sequence_rows[1] if len(sequence_rows) > 1 else current_row_id 
    coords_current = np.array([all_data[np.where(all_data[:,0]==pid)[0][0],1:3] for pid in row_data[current_row_id]])
    x_current_sorted = coords_current[:,0][np.argsort(coords_current[:,0])]
    y_current_sorted = coords_current[:,1][np.argsort(coords_current[:,0])]

    x_offset_start, y_offset_start = compute_auto_offset_line(
        x_current_sorted, y_current_sorted, current_row_id, row_data, all_data, offset_dist=offset_dist
    )
    
    if x_offset_start is None:
        x_offset_start, y_offset_start = compute_offset_line_from_row(x_current_sorted, y_current_sorted, offset_dist)

    x_offset_start, y_offset_start = extend_line(x_offset_start, y_offset_start, extend_dist)
    
    x_c_check, y_c_check = compute_centerline_between_two_rows(all_data, row_data[current_row_id], row_data[next_row_id])
    
    if x_c_check is not None:
        if not is_forward:
            x_offset_start = x_offset_start[::-1]
            y_offset_start = y_offset_start[::-1]

    path_x.extend(x_offset_start)
    path_y.extend(y_offset_start)
    path_labels.extend([label_counter] * len(x_offset_start))
    last_point = np.array([path_x[-1], path_y[-1]])

    for i in range(len(sequence_rows)-1):
        r1 = sequence_rows[i]
        r2 = sequence_rows[i+1]
        
        x_c, y_c = compute_centerline_between_two_rows(all_data, row_data[r1], row_data[r2])
        if x_c is None: continue
        
        x_c_ext, y_c_ext = extend_line(x_c, y_c, extend_dist)

        start_c = np.array([x_c_ext[0], y_c_ext[0]])
        end_c = np.array([x_c_ext[-1], y_c_ext[-1]])
        
        dist_start = np.linalg.norm(last_point - start_c)
        dist_end = np.linalg.norm(last_point - end_c)
        
        if dist_end < dist_start:
            x_c_ext = x_c_ext[::-1]
            y_c_ext = y_c_ext[::-1]

        label_counter += 1
        path_x.extend(x_c_ext)
        path_y.extend(y_c_ext)
        path_labels.extend([label_counter] * len(x_c_ext))
        last_point = np.array([path_x[-1], path_y[-1]])


    last_row_id = sequence_rows[-1]
    
    coords_last = np.array([all_data[np.where(all_data[:,0]==pid)[0][0],1:3] for pid in row_data[last_row_id]])
    x_last_sorted = coords_last[:,0][np.argsort(coords_last[:,0])]
    y_last_sorted = coords_last[:,1][np.argsort(coords_last[:,0])]
    
    x_offset_last, y_offset_last = compute_auto_offset_line(
        x_last_sorted, y_last_sorted, last_row_id, row_data, all_data, offset_dist=offset_dist
    )
    
    if x_offset_last is None:
        x_offset_last, y_offset_last = compute_offset_line_from_row(x_last_sorted, y_last_sorted, -offset_dist) 

    x_offset_last_ext, y_offset_last_ext = extend_line(x_offset_last, y_offset_last, extend_dist)

    start_offset = np.array([x_offset_last_ext[0], y_offset_last_ext[0]])
    end_offset = np.array([x_offset_last_ext[-1], y_offset_last_ext[-1]])

    dist_start = np.linalg.norm(last_point - start_offset)
    dist_end = np.linalg.norm(last_point - end_offset)
    
    if dist_end < dist_start:
        x_offset_last_ext = x_offset_last_ext[::-1]
        y_offset_last_ext = y_offset_last_ext[::-1]

    label_counter += 1
    path_x.extend(x_offset_last_ext)
    path_y.extend(y_offset_last_ext)
    path_labels.extend([label_counter] * len(x_offset_last_ext))

    return np.array(path_x), np.array(path_y), np.array(path_labels, dtype=np.int32)

import os

def run_vlm_analysis(image_file):
    if image_file and os.path.exists(image_file):
        try:
            with open(image_file, 'rb') as f:
                image_bytes = f.read()

            if not GOOGLE_API_KEY:
                print(" API key not found. Please set GEMINI_API_KEY or OPENAI_API_KEY environment variable as described in README.")
                return None

            client = genai.Client(api_key=GOOGLE_API_KEY)
            
            vlm_prompt = (
                "You are an image analysis tool specialized in determining the row structure and listing plant IDs in each row."
                "\n\n--- Few-Shot Examples ---\n\n"
                "Sample Image A (3 rows, 15 points):"
                "\nTotal number of rows: 3"
                "\nRow 1: 1,2,3,4,5"
                "\nRow 2: 6,7,8,9,10"
                "\nRow 3: 11,12,13,14,15"
                "\n\nSample Image B (4 rows, 20 points):"
                "\nTotal number of rows: 4"
                "\nRow 1: 1,2,3,4,5"
                "\nRow 2: 6,7,8,9,10"
                "\nRow 3: 11,12,13,14,15"
                "\nRow 4: 16,17,18,19,20"
                "\n\n--- Analysis of the Current Image ---\n\n"
                "Based on the currently provided image, please accurately determine the total number of rows and list the plant IDs belonging to each row. The output must strictly follow the format 'Row X: ID_LIST' for each row. Ignore any path lines on the image."
            )

            contents_payload = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                vlm_prompt
            ]

            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=contents_payload
                )
            except Exception as e:
                error_message = str(e)
                if "503" in error_message or "UNAVAILABLE" in error_message:
                    print(" Model 'gemini-2.5-flash' is overloaded (Error 503). Retrying with fallback model (gemini-3.5-flash)...")
                    fallback_model = 'gemini-3.5-flash' 
                    response = client.models.generate_content(
                        model=fallback_model,
                        contents=contents_payload
                    )
                else:
                    raise e

            vlm_text = response.text
            print("\n--- VLM Analysis Results ---")
            print(vlm_text)
            return extract_rows_from_vlm_text(vlm_text)

        except Exception as e:
            print(f" VLM processing failed: {e}")
            return None
    else:
        print("\n VLM analysis skipped because the image file could not be generated or found.")
        return None


def main(your_npz_file, reverse_path): 
    print("## 1. Loading Data and Initial Plot")
    try:
        data = np.load(your_npz_file)
        all_data = data['objects_with_id']
    except FileNotFoundError:
        print(f" Error: File {your_npz_file} not found. Cannot proceed.")
        return
    except Exception as e:
        print(f" Error loading data: {e}")
        return

    image_file = plot_plant_locations_simple(your_npz_file)
    if not image_file:
        return

    print("\n## 2. Running VLM/Loading Row Data")
    row_data_from_vlm = load_row_data_npz(ROW_NPZ_FILE)
    if not row_data_from_vlm or USE_VLM:
        row_data_from_vlm = run_vlm_analysis(image_file)
        if row_data_from_vlm:
            save_row_data_npz(row_data_from_vlm, ROW_NPZ_FILE)

    if not row_data_from_vlm:
        print(" No row data available to draw Centerline/Offset. Stopping.")
        return

    all_row_ids = sorted(row_data_from_vlm.keys())
    
    START_ROW = all_row_ids[0]
    END_ROW = all_row_ids[-1]
    
    print(f"\n## 4. Creating Final Robot Path (Custom Start/End: Row {START_ROW} -> Row {END_ROW})")
    
    path_x_rough, path_y_rough, path_labels_rough = create_robot_path_custom_start_end(
        all_data, 
        row_data_from_vlm,
        start_row_id=START_ROW,
        end_row_id=END_ROW,
        offset_dist=ROW_OFFSET_DIST,
        extend_dist=EXTEND_DIST
    )

    if reverse_path:
        path_x_rough = path_x_rough[::-1]
        path_y_rough = path_y_rough[::-1]
        path_labels_rough = path_labels_rough[::-1]
        print(" Path has been reversed: Start and End swapped.")

    if len(path_x_rough) == 0:
        print(" Error: Could not generate robot path.")
        return

    try:
        path_x, path_y = smooth_path_corners(path_x_rough, path_y_rough, k_degree=3, smooth_factor=len(path_x_rough)*2)
        print(" Path smoothed using B-spline.")
    except Exception as e:
        print(f" Error during path smoothing: {e}. Keeping angular path.")
        path_x, path_y = path_x_rough, path_y_rough

    try:
        labels_smoothed = np.zeros(len(path_x), dtype=np.int32)
        for i in range(len(path_x)):
            dx = path_x_rough - path_x[i]
            dy = path_y_rough - path_y[i]
            idx = np.argmin(dx*dx + dy*dy)
            labels_smoothed[i] = path_labels_rough[idx]
    except Exception:
        labels_smoothed = None

    save_robot_path_to_npz(path_x, path_y, 'global_path.npz', line_labels=labels_smoothed)


    print("\n## 5. Visualizing Final Robot Coverage Path (Optimized)")

    path_full = np.column_stack((
        np.arange(len(path_x)),
        path_x,
        path_y,
        np.zeros((len(path_x), 5))
    ))

    plot_spraying_strategy(all_data, row_data_from_vlm, path_full, 'global_path.png', line_labels=labels_smoothed)
    print(" Final visualization saved to: **global_path.png**")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Path planning for tree farm coverage.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--npz_path', 
        type=str, 
        nargs='?', 
        default=DEFAULT_NPZ_FILE, 
        help=f"Path to the input NPZ data file containing 'objects_with_id'. \n(Default: {DEFAULT_NPZ_FILE})"
    )

    parser.add_argument(
        '--reverse', 
        action='store_true', 
        help="If specified, the path order will be reversed (swap Start Row and End Row)."
    )

    parser.add_argument(
        '--run_vlm', 
        action='store_true',
        help="If specified, the VLM analysis will be forced to rerun, overwriting existing row data."
    )

    args = parser.parse_args()

    USE_VLM = args.run_vlm
    
    print("--- ⚙️ Configuration ---")
    print(f"Input NPZ File: {args.npz_path}")
    print(f"Reverse Path: {args.reverse}")
    print(f"Force Run VLM: {USE_VLM}")
    print("------------------------")


    main(args.npz_path, args.reverse)