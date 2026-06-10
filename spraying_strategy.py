import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from google import genai
from google.genai import types 
from PIL import Image

DEFAULT_TREE_FILE = 'tree_locations.npz'
DEFAULT_ROWS_FILE = 'crop_rows.npz'
DEFAULT_PATH_FILE = 'global_path.npz'
DEFAULT_PLOT_FILE = 'spraying_strategy.png'
MAX_SPRAY_RATE = 5.8 
GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

def load_tree_locations(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f'Tree locations file not found: {filename}')
    data = np.load(filename)
    if 'objects_with_id' not in data:
        raise ValueError(f"Expected key 'objects_with_id' in {filename}, found: {list(data.keys())}")
    objects = data['objects_with_id']
    if objects.ndim != 2 or objects.shape[1] < 3:
        raise ValueError(f'Invalid objects_with_id shape: {objects.shape}')
    return objects

def load_crop_rows(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Crop row file not found: {filename}. "
            "Run crop_row_detection.py first to generate the row classification file, "
            "or pass a valid file path with --rows."
        )
    data = np.load(filename, allow_pickle=True)
    if 'row_data' not in data:
        raise ValueError(f"Expected key 'row_data' in {filename}, found: {list(data.keys())}")
    row_data = data['row_data'].item()
    if not isinstance(row_data, dict):
        raise ValueError('row_data must be a dictionary mapping row numbers to plant IDs')
    return {int(k): [int(pid) for pid in v] for k, v in row_data.items()}

def load_global_path(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f'Global path file not found: {filename}')
    data = np.load(filename)
    if 'path' not in data:
        raise ValueError(f"Expected key 'path' in {filename}, found: {list(data.keys())}")
    path = data['path']
    if path.ndim != 2 or path.shape[1] < 3:
        raise ValueError(f'Invalid path shape: {path.shape}')
    line_labels = data['line_labels'] if 'line_labels' in data else None
    return path, line_labels

def build_id_to_coord_map(objects):
    return {int(row[0]): (float(row[1]), float(row[2])) for row in objects}

def get_row_coordinates(row_ids, id_map):
    coords = [id_map[pid] for pid in row_ids if pid in id_map]
    if len(coords) == 0:
        return np.empty((0, 2), dtype=float)
    return np.array(coords, dtype=float)


def compute_line_tree_sequences(path, line_labels, objects, max_distance=3.0):
    id_map = build_id_to_coord_map(objects)
    sequences = {}
    if line_labels is None or path is None:
        return sequences

    unique_labels = [int(lab) for lab in np.unique(line_labels) if lab != 0]
    for lab in sorted(unique_labels):
        idxs = np.where(line_labels == lab)[0]
        sequence = []
        last_tree = None
        for i in idxs:
            pt = np.array([path[i, 1], path[i, 2]])
            nearest_tree = None
            nearest_dist = float('inf')
            for tree_id, coord in id_map.items():
                dist = np.linalg.norm(pt - np.array(coord))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_tree = tree_id
            if nearest_tree is not None and nearest_dist <= max_distance and nearest_tree != last_tree:
                sequence.append(nearest_tree)
                last_tree = nearest_tree
        sequences[lab] = sequence
    return sequences


def format_line_definitions(line_sequences):
    lines = []
    if not line_sequences:
        return ""
    max_lab = max(line_sequences.keys())
    
    for lab in sorted(line_sequences.keys()):
        seq = line_sequences[lab]
        if not seq:
            continue
        tree_list = ', '.join(str(tid) for tid in seq)

        if lab == 1:
            aiming = "Path is ABOVE Row 1. The robot MUST aim SOUTH to spray Row 1."
        elif lab == max_lab:
            row_num = max_lab - 1
            aiming = f"Path is BELOW Row {row_num}. The robot MUST aim NORTH to spray Row {row_num}."
        else:
            upper_row = lab - 1
            lower_row = lab
            aiming = f"Path is between Row {upper_row} and Row {lower_row}. The robot alternates aiming NORTH (at Row {upper_row}) and SOUTH (at Row {lower_row})."
            
        lines.append(f'Line_{lab}: {aiming} Target Tree_IDs: [{tree_list}]')
    return '\n'.join(lines)

def plot_spraying_strategy(objects, row_data, path, output_file, wind_speed, wind_direction, crop_type, pesticide_type, line_labels=None):
    id_map = build_id_to_coord_map(objects)
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = cm.get_cmap('tab20', max(len(row_data), 1))
    sorted_rows = sorted(row_data.keys())
    for idx, row_num in enumerate(sorted_rows):
        coords = get_row_coordinates(row_data[row_num], id_map)
        if coords.size == 0:
            continue
        x_row, y_row = coords[:, 0], coords[:, 1]
        color = cmap(idx)

        ax.scatter(x_row, y_row, color=color, s=110, alpha=0.9,
                   edgecolors='k', linewidths=0.5, label=f'Row {row_num}')

        for i, tree_id in enumerate(row_data[row_num]):
            if tree_id in id_map:
                ax.text(id_map[tree_id][0], id_map[tree_id][1] + 0.35,
                        str(tree_id), fontsize=8, ha='center', va='bottom', color='black')

    path_x, path_y = path[:, 1], path[:, 2]
    ax.plot(path_x, path_y, color='blue', linewidth=2.5, linestyle='--',
            alpha=0.85, label='Global Path')
    if line_labels is None and 'line_labels' in locals():
        line_labels = locals().get('line_labels')
    if line_labels is not None:
        unique_labels = np.unique(line_labels)
        unique_labels = unique_labels[unique_labels != 0]
        if unique_labels.size > 0:
            label_cmap = cm.get_cmap('tab20', max(len(unique_labels), 1))
            last_label = unique_labels.max()
            for i, lab in enumerate(unique_labels):
                idxs = np.where(line_labels == lab)[0]
                if idxs.size == 0:
                    continue
                mid_idx = idxs[len(idxs) // 2]
                color = label_cmap(i)
                y_offset = 0.6
                va = 'bottom'
                if lab == last_label:
                    y_offset = -0.6
                    va = 'top'

                ax.text(path_x[mid_idx], path_y[mid_idx] + y_offset, f'Line {int(lab)}',
                    fontsize=10, ha='center', va=va, color=color,
                    fontweight='bold', bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'), zorder=9)
                runs = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1) if idxs.size > 0 else []
                for run in runs:
                    if run.size > 1:
                        ax.plot(path_x[run], path_y[run], color=color, linewidth=3.5, linestyle='-',
                                alpha=0.95, zorder=11)
                    else:
                        j = run[0]
                        ax.scatter([path_x[j]], [path_y[j]], color=color, s=40, zorder=11)

    if len(path_x) > 0:
        ax.scatter([path_x[0]], [path_y[0]], color='green', s=140, marker='o',
                   edgecolors='k', linewidths=1.0, zorder=6, label='Start')
        ax.scatter([path_x[-1]], [path_y[-1]], color='red', s=140, marker='o',
                   edgecolors='k', linewidths=1.0, zorder=6, label='End')

    angle_rad = np.radians(wind_direction)
    dx = np.cos(angle_rad) * 0.08
    dy = np.sin(angle_rad) * 0.08
    title = "Operational Context"
    info_cols = [f"Wind: {wind_speed} m/s @ {wind_direction}°",
         f"Crop: {crop_type}",
         f"Pesticide: {pesticide_type}",
         f"Temp: 28°C"]
    info_line = "   |   ".join(info_cols)

    ax.text(0.5, 1.04, title, transform=ax.transAxes, fontsize=12, fontweight='bold',
        ha='center', va='bottom', zorder=10)

    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    ax.text(0.5, 1.01, info_line, transform=ax.transAxes, fontsize=11,
        ha='center', va='bottom', bbox=props, zorder=10)

    arrow_base = np.array([0.08, 0.08])
    arrow_length = 0.06
    ux = np.cos(angle_rad)
    uy = np.sin(angle_rad)

    perp = np.array([-uy, ux]) * 0.03

    for i in range(3):
        offset = perp * (i - 1) 
        start = (arrow_base + offset)
        end = (start + np.array([ux, uy]) * arrow_length)
        ax.annotate('', xy=tuple(end), xytext=tuple(start),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(facecolor='cyan', edgecolor='black', shrink=0, width=4, headwidth=10), zorder=12)

    ax.text(arrow_base[0] + 0.03, arrow_base[1] - 0.03, "WIND",
            transform=ax.transAxes, fontsize=10, color='teal', fontweight='bold', ha='left', zorder=12)

    ax.set_xlabel('X coordinate', fontsize=14)
    ax.set_ylabel('Y coordinate', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)




def summarize_data(objects, row_data, path):
    total_trees = objects.shape[0]
    total_rows = len(row_data)
    path_length = path.shape[0]
    
    print(f'Loaded {total_trees} tree locations.')
    print(f'Loaded {total_rows} classified rows.')
    print(f'Loaded global path with {path_length} points.')

def generate_spraying_decision_with_gemini(image_path, wind_speed, wind_direction, crop_type, pesticide_type, path, line_labels, objects, row_data, travel_speed=0.6, spray_rate=5.0):
    api_key = GOOGLE_API_KEY
    line_sequences = compute_line_tree_sequences(path, line_labels, objects)
    line_definitions = format_line_definitions(line_sequences)

    if not api_key:
        print("API key not found in environment variables. Set GEMINI_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY.")
        return

    client = genai.Client(api_key=api_key)

    prompt = f"""
        You are an Autonomous Spraying Decision System for an intelligent agricultural robot operating in an orchard environment. 

        CURRENT OPERATIONAL CONDITIONS:
        - Wind Speed: {wind_speed} m/s
        - Wind Direction: {wind_direction} degrees (relative to the map grid. 90° means blowing UPWARDS from South to North. 270° means blowing DOWNWARDS from North to South).
        - Temperature: 28°C
        - Humidity: 65%
        - Crop Type: {crop_type}
        - Pesticide Type: {pesticide_type}

        The path has the following labeled line segments (CRITICAL: USE THESE EXACT AIMING DIRECTIONS):
        {line_definitions}

        CRITICAL BEHAVIORAL RULES FOR STRATEGY:
        1. TREE-LEVEL DYNAMIC RATE (RGB VISION): DO NOT apply a single uniform Spray_Rate. The rate MUST fluctuate slightly tree-by-tree based on assumed variations in canopy volume/density.
        2. WIND COMPENSATION LOGIC (THE OVERRIDE): 
           - First, analyze the wind vector ({wind_direction}°). Does it blow North or South?
           - When the robot aims its spray AGAINST the wind direction, apply a HIGH pressure range (4.5 - 5.8 L/min).
           - When the robot aims its spray WITH the wind direction, apply a LOW pressure range (2.0 - 3.5 L/min).
        3. SPEED VARIATION: Adjust Travel_Speed (0.5 to 1.5 m/s). Use faster speeds (~0.8 - 1.0 m/s) on straight single-row lines (Line 1 and the last line), and slower speeds (~0.6 m/s) on alternating lines.

        Output Format: Line_x: (Tree_ID, Travel_Speed, Spray_Rate)

        CoT Step 1: Wind Vector Analysis
        Analyze the {wind_direction}° wind direction. State explicitly: "The wind is blowing towards the [North/South]."

        CoT Step 2: Directional Range Mapping
        Based on Step 1, explicitly map the ranges: 
        - "Aiming NORTH is [WITH/AGAINST] the wind -> Requires [LOW range (2.0-3.5) / HIGH range (4.5-5.8)]."
        - "Aiming SOUTH is [WITH/AGAINST] the wind -> Requires [LOW range (2.0-3.5) / HIGH range (4.5-5.8)]."

        CoT Step 3: Trajectory Breakdown
        Match the AIMING DIRECTION from the Line Definitions to the rules in Step 2. 
        Explicitly explain Line_1: "Line_1 aims SOUTH. This is [WITH/AGAINST] wind. Using [HIGH/LOW] range."

        CoT Step 4: Final Decision Output
        CRITICAL INSTRUCTION: Output the final tuples directly as raw text. 
        WARNING 1: DO NOT WRITE PYTHON CODE, SCRIPTS, OR LOOPS. Just output the plain text lines.
        WARNING 2: You MUST simulate tree-by-tree variation manually in your text. Do NOT output the exact same decimal rate twice in a row for the same row. Generate realistic, fluctuating float numbers within the correct ranges.

        Example Output Format:
        Line_1: (1, 0.8, 5.1), (2, 0.8, 4.8), (3, 0.8, 5.3)
        Line_2: (16, 0.6, 2.9), (32, 0.6, 5.2), (15, 0.6, 3.1), (31, 0.6, 4.9)
        """
    
    primary_model = 'gemini-2.5-flash'
    fallback_model = 'gemini-3.5-flash'
    
    print(f"Sending request to {primary_model}...")
    print(f"   Context -> Wind: {wind_speed}m/s | Dir: {wind_direction}° | Crop: {crop_type} | Pesticide: {pesticide_type}")
    
    img = Image.open(image_path)
    response = None
    
    try:
        response = client.models.generate_content(
            model=primary_model,
            contents=[prompt, img]
        )
    except Exception as e:
        error_msg = str(e)
        if '503' in error_msg or 'UNAVAILABLE' in error_msg:
            print(f"{primary_model} is currently overloaded (503). Switching to fallback model: {fallback_model}...")
            try:
                response = client.models.generate_content(
                    model=fallback_model,
                    contents=[prompt, img]
                )
            except Exception as fallback_e:
                print(f"Error calling fallback Gemini API ({fallback_model}): {fallback_e}")
                return
        else:
            print(f"Error calling Gemini API: {e}")
            return

    if response:
        print("\n" + "="*50)
        print("RESPONSE FROM GEMINI:")
        print("="*50)
        print(response.text)
        print("="*50)


def main(tree_file, rows_file, path_file, output_image, wind_speed, wind_direction, crop_type, pesticide_type):
    objects = load_tree_locations(tree_file)
    row_data = load_crop_rows(rows_file)
    path, line_labels = load_global_path(path_file)

    summarize_data(objects, row_data, path)

    plot_spraying_strategy(objects, row_data, path, output_image, wind_speed, wind_direction, crop_type, pesticide_type, line_labels=line_labels)
    print(f'Visualization saved to {output_image}')
        
    generate_spraying_decision_with_gemini(output_image, wind_speed, wind_direction, crop_type, pesticide_type, path, line_labels, objects, row_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize global path together with classified crop rows and generate VLM Spraying Strategy.'
    )
    parser.add_argument('--tree', default=DEFAULT_TREE_FILE, help='Tree location NPZ file')
    parser.add_argument('--rows', default=DEFAULT_ROWS_FILE, help='Crop row classification NPZ file')
    parser.add_argument('--path', default=DEFAULT_PATH_FILE, help='Global path NPZ file')
    parser.add_argument('--output-image', default=DEFAULT_PLOT_FILE, help='Output visualization PNG file')

    parser.add_argument('--wind-speed', type=float, default=3.0, help='Wind speed in m/s (e.g., 2.5, 4.0)')
    parser.add_argument('--wind-direction', type=float, default=270.0, help='Wind direction in degrees (0=East, 90=North)')
    
    parser.add_argument('--crop-type', type=str, default='Mango', help='Type of crop being sprayed')
    parser.add_argument('--pesticide-type', type=str, default='Cypermethrin', help='Type of pesticide being used')

    args = parser.parse_args()
    main(
        tree_file=args.tree,
        rows_file=args.rows,
        path_file=args.path,
        output_image=args.output_image,
        wind_speed=args.wind_speed,
        wind_direction=args.wind_direction,
        crop_type=args.crop_type,
        pesticide_type=args.pesticide_type
    )