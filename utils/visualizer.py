
#!/usr/bin/env python3
"""
Cardiac MRI Visualization Tool

This script visualizes processed cardiac MRI data stored as numpy (.npy) files.
It can display multiple types of images (raw, cine, cine_whole) side by side for 
comparison and analysis, with navigation between different slices.

Usage:
    python cardiac_mri_visualizer.py [options]

Environment Variables:
    PROCESSED_DATA_DIR: Directory containing processed MRI data
"""
"""
The cine slice is wrong - iut should be that - what cuses this kdin of error?

"""

# TODO: Get algorithm in state where I can visualize LGE Output from the model as ground truth


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg') # WHYIM: use for rendering via x11 forwarding


def parse_arguments():
    """Parse command line arguments for the visualization tool."""
    parser = argparse.ArgumentParser(
        description="Cardiac MRI Data Visualization Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Define arguments
    parser.add_argument(
        "-d", "--data-dir",
        type=str,
        help="Directory containing processed MRI data (overrides PROCESSED_DATA_DIR env var)"
    )
    
    parser.add_argument(
        "-p", "--patients",
        type=str,
        nargs="+",
        help="Specific patient IDs to visualize (default: all patients)"
    )
    
    parser.add_argument(
        "-t", "--types",
        type=str,
        nargs="+",
        choices=["raw", "cine", "cine_whole"],
        default=["raw", "cine", "cine_whole", "lge"],
        help="Types of images to display"
    )
    
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6],
        help="Figure size (width, height) in inches"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="Figure resolution (dots per inch)"
    )
    
    return parser.parse_args()


def get_data_directory(cmd_arg=None):
    """
    Get the data directory path from command line argument or environment variable.
    
    Args:
        cmd_arg: Command line argument for data directory
        
    Returns:
        Path object to the data directory
    """
    # First check command line argument
    if cmd_arg:
        data_dir = Path(cmd_arg)
    # Then check environment variable
    elif os.environ.get("PROCESSED_DATA_DIR"):
        data_dir = Path(os.environ.get("PROCESSED_DATA_DIR"))
    else:
        raise ValueError(
            "Data directory not specified. Either set PROCESSED_DATA_DIR environment "
            "variable or use the --data-dir command line option."
        )
    
    # Verify directory exists
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    return data_dir


def get_patient_directories(data_dir, patient_ids=None):
    """
    Get directories for specified patients or all patients if none specified.
    
    Args:
        data_dir: Path to data directory
        patient_ids: List of specific patient IDs to include
        
    Returns:
        Dictionary mapping patient IDs to their directory paths
    """
    patient_dirs = {}
    
    # List all subdirectories in the data directory
    all_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    
    if not all_dirs:
        raise FileNotFoundError(f"No patient directories found in {data_dir}")
    
    # Filter by patient_ids if specified
    if patient_ids:
        for patient_id in patient_ids:
            patient_dir = data_dir / patient_id
            if patient_dir.exists() and patient_dir.is_dir():
                patient_dirs[patient_id] = patient_dir
            else:
                print(f"Warning: Patient directory not found: {patient_id}")
    else:
        # Use all directories
        for patient_dir in all_dirs:
            patient_dirs[patient_dir.name] = patient_dir
    
    if not patient_dirs:
        raise FileNotFoundError("No valid patient directories found")
    
    return patient_dirs

def get_files_by_regex(patient_dir, img_type):
    """
    Find files that match a specific image type using regex patterns.
    
    Args:
        patient_dir: Path to patient directory (pathlib.Path object)
        img_type: Type of image to match ("raw", "cine", "cine_whole", etc.)
        
    Returns:
        List of file paths that match the pattern for the specified image type
    """
    import re
    # Define regex patterns for each image type
    patterns = {
        "raw": r"^raw_\d+\.npy$",
        "cine": r"^cine_\d+\.npy$",
        "cine_whole": r"^cine_whole_\d+\.npy$",
        "lge": r"^lge_\d+\.npy$"
    }
    
    # Make sure the requested image type has a defined pattern
    if img_type not in patterns:
        print(f"Warning: No pattern defined for image type '{img_type}'")
        return []
    
    # Get the pattern for this image type
    pattern = re.compile(patterns[img_type])
    
    # Find all files in the directory
    all_files = list(patient_dir.iterdir())
    
    # Filter files that match the pattern
    matching_files = [f for f in all_files if f.is_file() and pattern.match(f.name)]
    
    return matching_files
def find_slice_files(patient_dir, image_types):
    """
    Find and organize slice files for a patient by type and slice number.
    
    Args:
        patient_dir: Path to patient directory
        image_types: List of image types to include
        
    Returns:
        Dictionary mapping slice numbers to dictionaries of image types and file paths
    """
    slice_files = {}


    # Process each image type
    for img_type in image_types:
        #pattern = f"{img_type}_*.npy"  # this pattern is to global and liberal i need to find this
        #files = list(patient_dir.glob(pattern))

        files = get_files_by_regex(patient_dir, img_type)


        for file_path in files:
            # Extract slice number from filename (e.g., raw_5.npy -> 5)
            try:
                filename = file_path.name
                # Get all of the stuff

                # Find the last underscore in the filename
                last_underscore_pos = filename.rindex('_')

                # Extract everything between the last underscore and the file extension
                slice_num = int(filename[last_underscore_pos+1:filename.index('.')])              

                # Initialize dictionary for this slice if it doesn't exist
                if slice_num not in slice_files:
                    slice_files[slice_num] = {}
                
                # Add file path for this type
                slice_files[slice_num][img_type] = file_path
            except (ValueError, IndexError) as e:



                print(f"Warning: Could not parse slice number from filename: {file_path}")
    
    return slice_files


def load_and_validate_slice(slice_data, image_types):
    """
    Load slice files and validate that requested image types are available.
    
    Args:
        slice_data: Dictionary mapping image types to file paths
        image_types: List of image types to include
        
    Returns:
        Dictionary mapping image types to numpy arrays
    """
    images = {}
    
    # Check if all requested image types are available
    missing_types = [t for t in image_types if t not in slice_data]
    if missing_types:
        print(f"Warning: Missing image types for this slice: {', '.join(missing_types)}")
    
    # Load available images
    for img_type in image_types:
        if img_type in slice_data:
            try:
                img_data = np.load(slice_data[img_type])
                images[img_type] = img_data

            except (IOError, ValueError) as e:
                print(f"Error loading {img_type} image: {e}")
    
    return images


def create_visualization(patient_dirs, image_types, figsize=(12, 6), dpi=100):
    """
    Create an interactive visualization of cardiac MRI data.
    
    Args:
        patient_dirs: Dictionary mapping patient IDs to directory paths
        image_types: List of image types to display
        figsize: Figure size as (width, height) in inches
        dpi: Figure resolution (dots per inch)
    """
    # Find all patients' slice files
    all_patients_data = {}
    for patient_id, patient_dir in patient_dirs.items():

        try:
            slice_files = find_slice_files(patient_dir, image_types)
            if slice_files:
                all_patients_data[patient_id] = slice_files
            else:
                print(f"Warning: No valid slice files found for patient {patient_id}")
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
    
    if not all_patients_data:
        print("Error: No valid data found for any patient")
        return


    # Setup the figure and axes for visualization
    num_cols = len(image_types)
    fig, axes = plt.subplots(1, num_cols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(bottom=0.25)  # Make room for sliders
    
    # If there's only one image type, ensure axes is a list
    if num_cols == 1:
        axes = [axes]
    
    # Set up patient selection and navigation controls
    patient_ids = list(all_patients_data.keys())
    current_patient = patient_ids[0]
    
    # Get slice numbers for current patient
    slice_numbers = sorted(all_patients_data[current_patient].keys())
    current_slice = slice_numbers[0] if slice_numbers else 0
    
    # Initialize images
    img_objects = []
    
    def update_display(patient_id, slice_num):
        """Update the display with the specified patient and slice."""
        # Clear existing images
        for ax in axes:
            ax.clear()
            
        # Load new images
        try:
            slice_data = all_patients_data[patient_id][slice_num]
            images = load_and_validate_slice(slice_data, image_types)
           


            # Update each axis with new image Key place or uuncertainty inside? 
            for i, img_type in enumerate(image_types):
                if img_type in images:
                    img_data = images[img_type] # key focus what happens image type for patient file? TODO: Examine via breakpiont
                    
                    

                    img_obj = axes[i].imshow(img_data, cmap='gray')

                    if img_type == 'lge': 
                        axes[i].set_title(f"{img_type} (ground truth from SwinUnet) (Slice {slice_num})")
                    else:
                        axes[i].set_title(f"{img_type} (Slice {slice_num})")



                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f"No {img_type} data", 
                                 horizontalalignment='center',
                                 verticalalignment='center')
            
            fig.suptitle(f"Patient: {patient_id}", fontsize=16)
            plt.draw()
            
        except (KeyError, IndexError) as e:
            print(f"Error updating display: {e}")
    
    # Create slider for slice selection
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=ax_slice,
        label='Slice',
        valmin=0,
        valmax=max(slice_numbers) if slice_numbers else 0,
        valinit=current_slice,
        valstep=slice_numbers
    )
    
    def update_slice(val):
        nonlocal current_slice
        current_slice = int(val)
        update_display(current_patient, current_slice)
    
    slice_slider.on_changed(update_slice)
    
    # Create patient selection buttons
    ax_patient = plt.axes([0.25, 0.05, 0.65, 0.03])
    patient_slider = Slider(
        ax=ax_patient,
        label='Patient',
        valmin=0,
        valmax=len(patient_ids) - 1,
        valinit=0,
        valstep=1
    )
    
    def update_patient(val):
        nonlocal current_patient, current_slice, slice_numbers
        patient_idx = int(val)
        current_patient = patient_ids[patient_idx]
        
        # Update slice slider for new patient
        slice_numbers = sorted(all_patients_data[current_patient].keys())
        
        if not slice_numbers:
            print(f"Warning: No slices found for patient {current_patient}")
            return
        
        slice_slider.valmax = max(slice_numbers)
        slice_slider.valstep = slice_numbers
        
        # Reset to first slice for new patient
        current_slice = slice_numbers[0]
        slice_slider.set_val(current_slice)
        
        update_display(current_patient, current_slice)
    
    patient_slider.on_changed(update_patient)
    
    # Add navigation buttons
    ax_prev = plt.axes([0.1, 0.05, 0.1, 0.05])
    btn_prev = Button(ax_prev, 'Previous')
    
    ax_next = plt.axes([0.85, 0.05, 0.1, 0.05])
    btn_next = Button(ax_next, 'Next')
    
    def go_to_prev(event):
        current_idx = patient_ids.index(current_patient)
        new_idx = max(0, current_idx - 1)
        patient_slider.set_val(new_idx)
    
    def go_to_next(event):
        current_idx = patient_ids.index(current_patient)
        new_idx = min(len(patient_ids) - 1, current_idx + 1)
        patient_slider.set_val(new_idx)
    
    btn_prev.on_clicked(go_to_prev)
    btn_next.on_clicked(go_to_next)
    
    # Initial display
    update_display(current_patient, current_slice)
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])  # Adjust layout to make room for controls
    plt.show()





def main():
    """Main function to run the visualization tool."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Get data directory
        data_dir = get_data_directory(args.data_dir)
        print(f"Using data directory: {data_dir}")


        # Get patient directories
        patient_dirs = get_patient_directories(data_dir, args.patients)
        print(f"Found {len(patient_dirs)} patient directories")
        
        # Create visualization
        create_visualization(
            patient_dirs, 
            args.types, 
            figsize=args.figsize, 
            dpi=args.dpi
        )
        
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nVisualization terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
