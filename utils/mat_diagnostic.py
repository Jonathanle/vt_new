#!/usr/bin/env python3
"""
Cardiac MRI Preprocessing Diagnostic Script

This script analyzes the input .mat files and output .npy files from the MAT preprocessing
pipeline to document their structure, content, and the relationships between them.

Usage:
    python mat_diagnostic.py --input-dir ./data/Matlab/ --output-dir ./cropped_myo_lge_testing/ --report-dir ./diagnostic_report/

The script will:
1. Analyze input .mat files structure and content
2. Analyze output .npy files structure and content
3. Map the relationships between inputs and outputs
4. Generate a comprehensive report with visualizations
"""

import argparse
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Polygon
import pandas as pd
import json
from PIL import Image
import tomni
from datetime import datetime
import seaborn as sns
from pathlib import Path
import traceback

#TODO: Replace the default directories with config.py defined local directories



# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze cardiac MRI preprocessing pipeline data structures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./data/Matlab/', #DF - how to have a consitent 'root' directory that I can refrence while not being dependent
        help='Directory containing patient MATLAB (.mat) files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./cropped_myo_lge_testing/',
        help='Directory containing processed numpy (.npy) files'
    )
    
    parser.add_argument(
        '--report-dir',
        type=str,
        default='./diagnostic_report/',
        help='Directory to save the diagnostic report'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=3,
        help='Number of patient samples to analyze'
    )
    
    parser.add_argument(
        '--max-slices',
        type=int,
        default=5,
        help='Maximum number of slices per patient to analyze'
    )
    
    return parser.parse_args()

# Initialize global variables
report_data = {
    'mat_files': [],
    'npy_files': [],
    'transformations': [],
    'statistics': {
        'raw': {},
        'cine': {},
        'cine_whole': {},
        'lge': {}
    },
    'visualizations': []
}


def standardize_image(img):
    """Standardize image to 0-1 range"""
    if np.amin(img) == np.amax(img):
        return np.zeros_like(img)
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


def setup_report_directory(report_dir):
    """Create report directory and subdirectories"""
    os.makedirs(report_dir, exist_ok=True)
    
    # Create subdirectories
    dirs = ['mat_analysis', 'npy_analysis', 'transformations', 'statistics']
    for d in dirs:
        os.makedirs(os.path.join(report_dir, d), exist_ok=True)
    
    return report_dir


def analyze_mat_file(mat_path, report_dir, slice_limit=5):
    """
    Analyze a single .mat file and document its structure
    
    Args:
        mat_path: Path to the .mat file
        report_dir: Directory to save visualizations
        slice_limit: Maximum number of slices to analyze
    
    Returns:
        Dictionary with mat file analysis data
    """
    print(f"Analyzing MAT file: {mat_path}")
    mat_data = sio.loadmat(mat_path)
    
    # Extract file information
    file_info = {
        'filename': os.path.basename(mat_path),
        'directory': os.path.dirname(mat_path),
        'keys': list(mat_data.keys()),
        'structures': {}
    }
    
    # Check if this is a valid cardiac MRI file
    try:
        series_type = mat_data.get('series_type', ['Unknown'])[0]
        file_info['series_type'] = series_type
        is_valid = series_type == 'Myocardial Evaluation'
        file_info['is_valid'] = is_valid
        
        if not is_valid:
            print(f"  Warning: Not a Myocardial Evaluation file: {series_type}")
            return file_info
    except Exception as e:
        print(f"  Error checking series type: {e}")
        file_info['is_valid'] = False
        return file_info
    
    # Analyze main components
    try:
        # Number of slices
        slice_count = mat_data['enhancement'][0].shape[0]
        file_info['slice_count'] = slice_count
        
        # Analyze raw_image
        raw_shape = mat_data['raw_image'].shape
        file_info['structures']['raw_image'] = {
            'shape': raw_shape,
            'dtype': str(mat_data['raw_image'].dtype),
            'min': float(np.nanmin(mat_data['raw_image'])),
            'max': float(np.nanmax(mat_data['raw_image'])),
            'mean': float(np.nanmean(mat_data['raw_image'])),
            'std': float(np.nanstd(mat_data['raw_image'])),
            'has_nan': bool(np.isnan(mat_data['raw_image']).any())
        }
        
        # Analyze enhancement
        file_info['structures']['enhancement'] = {
            'shape': mat_data['enhancement'][0].shape,
            'dtype': str(mat_data['enhancement'].dtype),
            'min': float(np.nanmin(mat_data['enhancement'][0])),
            'max': float(np.nanmax(mat_data['enhancement'][0])),
            'has_nan': bool(np.isnan(mat_data['enhancement'][0]).any())
        }
        
        # Analyze contours if available
        for contour in ['lv_endo', 'lv_epi']:
            if contour in mat_data:
                try:
                    points_info = []
                    for i in range(min(slice_count, slice_limit)):
                        try:
                            points = mat_data[contour][0][i][0][0][0]
                            points_info.append({
                                'slice': i,
                                'points_count': points.shape[0],
                                'min_x': float(np.min(points[:, 0])),
                                'max_x': float(np.max(points[:, 0])),
                                'min_y': float(np.min(points[:, 1])),
                                'max_y': float(np.max(points[:, 1]))
                            })
                        except (IndexError, ValueError) as e:
                            points_info.append({
                                'slice': i,
                                'error': str(e)
                            })
                    
                    file_info['structures'][contour] = {
                        'available_slices': len(points_info),
                        'points_details': points_info
                    }
                except Exception as e:
                    file_info['structures'][contour] = {
                        'error': str(e)
                    }
        
        # Create visualizations for sample slices
        file_info['visualizations'] = []
        vis_dir = os.path.join(report_dir, 'mat_analysis', os.path.basename(mat_path).replace('.mat', ''))
        os.makedirs(vis_dir, exist_ok=True)
        
        for slice_idx in range(min(slice_count, slice_limit)):
            try:
                # Create a visualization with multiple subplots
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"MAT File Analysis - {os.path.basename(mat_path)} - Slice {slice_idx}", fontsize=16)
                
                # Raw image
                raw_img = np.transpose(mat_data['raw_image'][0, slice_idx])
                axs[0, 0].imshow(raw_img, cmap='gray')
                axs[0, 0].set_title(f"Raw Image (Slice {slice_idx})")
                axs[0, 0].axis('off')
                
                # Enhancement data
                enhancement = np.copy(mat_data['enhancement'][0][slice_idx]).astype('float')
                enhancement[enhancement == 0] = np.nan
                axs[0, 1].imshow(raw_img, cmap='gray')
                if not np.all(np.isnan(enhancement)):
                    axs[0, 1].imshow(enhancement, cmap='jet', alpha=0.7)
                axs[0, 1].set_title(f"Enhancement Overlay (Slice {slice_idx})")
                axs[0, 1].axis('off')
                
                # Image with contours
                axs[1, 0].imshow(raw_img, cmap='gray')
                
                # Add contours if available
                contour_drawn = False
                for contour, color in [('lv_endo', 'r'), ('lv_epi', 'g')]:
                    if contour in mat_data:
                        try:
                            points = mat_data[contour][0][slice_idx][0][0][0]
                            axs[1, 0].plot(points[:, 0], points[:, 1], color=color, linewidth=2, 
                                         label=f"{contour}")
                            contour_drawn = True
                        except (IndexError, ValueError) as e:
                            print(f"  Cannot draw contour {contour} for slice {slice_idx}: {e}")
                
                if contour_drawn:
                    axs[1, 0].legend()
                axs[1, 0].set_title(f"Image with Contours (Slice {slice_idx})")
                axs[1, 0].axis('off')
                
                # Myocardium segmentation based on contours
                try:
                    img_shape = raw_img.shape
                    
                    if 'lv_endo' in mat_data and 'lv_epi' in mat_data:
                        try:
                            myo_seg_endo = tomni.make_mask.make_mask_contour(
                                img_shape, mat_data['lv_endo'][0][slice_idx][0][0][0])
                            myo_seg_epi = tomni.make_mask.make_mask_contour(
                                img_shape, mat_data['lv_epi'][0][slice_idx][0][0][0])
                            myo_seg = (myo_seg_epi - myo_seg_endo).astype('float')
                            
                            axs[1, 1].imshow(raw_img, cmap='gray')
                            axs[1, 1].imshow(myo_seg, cmap='autumn', alpha=0.5)
                            axs[1, 1].set_title(f"Myocardium Segmentation (Slice {slice_idx})")
                        except Exception as e:
                            axs[1, 1].text(0.5, 0.5, f"Segmentation Error: {str(e)[:50]}...", 
                                         horizontalalignment='center', verticalalignment='center',
                                         transform=axs[1, 1].transAxes, fontsize=10)
                            axs[1, 1].set_title("Myocardium Segmentation (Failed)")
                    else:
                        axs[1, 1].text(0.5, 0.5, "Missing contour data", 
                                     horizontalalignment='center', verticalalignment='center',
                                     transform=axs[1, 1].transAxes)
                        axs[1, 1].set_title("Myocardium Segmentation (No Data)")
                except Exception as e:
                    axs[1, 1].text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=axs[1, 1].transAxes, fontsize=10)
                    axs[1, 1].set_title("Processing Error")
                
                axs[1, 1].axis('off')
                
                # Save the figure
                vis_path = os.path.join(vis_dir, f"slice_{slice_idx}.png")
                plt.tight_layout()
                plt.savefig(vis_path, dpi=150)
                plt.close()
                
                file_info['visualizations'].append({
                    'slice': slice_idx,
                    'path': vis_path
                })
                
            except Exception as e:
                print(f"  Error creating visualization for slice {slice_idx}: {e}")
                traceback.print_exc()
    
    except Exception as e:
        print(f"  Error analyzing MAT file: {e}")
        traceback.print_exc()
        file_info['error'] = str(e)
    
    return file_info


def analyze_npy_files(patient_dir, report_dir, slice_limit=5):
    """
    Analyze .npy files for a patient and document their structure
    
    Args:
        patient_dir: Directory containing patient's .npy files
        report_dir: Directory to save visualizations
        slice_limit: Maximum number of slices to analyze
    
    Returns:
        Dictionary with npy file analysis data
    """
    print(f"Analyzing NPY files in: {patient_dir}")
    
    patient_id = os.path.basename(patient_dir)
    npy_files = os.listdir(patient_dir)
    
    # Group files by slice number
    slices = {}
    for f in npy_files:
        if not f.endswith('.npy'):
            continue
        
        prefix = f.split('_')[0]
        try:
            slice_num = int(f.split('_')[1].split('.')[0])
            if slice_num not in slices:
                slices[slice_num] = {}
            slices[slice_num][prefix] = os.path.join(patient_dir, f)
        except (IndexError, ValueError):
            print(f"  Unexpected file naming format: {f}")
    
    # Analyze slices
    analysis_results = {
        'patient_id': patient_id,
        'total_slices': len(slices),
        'file_types': {},
        'slice_analyses': []
    }
    
    # Create visualization directory
    vis_dir = os.path.join(report_dir, 'npy_analysis', patient_id)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Count file types
    file_types = {'raw': 0, 'cine': 0, 'cine_whole': 0, 'lge': 0}
    for slice_files in slices.values():
        for prefix in slice_files:
            if prefix in file_types:
                file_types[prefix] += 1
    
    analysis_results['file_types'] = file_types
    
    # Calculate overall statistics for each file type
    stats = {
        'raw': {'arrays': [], 'shapes': []},
        'cine': {'arrays': [], 'shapes': []},
        'cine_whole': {'arrays': [], 'shapes': []},
        'lge': {'arrays': [], 'shapes': []}
    }
    
    # Analyze a subset of slices
    slice_nums = sorted(list(slices.keys()))
    for slice_num in slice_nums[:slice_limit]:
        slice_files = slices[slice_num]
        
        slice_analysis = {
            'slice_num': slice_num,
            'files': {}
        }
        
        # Create a visualization with subplots for this slice
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"NPY File Analysis - Patient {patient_id} - Slice {slice_num}", fontsize=16)
        
        subplot_positions = {
            'raw': (0, 0),
            'cine': (0, 1),
            'cine_whole': (1, 0),
            'lge': (1, 1)
        }
        
        # Load and analyze each file type for this slice
        for prefix in ['raw', 'cine', 'cine_whole', 'lge']:
            if prefix in slice_files:
                file_path = slice_files[prefix]
                try:
                    # Load and analyze the array
                    arr = np.load(file_path)
                    stats[prefix]['arrays'].append(arr)
                    stats[prefix]['shapes'].append(arr.shape)
                    
                    arr_analysis = {
                        'file': os.path.basename(file_path),
                        'shape': arr.shape,
                        'dtype': str(arr.dtype),
                        'min': float(np.min(arr)),
                        'max': float(np.max(arr)),
                        'mean': float(np.mean(arr)),
                        'std': float(np.std(arr)),
                        'has_nan': bool(np.isnan(arr).any()),
                        'zeros_percentage': float(np.sum(arr == 0) / arr.size * 100)
                    }
                    
                    slice_analysis['files'][prefix] = arr_analysis
                    
                    # Plot the array
                    row, col = subplot_positions[prefix]
                    if prefix == 'lge':
                        # For LGE segmentation, use a different colormap
                        im = axs[row, col].imshow(arr, cmap='jet')
                        plt.colorbar(im, ax=axs[row, col], fraction=0.046, pad=0.04)
                    else:
                        axs[row, col].imshow(arr, cmap='gray')
                    
                    axs[row, col].set_title(f"{prefix} - {arr.shape}")
                    axs[row, col].axis('off')
                    
                except Exception as e:
                    print(f"  Error analyzing {file_path}: {e}")
                    slice_analysis['files'][prefix] = {'error': str(e)}
                    
                    # Display error in subplot
                    row, col = subplot_positions[prefix]
                    axs[row, col].text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                                      horizontalalignment='center', verticalalignment='center',
                                      transform=axs[row, col].transAxes, fontsize=10)
                    axs[row, col].set_title(f"{prefix} - Error")
            else:
                # Mark missing file in subplot
                row, col = subplot_positions[prefix]
                axs[row, col].text(0.5, 0.5, "File not found", 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=axs[row, col].transAxes)
                axs[row, col].set_title(f"{prefix} - Missing")
        
        # Save the visualization
        vis_path = os.path.join(vis_dir, f"slice_{slice_num}.png")
        plt.tight_layout()
        plt.savefig(vis_path, dpi=150)
        plt.close()
        
        slice_analysis['visualization'] = vis_path
        analysis_results['slice_analyses'].append(slice_analysis)
    
    # Calculate overall statistics for each file type
    for prefix in stats:
        if stats[prefix]['arrays']:
            arrays = stats[prefix]['arrays']
            shapes = stats[prefix]['shapes']
            
            analysis_results[f'{prefix}_stats'] = {
                'count': len(arrays),
                'shapes': shapes,
                'common_shape': max(set(map(str, shapes)), key=shapes.count),
                'min': float(np.min([np.min(arr) for arr in arrays])),
                'max': float(np.max([np.max(arr) for arr in arrays])),
                'mean_of_means': float(np.mean([np.mean(arr) for arr in arrays])),
                'mean_of_stds': float(np.mean([np.std(arr) for arr in arrays]))
            }
    
    return analysis_results


def analyze_transformations(mat_info, npy_info, report_dir):
    """
    Document the transformations from .mat files to .npy files
    
    Args:
        mat_info: Dictionary with mat file analysis data
        npy_info: Dictionary with npy file analysis data
        report_dir: Directory to save visualizations
    
    Returns:
        Dictionary with transformation analysis
    """
    print("Analyzing transformations between MAT and NPY files")
    
    transformations = {
        'mat_to_npy_mapping': {
            'raw': 'Myocardium-masked image (myo_seg * raw_image) -> Standardized, cropped, resized',
            'cine': 'Raw image -> Cropped to myocardium bounding box, standardized, resized',
            'cine_whole': 'Raw image -> Standardized to 0-1 range',
            'lge': 'Enhancement data -> Cropped to myocardium bounding box, standardized, resized'
        },
        'operations': {
            'standardization': 'Images are standardized to 0-1 range using (img-min)/(max-min)',
            'cropping': 'Images are cropped to the myocardium bounding box using PIL getbbox()',
            'resizing': 'Images are resized to a standard size (default 128x128) using scipy.ndimage.zoom',
            'masking': 'For raw_*.npy, the raw image is masked with the myocardium segmentation'
        },
        'visualizations': []
    }
    
    # Create a visualization showing the transformation pipeline
    try:
        # Iterate over available patients and slices to find examples where we have both input and output
        for patient_analysis in npy_info:
            patient_id = patient_analysis['patient_id']
            
            # Find corresponding mat file analysis
            matching_mat = None
            for mat_analysis in mat_info:
                if patient_id in mat_analysis['filename']:
                    matching_mat = mat_analysis
                    break
            
            if not matching_mat:
                continue
                
            # Create visual examples for a few slices
            for slice_analysis in patient_analysis['slice_analyses']:
                slice_num = slice_analysis['slice_num']
                
                # Check if we have both input and output visualizations
                mat_vis = None
                for vis in matching_mat.get('visualizations', []):
                    if vis['slice'] == slice_num:
                        mat_vis = vis['path']
                        break
                
                if not mat_vis or 'visualization' not in slice_analysis:
                    continue
                
                # Create a transformation visualization
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
                fig.suptitle(f"MAT to NPY Transformation - Patient {patient_id} - Slice {slice_num}", fontsize=16)
                
                # Load and display the visualizations
                mat_img = plt.imread(mat_vis)
                npy_img = plt.imread(slice_analysis['visualization'])
                
                axs[0].imshow(mat_img)
                axs[0].set_title("Input (.mat file)")
                axs[0].axis('off')
                
                axs[1].imshow(npy_img)
                axs[1].set_title("Output (.npy files)")
                axs[1].axis('off')
                
                # Save the visualization
                vis_dir = os.path.join(report_dir, 'transformations')
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{patient_id}_slice_{slice_num}_transformation.png")
                plt.tight_layout()
                plt.savefig(vis_path, dpi=150)
                plt.close()
                
                transformations['visualizations'].append({
                    'patient_id': patient_id,
                    'slice_num': slice_num,
                    'path': vis_path
                })
                
                # Only create a few examples
                if len(transformations['visualizations']) >= 3:
                    break
            
            if len(transformations['visualizations']) >= 3:
                break
    
    except Exception as e:
        print(f"Error creating transformation visualizations: {e}")
        traceback.print_exc()
    
    return transformations


def generate_statistical_analysis(npy_info, report_dir):
    """
    Generate statistical analysis of the processed .npy files
    
    Args:
        npy_info: List of dictionaries with npy file analysis data
        report_dir: Directory to save visualizations
    
    Returns:
        Dictionary with statistical analysis
    """
    print("Generating statistical analysis")
    
    # Collect statistics across all patients
    stats = {
        'raw': {'min': [], 'max': [], 'mean': [], 'std': [], 'zeros': []},
        'cine': {'min': [], 'max': [], 'mean': [], 'std': [], 'zeros': []},
        'cine_whole': {'min': [], 'max': [], 'mean': [], 'std': [], 'zeros': []},
        'lge': {'min': [], 'max': [], 'mean': [], 'std': [], 'zeros': []}
    }
    
    # Collect shape information
    shapes = {
        'raw': [],
        'cine': [],
        'cine_whole': [],
        'lge': []
    }
    
    # Collect values from all patients and slices
    for patient_analysis in npy_info:
        for slice_analysis in patient_analysis['slice_analyses']:
            for prefix in ['raw', 'cine', 'cine_whole', 'lge']:
                if prefix in slice_analysis['files'] and 'error' not in slice_analysis['files'][prefix]:
                    file_stats = slice_analysis['files'][prefix]
                    stats[prefix]['min'].append(file_stats['min'])
                    stats[prefix]['max'].append(file_stats['max'])
                    stats[prefix]['mean'].append(file_stats['mean'])
                    stats[prefix]['std'].append(file_stats['std'])
                    stats[prefix]['zeros'].append(file_stats['zeros_percentage'])
                    shapes[prefix].append(str(file_stats['shape']))
    
    # Create summary statistics
    summary = {}
    for prefix in stats:
        if stats[prefix]['min']:
            summary[prefix] = {
                'count': len(stats[prefix]['min']),
                'min': {
                    'min': np.min(stats[prefix]['min']),
                    'max': np.max(stats[prefix]['min']),
                    'mean': np.mean(stats[prefix]['min'])
                },
                'max': {
                    'min': np.min(stats[prefix]['max']),
                    'max': np.max(stats[prefix]['max']),
                    'mean': np.mean(stats[prefix]['max'])
                },
                'mean': {
                    'min': np.min(stats[prefix]['mean']),
                    'max': np.max(stats[prefix]['mean']),
                    'mean': np.mean(stats[prefix]['mean'])
                },
                'std': {
                    'min': np.min(stats[prefix]['std']),
                    'max': np.max(stats[prefix]['std']),
                    'mean': np.mean(stats[prefix]['std'])
                },
                'zeros_percentage': {
                    'min': np.min(stats[prefix]['zeros']),
                    'max': np.max(stats[prefix]['zeros']),
                    'mean': np.mean(stats[prefix]['zeros'])
                },
                'common_shape': max(set(shapes[prefix]), key=shapes[prefix].count) if shapes[prefix] else 'Unknown'
            }
    
    # Create visualizations
    vis_dir = os.path.join(report_dir, 'statistics')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create statistical plots for each file type
    for prefix in ['raw', 'cine', 'cine_whole', 'lge']:
        if prefix in summary:
            try:
                # Create distribution plots
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"Statistical Analysis - {prefix}", fontsize=16)
                
                # Min values distribution
                sns.histplot(stats[prefix]['min'], ax=axs[0, 0], kde=True)
                axs[0, 0].set_title(f"{prefix} - Min Values Distribution")
                axs[0, 0].set_xlabel("Min Value")
                
                # Max values distribution
                sns.histplot(stats[prefix]['max'], ax=axs[0, 1], kde=True)
                axs[0, 1].set_title(f"{prefix} - Max Values Distribution")
                axs[0, 1].set_xlabel("Max Value")
                
                # Mean values distribution
                sns.histplot(stats[prefix]['mean'], ax=axs[1, 0], kde=True)
                axs[1, 0].set_title(f"{prefix} - Mean Values Distribution")
                axs[1, 0].set_xlabel("Mean Value")
                
                # Std values distribution
                sns.histplot(stats[prefix]['std'], ax=axs[1, 1], kde=True)
                axs[1, 1].set_title(f"{prefix} - Standard Deviation Distribution")
                axs[1, 1].set_xlabel("Std Value")
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{prefix}_statistics.png"), dpi=150)
                plt.close()
                
                # Add visualization path to summary
                summary[prefix]['visualization'] = os.path.join(vis_dir, f"{prefix}_statistics.png")
                
                # Create boxplot of all statistics
                plt.figure(figsize=(10, 6))
                df = pd.DataFrame({
                    'Min': stats[prefix]['min'],
                    'Max': stats[prefix]['max'],
                    'Mean': stats[prefix]['mean'],
                    'Std': stats[prefix]['std']
                })
                
                sns.boxplot(data=df)
                plt.title(f"{prefix} - Statistical Measures")
                plt.ylabel("Value")
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"{prefix}_boxplot.png"), dpi=150)
                plt.close()
                
                # Add boxplot visualization to summary
                summary[prefix]['boxplot_visualization'] = os.path.join(vis_dir, f"{prefix}_boxplot.png")
                
                # Create zeros percentage plot
                plt.figure(figsize=(8, 6))
                sns.histplot(stats)
