from dataset import organize_patient_data
import os




def create_dictionary_visualization(data_dict, image_types=None, figsize=(12, 6), dpi=100):

    """
    Create an interactive visualization of cardiac MRI data from dictionary structure.
    
    Args:
        data_dict: Dictionary where keys are patient IDs and values are dictionaries containing
                  numpy arrays for different image types ('cine', 'cine_whole', 'lge', 'raw')
        image_types: List of image types to display (default: all available types)
        figsize: Figure size as (width, height) in inches
        dpi: Figure resolution (dots per inch)
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    import numpy as np
    
    # Validate input data
    if not data_dict:
        print("Error: No data provided")
        return
    
    # Determine available image types if not specified
    if image_types is None:
        # Find all unique image types across patients
        all_types = set()
        for patient_data in data_dict.values():
            all_types.update(key for key, value in patient_data.items() if value is not None)
        image_types = sorted(list(all_types))
    
    # Setup the figure and axes for visualization
    num_cols = len(image_types)
    fig, axes = plt.subplots(1, num_cols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(bottom=0.25)  # Make room for sliders
    
    # If there's only one image type, ensure axes is a list
    if num_cols == 1:
        axes = [axes]
    
    # Set up patient selection and navigation controls
    patient_ids = sorted(list(data_dict.keys()))
    if not patient_ids:
        print("Error: No patients found in data")
        plt.close(fig)
        return
    
    current_patient = patient_ids[0]
    
    # Function to get max number of slices for current patient
    def get_max_slices(patient_id):
        max_slices = 0
        for img_type, img_data in data_dict[patient_id].items():
            if img_data is not None:
                if isinstance(img_data, np.ndarray):
                    max_slices = max(max_slices, img_data.shape[0])
                elif isinstance(img_data, list):
                    max_slices = max(max_slices, len(img_data))
        return max_slices - 1  # 0-based indexing
    
    # Initialize current slice
    max_slice_idx = get_max_slices(current_patient)
    current_slice = 0 if max_slice_idx >= 0 else 0
    
    def update_display(patient_id, slice_idx):
        """Update the display with the specified patient and slice."""
        # Clear existing images
        for ax in axes:
            ax.clear()
        
        # Access patient data
        patient_data = data_dict[patient_id]
        
        # Update each axis with new image
        for i, img_type in enumerate(image_types):
            if img_type in patient_data and patient_data[img_type] is not None:
                img_data = patient_data[img_type]
                
                # Handle different data types (tensor vs list of arrays)
                if isinstance(img_data, np.ndarray):
                    # Check if slice index is valid
                    if slice_idx < img_data.shape[0]:
                        # Extract the 2D slice from the 3D tensor
                        slice_data = img_data[slice_idx]
                        axes[i].imshow(slice_data, cmap='gray')
                    else:
                        axes[i].text(0.5, 0.5, f"Slice {slice_idx} not available", 
                                     horizontalalignment='center',
                                     verticalalignment='center')
                elif isinstance(img_data, list):
                    # For list of arrays (possibly cine_whole with varying dimensions)
                    if slice_idx < len(img_data):
                        slice_data = img_data[slice_idx]
                        axes[i].imshow(slice_data, cmap='gray')
                    else:
                        axes[i].text(0.5, 0.5, f"Slice {slice_idx} not available", 
                                     horizontalalignment='center',
                                     verticalalignment='center')
                
                # Set title based on image type
                if img_type == 'lge': 
                    axes[i].set_title(f"{img_type} (ground truth) (Slice {slice_idx})")
                else:
                    axes[i].set_title(f"{img_type} (Slice {slice_idx})")
                
                axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"No {img_type} data", 
                             horizontalalignment='center',
                             verticalalignment='center')
        
        fig.suptitle(f"Patient: {patient_id}", fontsize=16)
        plt.draw()
    
    # Create slider for slice selection
    ax_slice = plt.axes([0.25, 0.1, 0.65, 0.03])
    slice_slider = Slider(
        ax=ax_slice,
        label='Slice',
        valmin=0,
        valmax=max(1, max_slice_idx),
        valinit=current_slice,
        valstep=1
    )
    
    def update_slice(val):
        nonlocal current_slice
        current_slice = int(val)
        update_display(current_patient, current_slice)
    
    slice_slider.on_changed(update_slice)
    
    # Create patient selection slider
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
        nonlocal current_patient, current_slice
        patient_idx = int(val)
        current_patient = patient_ids[patient_idx]
        
        # Update slice slider for new patient
        max_slice_idx = get_max_slices(current_patient)
        slice_slider.valmax = max(1, max_slice_idx)
        
        # Reset to first slice for new patient
        current_slice = 0
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
    filepath = os.environ.get('PROCESSED_DATA_DIR')
    patient_data = organize_patient_data(filepath) #come up with better name var #TODO


    create_dictionary_visualization(patient_data)

if __name__ == '__main__':
    main()
