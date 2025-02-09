# utils.py - Acts as the bridge between Jupyter Notebook and the modules
#---------------------------------Setting parameters (config.py) ------------------------------------#
# Import functions from config.py
from config import initialize_static_parameters, calculate_dependent_parameters, setup_dropdown_widgets

# Expose functions for direct use in the notebook
__all__ = [
    "initialize_static_parameters",
    "calculate_dependent_parameters",
    "setup_dropdown_widgets"
]

#--------------------------------Load the data (data_loader.py)--------------------------------------#
# Import functions from data_loader.py
from data_loader import load_dw_results, load_parks_and_buffers, generate_fishnet, process_window_geometries

# Expose functions for direct use in the notebook
__all__ = [
    "initialize_static_parameters",
    "calculate_dependent_parameters",
    "setup_dropdown_widgets",
    "load_dw_results",
    "load_parks_and_buffers",
    "generate_fishnet",
    "process_window_geometries"
]

#-------------------------------Data Collection (data_collection.py)---------------------------------#
# Import functions from data_collection.py
from data_collection import save_lulc_data, compute_lulc_changes, process_lulc_transitions

__all__.extend([
    "save_lulc_data",
    "compute_lulc_changes",
    "process_lulc_transitions"
])

#--------------------------------Data visualisation (data_visualisation.py) ------------------------------#
from data_visualisation import (plot_lulc, load_csv, plot_line_chart, add_pie_charts, plot_lulc_comparison, 
calculate_normalized_diff, plot_normalized_differences, load_dissolved_change_data, process_sankey_data, plot_sankey_diagram)

__all__.extend([
    "plot_lulc",
    "load_csv",
    "plot_line_chart",
    "add_pie_charts",
    "plot_lulc_comparison",
    "calculate_normalized_diff",
    "plot_normalized_differences",
    "load_dissolved_change_data",
    "process_sankey_data",
    "plot_sankey_diagram"
])

#-------------------------------LULC change intensity analysis (lulc_change_intensity.py)--------------------------------------#
from lulc_change_intensity import (
    setup_sub_area_dropdown,
    load_sub_area_data,
    compute_transition_matrices,
    compute_time_intensity,
    plot_time_intensity_analysis,
    calculate_category_intensities,
    plot_category_intensity_analysis,
    calculate_category_transition_intensities,
    plot_transition_heatmap,
    calculate_category_transition_intensities_v2,
    plot_transition_heatmap_v2
)

__all__.extend([
    "setup_sub_area_dropdown",
    "load_sub_area_data",
    "compute_transition_matrices",
    "compute_time_intensity",
    "plot_time_intensity_analysis",
    "calculate_category_intensities",
    "plot_category_intensity_analysis",
    "calculate_category_transition_intensities",
    "plot_transition_heatmap",
    "calculate_category_transition_intensities_v2",
    "plot_transition_heatmap_v2"
])

#--------------------------------Data visualisation of LULC change hotspots (data_visualisation.py)------------------------------#
from data_visualisation import setup_pre_post_year_dropdown, on_pre_post_year_change, setup_lulcc_change_selection, generate_filtered_lulcc_map

__all__.extend([
    "setup_pre_post_year_dropdown",
    "on_pre_post_year_change",
    "setup_lulcc_change_selection",
    "generate_filtered_lulcc_map"
])

#---------------------------------End of script------------------------------------#