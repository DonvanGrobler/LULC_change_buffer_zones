# config.py - Handles static parameters and dropdown widgets

from pathlib import Path
import json
from datetime import datetime
import ipywidgets as widgets

def initialize_static_parameters():
    """
    Initialize and return static parameters that do not change dynamically.

    Returns:
        dw_vis (dict): Visualization properties for the layer.
        class_labels (list): Class labels for visualization.
        base_path (Path): Base path for data from configuration file.
        potential_sub_areas (list): List of potential sub-areas.
        palette (list): List of colors corresponding to class labels.
    """
    # Set visualization properties
    dw_vis = {"min": 0, "max": 8, "palette": [
        "#419BDF", "#397D49", "#88B053", "#7A87C6", "#E49635", "#DFC35A", "#C4281B", "#A59B8F", "#B39FE1"
    ]}
    palette = dw_vis["palette"]

    # Define class labels
    class_labels = ['water', 'trees', 'grass', 'flooded_vegetation', 'crops',
                    'shrub_and_scrub', 'built', 'bare_soil', 'snow_and_ice']

    # Load configuration settings safely
    config_path = Path("..") / "config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        base_path = Path("..") / config["base_path"]
    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ Warning: Missing or invalid config.json! Using default base path.")
        base_path = Path("..") / "default_data_path"

    # Define sub-areas
    potential_sub_areas = ['CPA', 'VPA', 'PNA', 'Parks', 'Dissolved']

    return dw_vis, class_labels, base_path, potential_sub_areas, palette


def calculate_dependent_parameters(selection, base_path, potential_sub_areas):
    """
    Calculate parameters that depend on user selection.

    Args:
        selection (dict): Dictionary containing user-selected values.
        base_path (Path): Base path for data.
        potential_sub_areas (list): List of potential sub-areas.

    Returns:
        Years (list): List of years from starting to ending year.
        sub_areas (list): List of available sub-areas based on shapefile existence.
    """
    # Calculate the list of years
    Years = list(range(selection["Starting Year"], selection["Ending Year"] + 1))

    # Filter sub-areas that have shapefiles
    sub_areas = [
        sub_area for sub_area in potential_sub_areas
        if (base_path / sub_area / f"{selection['Park']}_{sub_area}.shp").exists()
    ]

    return Years, sub_areas


def setup_dropdown_widgets(base_path, potential_sub_areas):
    """
    Set up dropdown widgets for park selection and year range selection.

    Args:
        base_path (Path): Base path for data.
        potential_sub_areas (list): List of potential sub-areas.

    Returns:
        park_dropdown (Dropdown): Dropdown widget for selecting a park.
        year_start_dropdown (Dropdown): Dropdown widget for selecting the start year.
        year_end_dropdown (Dropdown): Dropdown widget for selecting the end year.
        selection (dict): Dictionary storing the current selections and computed values.
    """
    # Define available parks
    parks = [
        'Addo Elephant', 'Agulhas', 'Augrabies Falls', 'Bontebok', 'Camdeboo', 'Garden Route',
        'Golden Gate Highlands', 'Graspan', 'Groenkloof', 'Kalahari Gemsbok', 'Karoo', 'Kruger',
        'Mapungubwe', 'Marakele', 'Mokala', 'Mountain Zebra', 'Namaqua', 'Richtersveld',
        'Table Mountain', 'Tankwa Karoo', 'West Coast'
    ]

    # Get current year
    current_year = datetime.now().year
    years = [str(y) for y in range(2016, current_year)]

    # Dropdown widgets
    park_dropdown = widgets.Dropdown(options=parks, value='Addo Elephant', description='Park:')
    year_start_dropdown = widgets.Dropdown(options=years, value='2016', description='Start Year:')
    year_end_dropdown = widgets.Dropdown(options=years, value='2023', description='End Year:')

    # Initialize selection dictionary
    selection = {
        "Park": park_dropdown.value,
        "Starting Year": int(year_start_dropdown.value),
        "Ending Year": int(year_end_dropdown.value),
        "Years": list(range(int(year_start_dropdown.value), int(year_end_dropdown.value) + 1)),
        "Sub Areas": []
    }

    def update_selection(change, key):
        """Update selection when dropdown changes."""
        selection[key] = int(change.new) if key != "Park" else change.new
        selection["Years"], selection["Sub Areas"] = calculate_dependent_parameters(selection, base_path, potential_sub_areas)

    # Attach event listeners
    park_dropdown.observe(lambda change: update_selection(change, "Park"), names='value')
    year_start_dropdown.observe(lambda change: update_selection(change, "Starting Year"), names='value')
    year_end_dropdown.observe(lambda change: update_selection(change, "Ending Year"), names='value')

    # Set initial values
    selection["Years"], selection["Sub Areas"] = calculate_dependent_parameters(selection, base_path, potential_sub_areas)

    return park_dropdown, year_start_dropdown, year_end_dropdown, selection