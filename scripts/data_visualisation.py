import ee
import geemap
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import ipywidgets as widgets
sns.set_theme()

#--------------------------------------------Map visualisation-------------------------------------------------#
def plot_lulc(Map, selection, results_per_area_and_year, dw_vis, sub_area=None, years=None):
    """
    Adds LULC data for a specified sub-area and year(s) to the map.
    Users can specify a single year or a list of years.
    """
    if sub_area is None:
        sub_area = "Dissolved"  # Default sub-area
    if years is None:
        years = [max(selection["Years"])]  # Default to latest available year
    elif isinstance(years, int):
        years = [years]  # Convert single year input to list

    for year in years:
        if sub_area in results_per_area_and_year and year in results_per_area_and_year[sub_area]:
            Map.addLayer(results_per_area_and_year[sub_area][year], dw_vis, f"LULC {sub_area} {year}", False)
            print(f"✅ Added LULC layer for {sub_area}, {year} to the map.")
        else:
            print(f"⚠️ No data available for {sub_area} in {year}. Modify the function to select a different sub-area or year.")

#--------------------------------------------Line and pie charts-------------------------------------------------#
def load_csv(file_path):
    """
    Loads a CSV file into a DataFrame, handling missing files gracefully.
    """
    if not file_path.exists():
        print(f"Warning: File not found - {file_path}")
        return None
    return pd.read_csv(file_path, index_col="Year")

def plot_line_chart(ax, df, valid_labels, linestyle, palette, class_labels, label_prefix=""):
    """
    Plots line charts for the given DataFrame and valid class labels.
    """
    for column in valid_labels:
        if column in df.columns:
            color = palette[class_labels.index(column)]
            ax.plot(df.index, df[column], marker="o", linestyle=linestyle, label=f"{label_prefix}{column}", color=color)

def add_pie_charts(fig, ax, dissolved_df, valid_labels, palette, class_labels):
    """
    Adds pie charts to the plot for each year.
    """
    relative_pie_size = 0.15
    xticks = ax.get_xticks()
    xticks = xticks[(xticks >= dissolved_df.index.min()) & (xticks <= dissolved_df.index.max())]
    xticks = xticks[:len(dissolved_df.index)]
    
    for i, year in enumerate(dissolved_df.index):
        pie_x = xticks[i]
        trans = ax.transData + fig.transFigure.inverted()
        pie_x_fig, _ = trans.transform((pie_x, 0))
        pie_rect = [pie_x_fig - relative_pie_size / 2, 0.1, relative_pie_size, relative_pie_size]
        
        pie_data = dissolved_df.loc[year, valid_labels]
        pie_colors = [palette[class_labels.index(col)] for col in pie_data.index]
        
        ax_pie = fig.add_axes(pie_rect, frameon=False)
        ax_pie.pie(pie_data, colors=pie_colors, startangle=90)

def plot_lulc_comparison(selection, class_labels, palette):
    """
    Loads LULC CSV data and generates line and pie charts comparing park and dissolved areas.
    """
    park_dir = Path("..") / "data" / "DW_datasets" / selection["Park"]
    dissolved_file = park_dir / f"{selection['Park']}_Dissolved_LULC_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"
    parks_file = park_dir / f"{selection['Park']}_Parks_LULC_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"

    dissolved_df = load_csv(dissolved_file)
    parks_df = load_csv(parks_file)

    if dissolved_df is None or parks_df is None:
        print("Error: One or more required files are missing. Please check the dataset directory.")
        return
    
    dissolved_df_pie = dissolved_df.fillna(0)  # Fill missing values for pie chart
    parks_df_pie = parks_df.fillna(0)
    
    valid_class_labels = [label for label in class_labels if label in dissolved_df.columns or label in parks_df.columns]

    fig, ax = plt.subplots(figsize=(15, 10))

    plot_line_chart(ax, dissolved_df, valid_class_labels, linestyle="-", palette=palette, class_labels=class_labels)
    plot_line_chart(ax, parks_df, valid_class_labels, linestyle="--", palette=palette, class_labels=class_labels, label_prefix="Park ")

    ax.set_xlabel("Year")
    ax.set_ylabel("Pixel count")
    ax.set_yscale("log")
    ax.set_title(f"LULC for {selection['Park']}'s park vs. dissolved buffer zone from {selection['Starting Year']} to {selection['Ending Year']}", fontsize=18)
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticks(dissolved_df.index)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, which="both", linestyle="--")

    add_pie_charts(fig, ax, dissolved_df_pie, valid_class_labels, palette, class_labels)

    plt.subplots_adjust(bottom=0.3)
    plt.show()

#--------------------------------------------Normalized differences-------------------------------------------------#
def calculate_normalized_diff(df):
    """
    Calculates the difference between consecutive years and normalizes them
    to the maximum absolute value for each column.
    """
    diff_df = df.diff().fillna(0)  # Compute differences, filling NaN with 0
    max_values = diff_df.abs().max()  # Maximum absolute values per column
    
    # Avoid division by zero
    max_values[max_values == 0] = 1  # Set max to 1 where values are all zero to prevent NaN
    
    normalized_diff = diff_df / max_values  # Normalize
    return normalized_diff

def plot_normalized_differences(selection, class_labels, palette):
    """
    Plots normalized yearly differences between dissolved and park areas for LULC.
    """
    park_dir = Path("..") / "data" / "DW_datasets" / selection["Park"]
    dissolved_file = park_dir / f"{selection['Park']}_Dissolved_LULC_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"
    parks_file = park_dir / f"{selection['Park']}_Parks_LULC_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"

    dissolved_df = load_csv(dissolved_file)
    parks_df = load_csv(parks_file)
    dissolved_diff_normalized = calculate_normalized_diff(dissolved_df)
    parks_diff_normalized = calculate_normalized_diff(parks_df)

    available_classes = [label for label in class_labels if label in dissolved_diff_normalized.columns and label in parks_diff_normalized.columns]
    num_classes = len(available_classes)

    ncols = 3
    nrows = (num_classes + ncols - 1) // ncols  # Ensure all classes fit

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 12))
    axes = axes.flatten()

    for i, column in enumerate(available_classes):
        ax = axes[i]
        width = 0.35
        x = np.arange(len(dissolved_diff_normalized.index))
        color = palette[class_labels.index(column)]

        ax.bar(x - width/2, dissolved_diff_normalized[column], width, label=f'Dissolved {column}', color=color)
        ax.bar(x + width/2, parks_diff_normalized[column], width, label=f'Park {column}', 
               color='none', edgecolor=color, linestyle='--', hatch='//')
        
        ax.set_title(column)
        ax.set_xticks(x)
        ax.set_xticklabels(dissolved_diff_normalized.index, rotation=45)
        ax.set_ylim(-1.1, 1.1)
        ax.legend()
        ax.grid(True, which="both", linestyle="--")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(f"Normalized Yearly Difference in LULC for {selection['Park']}'s Park vs. Dissolved Buffer Zone "
                 f"from {selection['Starting Year']} to {selection['Ending Year']}", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

#--------------------------------------------Sankey diagrams-------------------------------------------------------#
def load_dissolved_change_data(selection):
    """
    Loads the dissolved LULC change data from the CSV file.
    """
    park_dir = Path("..") / "data" / "DW_datasets" / selection["Park"]
    change_file = park_dir / f"{selection['Park']}_Dissolved_LULC_change_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"
    
    if not change_file.exists():
        raise FileNotFoundError(f"Error: File not found - {change_file}")
    
    return pd.read_csv(change_file, index_col="Change")

def process_sankey_data(dissolved_change_df, selection, class_labels, palette):
    """
    Processes LULC change data to generate Sankey diagram inputs.
    """
    sources = []
    targets = []
    values = []
    
    num_labels = len(class_labels)  # Number of LULC classes

    for col in dissolved_change_df.columns:
        try:
            from_year, to_year = map(int, col.split("_to_"))  # Extract years from column name
        except ValueError:
            print(f"Skipping column {col}: Invalid format.")
            continue

        for index, value in dissolved_change_df[col].items():
            if value > 0:  # Only create a link if there's a non-zero value
                try:
                    from_class, to_class = index.split("_to_")  # Extract LULC classes
                except ValueError:
                    print(f"Skipping row {index}: Invalid format.")
                    continue

                if from_class in class_labels and to_class in class_labels:
                    source_index = (from_year - selection["Starting Year"]) * num_labels + class_labels.index(from_class)
                    target_index = (to_year - selection["Starting Year"]) * num_labels + class_labels.index(to_class)

                    sources.append(source_index)
                    targets.append(target_index)
                    values.append(value)

    node_labels = [f"{label}" for year in range(selection["Starting Year"], selection["Ending Year"] + 1) for label in class_labels]
    node_colors = palette * (selection["Ending Year"] - selection["Starting Year"] + 1)

    return sources, targets, values, node_labels, node_colors

def plot_sankey_diagram(sources, targets, values, node_labels, node_colors, selection):
    """
    Generates a Sankey diagram for visualizing LULC change over time.
    """
    if not (len(sources) == len(targets) == len(values)):
        raise ValueError("Mismatch in the length of sources, targets, and values.")

    if len(node_labels) != len(node_colors):
        raise ValueError("Mismatch between the number of node labels and node colors.")

    # Create Sankey Diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    # Calculate proportional positions for year annotations along the x-axis
    starting_year = selection["Starting Year"]
    ending_year = selection["Ending Year"]
    
    year_positions = {
        year: (year - starting_year) / (ending_year - starting_year) 
        for year in range(starting_year, ending_year + 1)
    }

    # Update layout with title and year annotations
    fig.update_layout(
        title=dict(
            text=f"LULC Change for {selection['Park']}'s Dissolved Buffer Zone from {starting_year} to {ending_year}",
            font=dict(size=20)
        ),
        font=dict(size=10),
        annotations=[
            dict(
                showarrow=False,
                text=str(year),
                xref="paper",
                yref="paper",
                x=year_positions[year],
                y=-0.1,
                align="center",
                font=dict(size=10)
            ) for year in range(starting_year, ending_year + 1)
        ],
        height=600
    )

    # Show the Sankey diagram
    fig.show()

#-----------------------------------------------------LULC change hotspot map-----------------------------------------------------#
#--------------------------------Setup of selction tools--------------------------------#
def setup_pre_post_year_dropdown(significant_intervals):
    """
    Creates a dropdown widget for selecting the time interval of interest.
    """
    year_options = [interval.replace("_to_", "-") for interval in significant_intervals] + ['Choose here...']
    
    dropdown = widgets.Dropdown(
        options=year_options,
        value='Choose here...',
        description='Year Interval:',
        disabled=False
    )
    return dropdown

def on_pre_post_year_change(change):
    """
    Updates the selected year interval when the dropdown value changes.
    """
    print(f"Pre- and Post-year of investigation updated to: {change['new']}")

def setup_lulcc_change_selection(selection, sub_area):
    """
    Reads LULC change transitions from the relevant CSV file and creates a multi-select widget.

    :param selection: A dictionary containing 'Park', 'Sub Areas', 'Starting Year', and 'Ending Year'.
    :param sub_area: The selected sub-area for filtering.
    :return: A tuple containing the interactive selection widget and a dictionary mapping transition keys to labels.
    """
    output_base_path = Path("..") / "data" / "DW_datasets" / selection["Park"]
    csv_file_path = output_base_path / f"{selection['Park']}_{sub_area}_LULC_change_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"

    # Read CSV and extract transition labels (column names, excluding 'Change')
    df = pd.read_csv(csv_file_path, skiprows=1, header=None)

    # Extract transition labels from the first column (excluding the first row)
    transition_labels = df.iloc[:, 0].tolist()  # First column, all rows

    # Create a mapping: numeric index -> label
    transition_label_map = {str(i): label for i, label in enumerate(transition_labels, start=1)}

    # Create the selection widget
    select_widget = widgets.SelectMultiple(
        options=transition_labels,
        value=[],
        description='Active Changes:',
        disabled=False,
        layout=widgets.Layout(width='350px', height='300px')
    )
    
    return select_widget, transition_label_map

#--------------------------------Generate the map based on selection--------------------------------#
def generate_filtered_lulcc_map(results_per_area_and_year_pairs, sub_area_dropdown, pre_post_year_dropdown, lulcc_select_widget, transition_label_map):
    """
    Generates an interactive map visualizing filtered LULCC transitions.

    Parameters:
    - results_per_area_and_year_pairs (dict): Dictionary containing area and year-specific results.
    - sub_area_dropdown (ipywidgets.Dropdown): Widget for selecting the sub-area.
    - pre_post_year_dropdown (ipywidgets.Dropdown): Widget for selecting the time interval.
    - lulcc_select_widget (ipywidgets.SelectMultiple): Widget for selecting active LULCC changes.
    - transition_label_map (dict): Mapping of transition keys to human-readable labels.

    Returns:
    - Displays an interactive geemap.Map with the filtered transitions.
    """
    # Retrieve selected values
    selected_sub_area = sub_area_dropdown.value
    selected_year_range = pre_post_year_dropdown.value
    selected_labels = lulcc_select_widget.value  # This returns a tuple of selected label strings

    # Ensure valid selection of time interval
    if selected_year_range == "Choose here...":
        print("⚠️ Error: No year range selected. Please choose a valid year range.")
        return
    
    # Ensure valid sub-area selection
    if selected_sub_area not in results_per_area_and_year_pairs:
        print(f"⚠️ Error: No data found for sub-area '{selected_sub_area}'. Please select a valid area.")
        return

    # Ensure valid year selection within the sub-area
    if selected_year_range not in results_per_area_and_year_pairs[selected_sub_area]:
        print(f"⚠️ Error: No data found for year range '{selected_year_range}' in sub-area '{selected_sub_area}'.")
        return
    
    # Map selected labels back to their corresponding transition keys
    selected_keys = [key for key, value in transition_label_map.items() if value in selected_labels]

    # Convert selected keys to integers, handling cases where no selections are made
    try:
        transitions_of_interest = list(map(int, selected_keys)) if selected_keys else []
    except ValueError as e:
        print(f"⚠️ Warning: Unable to convert selected keys to integers: {e}")
        transitions_of_interest = []

    # Debugging output
    print(f"📅 Selected Year Range: {selected_year_range}")
    print(f"🔄 Selected Transitions (Keys): {transitions_of_interest}")

    if not transitions_of_interest:
        print("⚠️ Warning: No transitions selected. The map may be empty.")

    # Define 'combined' image based on selected sub-area and year
    combined = results_per_area_and_year_pairs[selected_sub_area][selected_year_range]

    # Start with a condition that's always False
    mask = ee.Image(0)

    # Dynamically update the mask based on selected transitions
    for transition in transitions_of_interest:
        mask = mask.Or(combined.eq(transition))

    # Apply the mask to filter selected transitions
    filtered_transitions = combined.updateMask(mask)

    # Define visualization parameters
    vis_params = {
        'min': 0,
        'max': max(transitions_of_interest) if transitions_of_interest else 1,  # Avoid max() error
        'palette': ['red'] * len(transitions_of_interest) if transitions_of_interest else ['red']
    }

    # Initialize the geemap Map
    Map = geemap.Map(center=[-30.5595, 22.9375], zoom=5.5, basemap='Esri.WorldImagery')

    # Add the filtered transitions layer to the map
    Map.addLayer(filtered_transitions, vis_params, 'Filtered Transitions')

    # Display the map
    display(Map)