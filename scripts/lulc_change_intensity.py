# lulc_change_intensity.py - Handles LULC change intensity analysis

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import ipywidgets as widgets
from pathlib import Path

#--------------------------------------------Setting dropdown for sub-area selection--------------------------------------------#
def setup_sub_area_dropdown(sub_areas):
    """
    Sets up a dropdown widget for selecting a buffer sub-area.
    """
    if not sub_areas:
        raise ValueError("Error: 'sub_areas' list is not defined or empty. Please ensure it's available before running this function.")
    
    sub_area_dropdown = widgets.Dropdown(
        options=sub_areas,
        value='Dissolved',
        description='Buffer sub-area for further investigation:',
        disabled=False
    )
    
    def inv_sub_area_change(change):
        print(f"Buffer sub-area to investigate changed to: {change['new']}")
    
    sub_area_dropdown.observe(inv_sub_area_change, names='value')
    
    # Print guidance for users
    print(
        "💡 TIP: Start with 'Dissolved' and then explore other options as needed.\n\n"
        "Abbreviations:\n"
        "- CPA: Catchment Protected Area\n"
        "- VPA: Viewshed Protected Area\n"
        "- PNA: Priority Natural Areas\n"
        "- Parks: Park boundaries themselves"
    )
    
    return sub_area_dropdown

#-----------------------------------------Prepare data in cross-tabulation format---------------------------------------#
def load_sub_area_data(selection, sub_area):
    """
    Loads LULC transition data for a selected sub-area.

    Parameters:
    - selection (dict): Contains 'Park', 'Starting Year', and 'Ending Year'.
    - sub_area (str): Selected buffer sub-area.

    Returns:
    - df (pd.DataFrame): DataFrame containing LULC transitions.
    """
    park_dir = Path("..") / "data" / "DW_datasets" / selection["Park"]
    file_path = park_dir / f"{selection['Park']}_{sub_area}_LULC_change_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Error: File not found - {file_path}")

    df = pd.read_csv(file_path)

    if "Change" not in df.columns:
        raise KeyError(f"Error: Column 'Change' not found in {file_path}")

    return df

def compute_transition_matrices(df):
    """
    Computes yearly LULC transition matrices.

    Parameters:
    - df (pd.DataFrame): DataFrame containing LULC transitions.

    Returns:
    - transition_matrices (dict): Dictionary of DataFrames, each representing a transition matrix for a year.
    """
    if "Change" in df.columns:
        df[['from', 'to']] = df['Change'].str.split('_to_', expand=True)
    else:
        raise KeyError("Error: 'Change' column is missing in the dataset.")

    transition_matrices = {}
    year_columns = [col for col in df.columns if col not in ["Change", "from", "to"]]

    for year in year_columns:
        matrix = df.pivot(index='from', columns='to', values=year).fillna(0)
        all_classes = sorted(set(matrix.index) | set(matrix.columns))
        matrix = matrix.reindex(index=all_classes, columns=all_classes, fill_value=0)
        matrix = matrix.T
        matrix['Final total'] = matrix.sum(axis=1)
        matrix['Gross gain'] = matrix['Final total'] - [
            matrix.at[state, state] if state in matrix.columns and state in matrix.index else 0 for state in matrix.index
        ]
        matrix = matrix.T
        matrix['Initial total'] = matrix.sum(axis=1)
        matrix['Gross loss'] = matrix['Initial total'] - [
            matrix.at[state, state] if state in matrix.columns and state in matrix.index else 0 for state in matrix.index
        ]
        transition_matrices[year] = matrix

    return transition_matrices

#-------------------------------------------------Time intensity analysis-------------------------------------------------#
def compute_time_intensity(transition_matrices):
    """
    Computes the annual rate of LULCC change over time and determines significant intervals.
    
    Parameters:
    - transition_matrices (dict): Dictionary of transition matrices per year interval.
    
    Returns:
    - time_intervals (list): List of time intervals.
    - annual_rates_of_change (list): List of annual rates of change.
    - uniform_intensity (float): Uniform intensity value.
    - significant_intervals (list): List of significant time intervals.
    """
    time_intervals = []
    annual_rates_of_change = []

    for year_interval, matrix in transition_matrices.items():
        try:
            if 'Final total' not in matrix.index or 'Gross gain' not in matrix.index:
                print(f"⚠️ Warning: 'Final total' or 'Gross gain' is missing in {year_interval}, skipping.")
                continue
            total_area_of_change = matrix.loc['Gross gain'].sum()
            total_area_of_study_region = matrix.loc['Final total'].sum()
            duration_of_interval = 1
            time_intensity = (total_area_of_change / total_area_of_study_region) / duration_of_interval * 100
            time_intervals.append(year_interval)
            annual_rates_of_change.append(time_intensity)
        except KeyError as e:
            print(f"⚠️ Warning: Unexpected missing data in transition matrix for {year_interval}: {e}")
            continue

    try:
        valid_matrices = [matrix for matrix in transition_matrices.values() if 'Final total' in matrix.index]
        if not valid_matrices:
            raise ValueError("No valid transition matrices found with 'Final total' row.")
        total_change_over_all_intervals = sum(matrix.loc['Gross gain'].sum() for matrix in valid_matrices)
        total_study_area = valid_matrices[0].loc['Final total'].sum()
        total_time = len(time_intervals)
        uniform_intensity = (total_change_over_all_intervals / total_study_area) / total_time * 100
    except (KeyError, ValueError) as e:
        print(f"⚠️ Warning: Error while calculating uniform intensity: {e}")
        uniform_intensity = None

    significant_intervals = []
    if uniform_intensity is not None:
        significant_intervals = [year for year, rate in zip(time_intervals, annual_rates_of_change) if rate > uniform_intensity]

    return time_intervals, annual_rates_of_change, uniform_intensity, significant_intervals

def plot_time_intensity_analysis(time_intervals, annual_rates_of_change, uniform_intensity, selection, sub_area):
    """
    Plots the annual rates of LULCC change over time and compares them with uniform intensity.
    """
    if not time_intervals or not annual_rates_of_change or uniform_intensity is None:
        print("⚠️ Error: Missing required data for plotting. Ensure time_intervals, annual_rates_of_change, and uniform_intensity are available.")
        return

    plt.figure(figsize=(10, 5))
    plt.barh(time_intervals, annual_rates_of_change, color='gray', edgecolor='black', label="Annual Rate of Change")
    plt.axvline(x=uniform_intensity, color='red', linestyle='--', label=f'Uniform Intensity: {uniform_intensity:.2f}%')
    y_position = max(len(time_intervals) - 1, 1)
    plt.text(uniform_intensity - 0.08, y_position, 'Slow', verticalalignment='center', horizontalalignment='right', color='red', fontsize=12)
    plt.text(uniform_intensity + 0.08, y_position, 'Fast', verticalalignment='center', horizontalalignment='left', color='red', fontsize=12)
    plt.xlabel('Annual Change Area (percent of map)', fontsize=12)
    plt.ylabel('Time Interval', fontsize=12)
    plt.title(f"LULCC Time Intensity Analysis for {selection['Park']}'s {sub_area} Buffer Zone", fontsize=15)
    plt.legend()
    plt.show()

#------------------------------------------ Categroy intensity analysis ------------------------------------------------------#

def calculate_category_intensities(significant_intervals, transition_matrices, duration_of_interval=1):
    """
    Computes category gain and loss intensities for each significant time interval.

    Parameters:
    - significant_intervals (list): List of years with significant LULCC changes.
    - transition_matrices (dict): Dictionary containing transition matrices for each year.
    - duration_of_interval (int): Number of years in the interval (default=1).

    Returns:
    - category_intensities (dict): Dictionary with per-category loss and gain intensities.
    """
    # Dictionary to store category intensities per year
    category_intensities = {}

    # Loop through each significant year
    for year in significant_intervals:
        # Retrieve the transition matrix for the current year
        matrix = transition_matrices[year]

        # Initialize dictionaries for loss and gain intensities
        loss_intensities = {}
        gain_intensities = {}

        # Loop through the categories in the matrix (excluding summary rows)
        for category in matrix.index[:-2]:  # Excluding 'Final total' and 'Gross gain'
            try:
                # Retrieve necessary values from the matrix
                initial_total = matrix.loc[category, 'Initial total']
                final_total = matrix.T.loc[category, 'Final total']
                gross_loss = matrix.loc[category, 'Gross loss']
                gross_gain = matrix.T.loc[category, 'Gross gain']

                # Calculate loss intensity (only if `initial_total` > 0)
                if initial_total > 0:
                    loss_intensity = (gross_loss / duration_of_interval) / initial_total * 100
                    loss_intensities[category] = loss_intensity

                # Calculate gain intensity (only if `final_total` > 0)
                if final_total > 0:
                    gain_intensity = (gross_gain / duration_of_interval) / final_total * 100
                    gain_intensities[category] = gain_intensity

            except KeyError as e:
                print(f"⚠️ Warning: Missing data for category '{category}' in {year}: {e}")
                continue  # Skip problematic categories

        # Store computed intensities for the current year
        category_intensities[year] = {
            'loss_intensities': loss_intensities,
            'gain_intensities': gain_intensities
        }

    return category_intensities

def plot_category_intensity_analysis(significant_intervals, category_intensities, time_intervals, annual_rates_of_change, selection, sub_area, total_categories=9):
    """
    Plots the LULCC category intensity analysis for significant intervals.

    Parameters:
    - significant_intervals (list): List of significant years.
    - category_intensities (dict): Dictionary containing per-category gain and loss intensities.
    - time_intervals (list): All time intervals.
    - annual_rates_of_change (list): Corresponding annual rates of change.
    - selection (dict): Contains 'Park' name for title.
    - sub_area (str): Selected sub-area for investigation.
    - total_categories (int): Total expected LULC categories in the dataset (default=9).

    Returns:
    - Displays category intensity bar plots for each significant interval.
    """

    for interval in significant_intervals:
        # Ensure the interval exists in category_intensities
        if interval not in category_intensities:
            print(f"⚠️ Warning: No data found for interval {interval}, skipping.")
            continue

        # Get intensity data for the current interval
        data = category_intensities[interval]

        # Extract categories and fill missing ones with 0s
        categories = list(data['loss_intensities'].keys())  # Available categories
        loss_intensities = [data['loss_intensities'].get(cat, 0) for cat in categories]
        gain_intensities = [data['gain_intensities'].get(cat, 0) for cat in categories]

        # Padding missing categories with zeros
        empty_slots = total_categories - len(categories)
        loss_intensities += [0] * empty_slots
        gain_intensities += [0] * empty_slots
        categories += [''] * empty_slots  # Empty strings for missing category labels

        # Find the annual rate of change for this interval
        index = time_intervals.index(interval)
        specific_uniform_intensity = annual_rates_of_change[index]

        # Position of bars on the y-axis
        y_pos = np.arange(total_categories)

        # Create the figure
        plt.figure(figsize=(10, 5))

        # Horizontal bars for losses
        plt.barh(y_pos, loss_intensities, color='tomato', edgecolor='black', height=0.4, label='Loss Intensity')

        # Horizontal bars for gains (slightly offset on the y-axis)
        plt.barh(y_pos + 0.4, gain_intensities, color='mediumseagreen', edgecolor='black', height=0.4, label='Gain Intensity')

        # Draw a dashed line for uniform intensity
        plt.axvline(x=specific_uniform_intensity, color='black', linestyle='--', label=f'Uniform Intensity {specific_uniform_intensity:.2f}%')

        # Add 'Dormant' and 'Active' labels near the uniform intensity line
        plt.text(specific_uniform_intensity - 0.25, total_categories - 0.2, 'Dormant',
                 verticalalignment='center', horizontalalignment='right', color='black', fontsize=10)
        plt.text(specific_uniform_intensity + 0.5, total_categories - 0.2, 'Active',
                 verticalalignment='center', horizontalalignment='left', color='black', fontsize=10)

        # Labels and title
        plt.xlabel('Annual Change Intensity (percent of category)', fontsize=12)
        plt.title(f"LULCC Category Intensity Analysis for {selection['Park']}'s {sub_area} Buffer Zone ({interval})", fontsize=13)
        plt.yticks(y_pos + 0.2, categories)  # Center y-ticks for categories

        plt.legend()
        plt.show()    

#--------------------------------------------------------- Category transition intensity analysis -----------------------------------------------------------#

#----------------------------------From target category intensity analysis-------------------------------------------#
def calculate_category_transition_intensities(significant_intervals, transition_matrices, duration_of_interval=1):
    """
    Computes category transition intensities for each significant time interval.

    Parameters:
    - significant_intervals (list): List of years with significant LULCC changes.
    - transition_matrices (dict): Dictionary containing transition matrices for each year.
    - duration_of_interval (int): Number of years in the interval (default=1).

    Returns:
    - category_transitions (dict): Dictionary with transition intensities for each category.
    """
    category_transitions = {}

    for year in significant_intervals:
        matrix = transition_matrices.get(year)
        if matrix is None:
            print(f"⚠️ Warning: No transition matrix found for {year}, skipping.")
            continue
        
        try:
            # Extract 'Final total' row, excluding the last two summary rows ('Final total' & 'Gross gain')
            final_totals = matrix.loc['Final total'].iloc[:-2]
            total_area = final_totals.sum()

            # Initialize a DataFrame to store transition intensities for the current year
            transition_intensities = pd.DataFrame(0, 
                                                  index=matrix.index[:-2], 
                                                  columns=matrix.columns[:-2].append(pd.Index(['Uniform_Intensity'])))

            for category_from in matrix.index[:-2]:  # Exclude 'Final total' and 'Gross gain' rows
                gross_loss = matrix.loc[category_from, 'Gross loss']
                area_not_m = total_area - matrix.loc['Final total', category_from]

                if area_not_m > 0:  # Avoid division by zero
                    uniform_intensity = (gross_loss / duration_of_interval) / area_not_m * 100
                    transition_intensities.loc[category_from, 'Uniform_Intensity'] = uniform_intensity

                for category_to in matrix.columns[:-2]:  # Exclude 'Initial total' and 'Gross loss' columns
                    if category_from != category_to:  # Ensure transitions between different categories
                        transition_area = matrix.loc[category_from, category_to]
                        final_total_category_to = matrix.loc['Final total', category_to]

                        if final_total_category_to > 0:  # Avoid division by zero
                            transition_intensity = (transition_area / duration_of_interval) / final_total_category_to * 100
                            transition_intensities.at[category_from, category_to] = transition_intensity

            # Store computed transition intensities for the current year
            category_transitions[year] = transition_intensities

        except KeyError as e:
            print(f"⚠️ Warning: Missing data in transition matrix for {year}: {e}")
            continue  # Skip problematic years

    return category_transitions

def plot_transition_heatmap(significant_intervals, category_transitions, selection, sub_area):
    """
    Plots LULC transition intensity heatmaps for significant intervals.

    Parameters:
    - significant_intervals (list): List of significant years.
    - category_transitions (dict): Dictionary containing transition intensities.
    - selection (dict): Contains 'Park' name for title.
    - sub_area (str): Selected sub-area for investigation.

    Returns:
    - Displays a heatmap for each significant interval.
    """
    for year in significant_intervals:
        # Ensure the year exists in category_transitions
        if year not in category_transitions:
            print(f"⚠️ Warning: No data found for interval {year}, skipping.")
            continue

        matrix_to_plot = category_transitions[year].round(0)  # Round values for cleaner annotations

        # Create a mask to hide diagonal values
        mask = np.zeros_like(matrix_to_plot, dtype=bool)
        np.fill_diagonal(mask, True)

        # Ensure a copy of colormap before modifying
        cmap = mpl.colormaps.get_cmap('Greens').copy()
        cmap.set_bad("white")  # Set masked elements to white

        # Create the heatmap figure
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(matrix_to_plot, annot=True, fmt=".0f", cmap=cmap, linewidths=0.5, mask=mask)

        # Add vertical lines to separate categories
        for i in range(matrix_to_plot.shape[1] + 1):
            plt.axvline(x=i, color='black', linestyle='-', linewidth=1)

        # Labels and title
        plt.title(f"LULC Transition Intensity Analysis for {selection['Park']}'s {sub_area} Buffer Zone ({year})", fontsize=14)
        plt.xlabel('To Category & Uniform Intensity (percent of category)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('From Category (percent of category)', fontsize=12)

        # Show the plot
        plt.show()

#---------------------------------- To target category intensity analysis -------------------------------------------#
def calculate_category_transition_intensities_v2(significant_intervals, transition_matrices, duration_of_interval=1):
    """
    Computes category transition intensities for each significant time interval.

    Parameters:
    - significant_intervals (list): List of years with significant LULCC changes.
    - transition_matrices (dict): Dictionary containing transition matrices for each year.
    - duration_of_interval (int): Number of years in the interval (default=1).

    Returns:
    - category_transitions (dict): Dictionary with transition intensities for each category.
    """
    category_transitions = {}

    for year in significant_intervals:
        matrix = transition_matrices.get(year)
        if matrix is None:
            print(f"⚠️ Warning: No transition matrix found for {year}, skipping.")
            continue
        
        try:
            # Extract 'Initial total' column, excluding last two summary rows ('Final total' & 'Gross gain')
            initial_totals = matrix['Initial total'].iloc[:-2]
            total_area = initial_totals.sum()

            # Initialize DataFrame for transition intensities
            transition_intensities = pd.DataFrame(0, 
                                                  index=matrix.index[:-2], 
                                                  columns=matrix.columns[:-2].append(pd.Index(['Uniform_Intensity'])))

            for category_from in matrix.index[:-2]:  # Exclude 'Final total' and 'Gross gain' rows
                gross_gain = matrix.loc['Gross gain', category_from]
                area_not_n = total_area - matrix.loc[category_from, 'Initial total']

                if area_not_n > 0:  # Avoid division by zero
                    uniform_intensity = (gross_gain / duration_of_interval) / area_not_n * 100
                    transition_intensities.loc[category_from, 'Uniform_Intensity'] = uniform_intensity

                for category_to in matrix.columns[:-2]:  # Exclude 'Initial total' and 'Gross loss'
                    if category_from != category_to:  # Ensure transitions between different categories
                        transition_area = matrix.loc[category_from, category_to]
                        initial_total_category_from = matrix.loc[category_from, 'Initial total']

                        if initial_total_category_from > 0:  # Avoid division by zero
                            transition_intensity = (transition_area / duration_of_interval) / initial_total_category_from * 100
                            transition_intensities.at[category_from, category_to] = transition_intensity

            # Extract the 'Uniform_Intensity' column as a separate Series
            uniform_intensity_row = transition_intensities['Uniform_Intensity'].copy()

            # Remove the 'Uniform_Intensity' column
            transition_intensities.drop('Uniform_Intensity', axis=1, inplace=True)

            # Append the 'Uniform_Intensity' Series as a new row
            transition_intensities.loc['Uniform_Intensity'] = uniform_intensity_row

            # Store transition intensities matrix in dictionary
            category_transitions[year] = transition_intensities

        except KeyError as e:
            print(f"⚠️ Warning: Missing data in transition matrix for {year}: {e}")
            continue  # Skip problematic years

    return category_transitions

def plot_transition_heatmap_v2(significant_intervals, category_transitions, selection, sub_area):
    """
    Plots LULC transition intensity heatmaps for significant intervals.

    Parameters:
    - significant_intervals (list): List of significant years.
    - category_transitions (dict): Dictionary containing transition intensities.
    - selection (dict): Contains 'Park' name for title.
    - sub_area (str): Selected sub-area for investigation.

    Returns:
    - Displays a heatmap for each significant interval.
    """
    for year in significant_intervals:
        # Ensure the year exists in category_transitions
        if year not in category_transitions:
            print(f"⚠️ Warning: No data found for interval {year}, skipping.")
            continue

        matrix_to_plot = category_transitions[year].round(0)  # Round values for cleaner annotations

        # Create a mask to hide diagonal values
        mask = np.zeros_like(matrix_to_plot, dtype=bool)
        np.fill_diagonal(mask, True)

        # Ensure a copy of colormap before modifying
        cmap = mpl.colormaps.get_cmap('Reds').copy()
        cmap.set_bad("white")  # Set masked elements to white

        # Create the heatmap figure
        plt.figure(figsize=(12, 9))
        ax = sns.heatmap(matrix_to_plot, annot=True, fmt=".0f", cmap=cmap, linewidths=0.5, mask=mask)

        # Add horizontal lines to separate categories
        for i in range(matrix_to_plot.shape[0] + 1):
            plt.axhline(y=i, color='black', linestyle='-', linewidth=1)

        # Labels and title
        plt.title(f"LULC Transition Intensity Analysis for {selection['Park']}'s {sub_area} Buffer Zone ({year})", fontsize=14)
        plt.xlabel('To Category (percent of category)', fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('From Category & Uniform Intensity (percent of category)', fontsize=12)

        # Show the plot
        plt.show()