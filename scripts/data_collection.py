# data_collection.py - Handles LULC data processing and CSV storage

import pandas as pd
import ee
import tqdm
from collections import defaultdict
from pathlib import Path

def save_lulc_data(selection, base_path, window_geometries_per_sub_area, results_per_area_and_year, class_labels):
    """
    Process LULC data per sub-area per year and save results to CSV.
    """
    output_base_path = Path("..") / "data" / "DW_datasets" / selection["Park"]
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    for sub_area in selection["Sub Areas"]:
        window_geometries = window_geometries_per_sub_area[sub_area]
        all_years_data = []

        for year, dw_class in tqdm.tqdm(results_per_area_and_year[sub_area].items(), desc=f"Processing Years for {sub_area}"):
            aggregated_pixel_counts = defaultdict(int)

            def process_window(window_geometry):
                try:
                    pixel_count_stats = dw_class.reduceRegion(
                        reducer=ee.Reducer.frequencyHistogram(),
                        geometry=window_geometry,
                        scale=10,
                        maxPixels=1e10
                    ).getInfo()
                    return pixel_count_stats.get('label_mode', {})
                except Exception as e:
                    print(f"⚠️ Error processing window for {year} in {sub_area}: {e}")
                    return {}

            pixel_counts_list = [process_window(w) for w in tqdm.tqdm(window_geometries, desc=f"Processing Windows for Year {year}")]

            for pixel_counts in pixel_counts_list:
                for key, count in pixel_counts.items():
                    aggregated_pixel_counts[key] += count

            mapped_keys = {str(i): label for i, label in enumerate(class_labels)}
            pixel_counts_formatted = {mapped_keys.get(key, key): value for key, value in aggregated_pixel_counts.items()}

            all_years_data.append({'Year': year, **pixel_counts_formatted})

        df = pd.DataFrame(all_years_data).set_index('Year')
        filename = output_base_path / f"{selection['Park']}_{sub_area}_LULC_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"
        df.to_csv(filename)
        print(f"✅ Data saved: {filename}")


def compute_lulc_changes(selection, base_path, results_per_area_and_year):
    """
    Compute LULC changes for each sub-area and year pair.
    """
    results_per_area_and_year_pairs = {}

    for sub_area in selection["Sub Areas"]:
        dw_classes_per_year = results_per_area_and_year.get(sub_area, {})

        if len(dw_classes_per_year) < 2:
            print(f"⚠️ Not enough data for {sub_area} to compute LULC changes.")
            continue

        dw_classes_per_year_pairs = {
            f"{selection['Years'][i]}-{selection['Years'][i+1]}": dw_classes_per_year[selection['Years'][i]].select('label_mode').multiply(10).add(
                dw_classes_per_year[selection['Years'][i+1]].select('label_mode')
            )
            for i in range(len(selection["Years"]) - 1)
            if selection['Years'][i] in dw_classes_per_year and selection['Years'][i+1] in dw_classes_per_year
        }

        results_per_area_and_year_pairs[sub_area] = dw_classes_per_year_pairs

    return results_per_area_and_year_pairs


def process_lulc_transitions(selection, base_path, window_geometries_per_sub_area, results_per_area_and_year_pairs, class_labels):
    """
    Process LULC transitions per sub-area and save to CSV.
    """
    output_base_path = Path("..") / "data" / "DW_datasets" / selection["Park"]
    output_base_path.mkdir(parents=True, exist_ok=True)

    for sub_area in selection["Sub Areas"]:
        window_geometries = window_geometries_per_sub_area[sub_area]
        yearly_transition_counts = {}

        for year_pair, dw_class in tqdm.tqdm(results_per_area_and_year_pairs[sub_area].items(), desc=f"Processing Year Pairs for {sub_area}"):
            year_pair_transition_counts = defaultdict(int)
            pre_year, post_year = year_pair.split('-')

            transition_label_map = {str(i * 10 + j): f"{class_labels[i]}_to_{class_labels[j]}" 
                                    for i in range(len(class_labels)) for j in range(len(class_labels))}

            def process_window(window_geometry):
                try:
                    transition_counts = dw_class.reduceRegion(
                        reducer=ee.Reducer.frequencyHistogram(),
                        geometry=window_geometry,
                        scale=10,
                        maxPixels=1e10
                    ).getInfo()
                    return transition_counts.get('label_mode', {})
                except Exception as e:
                    print(f"⚠️ Error processing window for {year_pair} in {sub_area}: {e}")
                    return {}

            transition_counts_list = [process_window(w) for w in tqdm.tqdm(window_geometries, desc=f"Processing Windows for Year Pair {year_pair}")]

            for transition_counts_dict in transition_counts_list:
                for combined_value, count in transition_counts_dict.items():
                    transition_label = transition_label_map.get(str(combined_value), f"Unknown_{combined_value}")
                    year_pair_transition_counts[transition_label] += count

            yearly_transition_counts[f"{pre_year}_to_{post_year}"] = year_pair_transition_counts

        df_transitions = pd.DataFrame(yearly_transition_counts).fillna(0).reset_index().rename(columns={'index': 'Change'})
        ordered_cols = ['Change'] + sorted(df_transitions.columns[1:], key=lambda x: int(x.split('_to_')[0]) if x.split('_to_')[0].isdigit() else float('inf'))
        df_transitions = df_transitions[ordered_cols]

        csv_file_path = output_base_path / f"{selection['Park']}_{sub_area}_LULC_change_from_{selection['Starting Year']}_to_{selection['Ending Year']}.csv"
        df_transitions.to_csv(csv_file_path, index=False)
        print(f"✅ Data saved: {csv_file_path}")