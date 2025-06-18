import json
from pathlib import Path
import os
from typing import Union
import warnings
import pickle

def analyze_kspace_metadata(base_dir: Union[str, Path]):
    """
    Scans all SG_xxx/metadata.json files in the given base directory
    to find unique irreducible representation (irrep) names and the
    maximum length of decomposition_indices.

    Args:
        base_dir (Union[str, Path]): The base directory containing SG_xxx folders.
                                      e.g., "/scratch/gpfs/as0714/graph_vector_topological_insulator/nonmagnetic_3d/kspace_topology_graphs"

    Returns:
        tuple: A tuple containing:
            - list: Sorted list of all unique irrep names found.
            - int: The maximum length of 'decomposition_indices' list found.
    """
    base_path = Path(base_dir).resolve()
    if not base_path.is_dir():
        raise NotADirectoryError(f"Provided base_dir is not a directory: {base_path}")

    all_unique_irreps = set()
    max_decomposition_indices_len = 0
    processed_files_count = 0
    errors_count = 0

    print(f"Scanning directory: {base_path}")

    for entry in os.scandir(base_path):
        if entry.is_dir() and entry.name.startswith('SG_'):
            sg_dir = Path(entry.path)
            sg_metadata_path = sg_dir / 'metadata.json'

            if sg_metadata_path.exists():
                try:
                    with open(sg_metadata_path, 'r', encoding='utf-8') as f:
                        sg_metadata = json.load(f)
                        processed_files_count += 1

                        if 'ebr_data' in sg_metadata and 'irreps' in sg_metadata['ebr_data']:
                            irreps_list = sg_metadata['ebr_data']['irreps']
                            normalized_irreps = [ir.replace('\u0393', 'Γ') for ir in irreps_list]
                            all_unique_irreps.update(normalized_irreps)
                        
                        if 'decomposition_branches' in sg_metadata and 'decomposition_indices' in sg_metadata['decomposition_branches']:
                            indices_list = sg_metadata['decomposition_branches']['decomposition_indices']
                            current_len = len(indices_list)
                            if current_len > max_decomposition_indices_len:
                                max_decomposition_indices_len = current_len

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in {sg_metadata_path}: {e}")
                    errors_count += 1
                except FileNotFoundError: # Should not happen if .exists() check passes, but good for robustness
                    pass
                except Exception as e:
                    print(f"An unexpected error occurred with {sg_metadata_path}: {e}")
                    errors_count += 1
            else:
                # print(f"Warning: {sg_metadata_path} not found.") # Uncomment for detailed warnings
                pass

    if processed_files_count == 0:
        warnings.warn(f"No 'metadata.json' files found or processed in {base_path}. Check path and file structure.")

    final_all_possible_irreps = sorted(list(all_unique_irreps))

    print(f"\n--- Scan Summary ---")
    print(f"Total SG_xxx/metadata.json files processed: {processed_files_count}")
    if errors_count > 0:
        print(f"Files with errors: {errors_count}")
    print(f"Number of unique irreps found: {len(final_all_possible_irreps)}")
    print(f"Maximum length of 'decomposition_indices' found: {max_decomposition_indices_len}")
    #print(f"\nCopy this list for ALL_POSSIBLE_IRREPS in helper/config.py:")
    #print(f"ALL_POSSIBLE_IRREPS = {final_all_possible_irreps}")
    print(f"\nCopy this value for MAX_DECOMPOSITION_INDICES_LEN in helper/config.py:")
    print(f"MAX_DECOMPOSITION_INDICES_LEN = {max_decomposition_indices_len}")

    
    with open('/Users/abiralshakya/Documents/Research/MultitaskClassifier_Topo/multitask_ti_classification/irrep_unique', 'wb') as fp:
        pickle.dump(final_all_possible_irreps, fp)

    return final_all_possible_irreps, max_decomposition_indices_len

if __name__ == "__main__":
    # --- IMPORTANT: Set this path to your actual kspace_topology_graphs directory ---
    KSPACE_GRAPHS_BASE_DIR = "/Users/abiralshakya/Documents/Research/GraphVectorTopological/kspace_topology_graphs"
    
    unique_irreps, max_indices_len = analyze_kspace_metadata(KSPACE_GRAPHS_BASE_DIR)

    # Example of how you would use these in your config.py
    # print("\n--- Example for helper/config.py ---")
   # print(f"ALL_POSSIBLE_IRREPS = {unique_irreps}")
   
   # print(f"MAX_DECOMPOSITION_INDICES_LEN = {max_indices_len}")