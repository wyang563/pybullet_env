from collections import defaultdict
import os, re, json
import matplotlib.pyplot as plt
import numpy as np # Add numpy for averaging
import glob

def plot_accuracy_vs_agents_by_whale_count(data_dir="pybullet_env/icp/whale_data/eval_data_json", output_filename="pybullet_env/icp/accuracy_vs_agents_by_whale.pdf"):
    """
    Generates a plot showing box detection accuracy vs. number of agents,
    with separate lines colored by number of whales and styled by confidence threshold.

    Args:
        data_dir (str): Directory containing the conf_accuracy JSON files.
        output_filename (str): Path to save the generated plot.
    """
    # results[num_whales][model_conf][num_agents] = {'correct': 0, 'total': 0}
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0})))

    # Regex to extract num_agents and model_conf from filename
    pattern = re.compile(r"conf_accuracy_num_agents(\d+)_model_conf([\d.]+)\.json")

    # Find and process matching JSON files
    search_path = os.path.join(data_dir, "conf_accuracy_num_agents*.json")
    all_agent_counts = set() # Keep track of all agent counts encountered

    for filepath in glob.glob(search_path):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            num_agents = int(match.group(1))
            model_conf = float(match.group(2))
            all_agent_counts.add(num_agents)

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Iterate through all buckets (keys like "1", "2", "3") in the JSON
                for bucket_key in data:
                    for entry in data[bucket_key]:
                        try:
                            num_whales = entry.get("num_whales")
                            is_correct = entry.get("correct", False)
                            if num_whales is not None:
                                num_whales = int(num_whales) # Ensure it's an int
                                results[num_whales][model_conf][num_agents]['total'] += 1
                                if is_correct:
                                    results[num_whales][model_conf][num_agents]['correct'] += 1
                        except (ValueError, TypeError) as e_inner:
                            print(f"Warning: Skipping entry {entry} in {filename} due to data error: {e_inner}")


            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Prepare for plotting
    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap('tab10') # Colormap for whale counts
    line_styles = { # Line styles for confidence levels
        0.3: '--', # Dashed
        0.5: '-.', # Dash-dot
        0.7: '-'   # Solid
    }
    
    whale_counts_sorted = sorted(results.keys())
    conf_levels_sorted = sorted([k for k in line_styles.keys() if any(k in results[wc] for wc in results)]) # Only conf levels present in data

    color_idx = 0
    # Plot data
    for num_whales in whale_counts_sorted:
        color = cmap(color_idx % cmap.N)
        has_data_for_whale = False
        for model_conf in conf_levels_sorted:
            if model_conf in results[num_whales]:
                agent_data = results[num_whales][model_conf]
                
                plot_points = []
                for num_agents in sorted(agent_data.keys()):
                    counts = agent_data[num_agents]
                    if counts['total'] > 0:
                        accuracy = counts['correct'] / counts['total']
                        plot_points.append((num_agents, accuracy))
                
                # Sort points by number of agents for plotting
                plot_points.sort(key=lambda x: x[0])
                
                agents = [item[0] for item in plot_points]
                accuracies = [item[1] for item in plot_points]

                if agents: # Only plot if there's data
                    plt.plot(agents, accuracies, marker='o', linestyle=line_styles.get(model_conf, '-'), color=color, label=f'{num_whales} Whales, Conf={model_conf}')
                    has_data_for_whale = True
        
        if has_data_for_whale: # Increment color index only if we plotted something for this whale count
             color_idx += 1


    plt.xlabel("Number of Agents")
    plt.ylabel("Box Detection Accuracy")
    plt.title("Box Detection Accuracy vs. Agents (Colored by Whale Count, Styled by Confidence)")
    
    # Ensure all agent counts found across files are shown as ticks
    if all_agent_counts:
        plt.xticks(sorted(list(all_agent_counts))) 
        
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title="Whale Count & Confidence") # Adjust legend position
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    # Save the plot
    plt.savefig(output_filename, dpi=300)
    print(f"Accuracy by whale count plot saved to {output_filename}")
    plt.close()


def plot_model_accuracy_vs_whale_count(data_dir="pybullet_env/icp/whale_data/eval_data_json", output_filename="pybullet_env/icp/model_accuracy_vs_whale_count.pdf"):
    """
    Generates a plot showing model accuracy vs. number of whales,
    with separate lines for different model confidence thresholds.

    Args:
        data_dir (str): Directory containing the model_accuracy JSON files.
        output_filename (str): Path to save the generated plot.
    """
    # results[model_conf][num_whales] = accuracy
    results = defaultdict(dict)

    # Regex to extract model_conf from filename
    pattern = re.compile(r"model_accuracy_conf_([\d.]+)\.json")

    # Find and process matching JSON files
    search_path = os.path.join(data_dir, "model_accuracy_conf_*.json")
    all_whale_counts = set() # Keep track of all whale counts encountered

    for filepath in glob.glob(search_path):
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            model_conf = float(match.group(1))

            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                # Iterate through whale counts (keys like "8", "9") in the JSON
                for num_whales_str, accuracy_data in data.items():
                    try:
                        num_whales = int(num_whales_str)
                        accuracy = accuracy_data.get('accuracy')
                        if accuracy is not None:
                            results[model_conf][num_whales] = accuracy
                            all_whale_counts.add(num_whales)
                        else:
                            print(f"Warning: 'accuracy' key missing for whale count {num_whales_str} in {filename}")

                    except ValueError:
                        print(f"Warning: Could not convert key '{num_whales_str}' to int in {filename}. Skipping.")
                    except Exception as e_inner:
                         print(f"Error processing key '{num_whales_str}' in {filename}: {e_inner}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Prepare for plotting
    plt.figure(figsize=(12, 7))
    cmap = plt.get_cmap('viridis') # Get a different colormap
    conf_levels = sorted(results.keys())
    color_idx = np.linspace(0, 1, len(conf_levels)) # Generate colors across the map

    # Sort data and plot
    for i, model_conf in enumerate(conf_levels):
        whale_data = results[model_conf]
        
        # Sort points by number of whales for plotting
        plot_points = sorted(whale_data.items())
        
        whales = [item[0] for item in plot_points]
        accuracies = [item[1] for item in plot_points]

        if whales: # Only plot if there's data for this confidence level
            plt.plot(whales, accuracies, marker='o', linestyle='-', color=cmap(color_idx[i]), label=f'Conf={model_conf}')

    plt.xlabel("Number of Whales in Image")
    plt.ylabel("Model Accuracy (Correct Whale Count / Total Augmentations)")
    plt.title("Model Accuracy vs. Number of Whales by Confidence Threshold")
    
    # Ensure all whale counts found across files are shown as ticks
    if all_whale_counts:
        # Create integer ticks from min to max whale count
        min_whales = min(all_whale_counts)
        max_whales = max(all_whale_counts)
        plt.xticks(range(min_whales, max_whales + 1)) 
        
    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--')
    plt.legend(title="Confidence Threshold")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_filename, dpi=300)
    print(f"Model accuracy plot saved to {output_filename}")
    plt.close()


def plot_whale_icp_eval(data_dir="pybullet_env/icp/whale_data/eval_data_json", vary_heights=False):
    """
    Generates a plot showing ICP accuracy vs. number of whales,
    with separate lines for different numbers of agents (2, 4, 8, 16).

    Args:
        data_dir (str): Directory containing the icp_accuracy JSON files.
        output_filename (str): Path to save the generated plot.
    """
    output_filename = "pybullet_env/icp/icp_accuracy_vs_whales_by_agents_vary_heights.pdf" if vary_heights else "pybullet_env/icp/icp_accuracy_vs_whales_by_agents.pdf"
    # results[num_agents][num_whales] = {'correct': 0, 'total': 0}
    results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    target_agents = {2, 4, 8, 16} # Specific agent counts to plot
    all_whale_counts = set()

    # Regex to find relevant files (handles variations in conf/vary_heights)
    # Example filename: icp_accuracy_vs_agents_conf_0.3_vary_heights_True.json
    if vary_heights:
        pattern = re.compile(r"icp_accuracy_vs_agents_conf_[\d.]+_vary_heights_True\.json")
    else:
        pattern = re.compile(r"icp_accuracy_vs_agents_conf_[\d.]+_vary_heights_False\.json")

    search_path = os.path.join(data_dir, "icp_accuracy_vs_agents_conf_*.json") # Broad search first

    for filepath in glob.glob(search_path):
        filename = os.path.basename(filepath)
        if not pattern.match(filename): # Ensure it matches the specific ICP eval pattern
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Iterate through agent counts (keys like "2", "4")
            for num_agents_str, whale_data in data.items():
                try:
                    num_agents = int(num_agents_str)
                    if num_agents not in target_agents:
                        continue # Skip agent counts we don't need to plot

                    # Iterate through whale counts (keys like "1", "5")
                    for num_whales_str, success_list in whale_data.items():
                        try:
                            num_whales = int(num_whales_str)
                            if isinstance(success_list, list):
                                correct_count = sum(1 for success in success_list if success is True)
                                total_count = len(success_list)

                                results[num_agents][num_whales]['correct'] += correct_count
                                results[num_agents][num_whales]['total'] += total_count
                                all_whale_counts.add(num_whales)
                            else:
                                print(f"Warning: Expected list for whale count {num_whales_str} under agent {num_agents_str} in {filename}, got {type(success_list)}. Skipping.")

                        except ValueError:
                            print(f"Warning: Could not convert whale count key '{num_whales_str}' to int in {filename}. Skipping.")
                        except Exception as e_inner:
                            print(f"Error processing whale entry '{num_whales_str}' for agent {num_agents_str} in {filename}: {e_inner}")

                except ValueError:
                    print(f"Warning: Could not convert agent key '{num_agents_str}' to int in {filename}. Skipping.")
                except Exception as e_agent:
                    print(f"Error processing agent entry '{num_agents_str}' in {filename}: {e_agent}")

        except json.JSONDecodeError:
            print(f"Error decoding JSON from {filename}")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Prepare for plotting
    plt.figure(figsize=(12, 7))
    colors = plt.get_cmap('tab10') # Use a colormap for distinct colors

    agent_plot_order = sorted(list(target_agents)) # Plot in order [2, 4, 8, 16]

    # Plot data for each target agent count
    for i, num_agents in enumerate(agent_plot_order):
        if num_agents in results:
            agent_data = results[num_agents]
            plot_points = []
            for num_whales in sorted(agent_data.keys()):
                counts = agent_data[num_whales]
                if counts['total'] > 0:
                    accuracy = counts['correct'] / counts['total']
                    plot_points.append((num_whales, accuracy))
                else:
                     plot_points.append((num_whales, 0)) # Add point with 0 accuracy if no runs

            # Sort points by number of whales for plotting
            plot_points.sort(key=lambda x: x[0])

            whales = [item[0] for item in plot_points]
            accuracies = [item[1] for item in plot_points]

            if whales: # Only plot if there's data
                plt.plot(whales, accuracies, marker='o', linestyle='-', color=colors(i), label=f'{num_agents} Agents')

    plt.xlabel("Number of Whales")
    plt.ylabel("ICP Accuracy")
    plt.title(f"ICP Accuracy vs. Number of Whales by Agent Count - Heights Varied: {vary_heights}")

    # Ensure integer ticks for whale counts
    if all_whale_counts:
        min_whales = min(all_whale_counts) if all_whale_counts else 1
        max_whales = max(all_whale_counts) if all_whale_counts else 1
        # Ensure ticks cover the full range, potentially adding ticks if sparse
        # Generate ticks for every integer whale count in the range
        plt.xticks(np.arange(min_whales, max_whales + 1, 1))


    plt.ylim(0, 1.1)
    plt.grid(True, linestyle='--')
    plt.legend(title="Number of Agents")
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_filename, dpi=300)
    print(f"ICP accuracy plot saved to {output_filename}")
    plt.close()

def plot_icp_noise_eval(json_filepaths, output_filename="pybullet_env/icp/icp_noise_eval.pdf"):
    """
    Plot ICP accuracy vs. number of whales for multiple JSON results files.
    Lines colored by agent count [2,4,8,16]; line style (dash) by noise_level extracted from filename.
    """
    agents = [2, 4, 8, 16]
    cmap = plt.get_cmap("tab10")
    color_map = {a: cmap(i % cmap.N) for i, a in enumerate(agents)}

    # extract all noise levels
    nl_values = []
    pattern_nl = re.compile(r"noiselevel_(\d+)")
    for fp in json_filepaths:
        m = pattern_nl.search(fp)
        if m: nl_values.append(int(m.group(1)))
    nl_values = sorted(set(nl_values))

    # assign dash styles
    dash_styles = ["-", "--", "-.", ":"]
    nl_style = {nl: dash_styles[i % len(dash_styles)] for i, nl in enumerate(nl_values)}

    plt.figure(figsize=(10, 6))
    for fp in json_filepaths:
        m = pattern_nl.search(fp)
        nl = int(m.group(1)) if m else 0
        style = nl_style.get(nl, "-")
        with open(fp) as f:
            data = json.load(f)

        for ag_str, whale_dict in data.items():
            ag = int(ag_str)
            if ag not in agents: continue
            pts = []
            for wh_str, runs in whale_dict.items():
                wc = int(wh_str)
                if wc <= 2:  # skip 2 whales or less
                    continue
                if runs:
                    acc = sum(runs) / len(runs)
                    pts.append((wc, acc))
            if not pts: continue
            pts.sort(key=lambda x: x[0])
            xs, ys = zip(*pts)
            plt.plot(xs, ys,
                     color=color_map[ag],
                     linestyle=style,
                     marker='o',
                     label=f"{ag} ag, nl={nl}")

    plt.xlabel("Number of Whales")
    plt.ylabel("ICP Accuracy")
    plt.title("ICP Accuracy vs. Whales for varying agents & noise levels")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--')
    plt.legend(bbox_to_anchor=(1.04, 1),
               loc="upper left",
               title="Agents & noise")
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(output_filename, dpi=300)
    plt.close()


def plot_icp_height_eval(json_filepaths, output_filename="pybullet_env/icp/icp_height_eval.pdf"):
    """
    Plot ICP accuracy vs. number of whales for multiple JSON results files.
    Lines colored by agent count [2,4,8,16]; line style (dash) by height_variation extracted from filename.
    """
    # prepare color map for agents
    agents = [2, 4, 8, 16]
    cmap = plt.get_cmap("tab10")
    color_map = {a: cmap(i % cmap.N) for i, a in enumerate(agents)}

    # extract all height_variations
    hv_values = []
    pattern_hv = re.compile(r"heightvar[_]?(\d*\.?\d+)")
    for fp in json_filepaths:
        m = pattern_hv.search(fp)
        if m: hv_values.append(float(m.group(1)))
    hv_values = sorted(set(hv_values))
    # assign dash styles
    dash_styles = ["-", "--", "-.", ":"]
    hv_style = {hv: dash_styles[i % len(dash_styles)] for i, hv in enumerate(hv_values)}

    plt.figure(figsize=(10, 6))
    for fp in json_filepaths:
        # extract height variation
        m = pattern_hv.search(fp)
        hv = float(m.group(1)) if m else 0.0
        style = hv_style.get(hv, "-")
        # load data
        with open(fp) as f:
            data = json.load(f)
        # for each agent
        for ag_str, whale_dict in data.items():
            ag = int(ag_str)
            if ag not in agents: continue
            # collect (whale_count, accuracy)
            pts = []
            for wh_str, runs in whale_dict.items():
                wc = int(wh_str)
                # skip whale counts of 2 or less
                if wc <= 2:
                    continue
                if runs:
                    acc = sum(1 for x in runs if x) / len(runs)
                    pts.append((wc, acc))
            if not pts: continue
            pts.sort(key=lambda x: x[0])
            xs, ys = zip(*pts)
            plt.plot(xs, ys, color=color_map[ag], linestyle=style,
                     marker='o',
                     label=f"{ag} ag, hv={hv}")

    plt.xlabel("Number of Whales")
    plt.ylabel("ICP Accuracy")
    plt.title("ICP Accuracy vs. Whales for varying agents & height variation")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--')
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", title="Agents & hv")
    plt.tight_layout(rect=[0,0,0.8,1])
    plt.savefig(output_filename, dpi=300)
    plt.close()

def plot_box_icp_vs_point_icp(point_icp_json_path, box_icp_json_path, output_path="pybullet_env/icp/box_vs_point_icp.pdf"):
    """
    Generates a plot comparing ICP accuracy vs. number of agents for point ICP and box ICP.

    Args:
        point_icp_json_path (str): Path to the JSON file containing point ICP results.
        box_icp_json_path (str): Path to the JSON file containing box ICP results.
        output_path (str): Path to save the generated plot.
    """
    def calculate_accuracies(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        accuracies = {}
        agent_counts = sorted([int(k) for k in data.keys()]) # Sort agent counts numerically

        for agent_count in agent_counts:
            agent_data = data[str(agent_count)] # Agent count is string in JSON keys
            total_successes = 0
            total_attempts = 0
            for whale_runs in agent_data.values():
                total_successes += sum(1 for success in whale_runs if success)
                total_attempts += len(whale_runs)
            
            if total_attempts > 0:
                accuracies[agent_count] = total_successes / total_attempts
            else:
                accuracies[agent_count] = 0 # Or handle as NaN or skip if preferred
        # Ensure all agent counts are present, even if with 0 accuracy
        # This helps in aligning plots if one dataset is missing an agent count
        all_agent_keys = sorted(list(set(accuracies.keys()))) # Get all unique sorted agent keys
        
        return [accuracies.get(agent, 0) for agent in all_agent_keys], all_agent_keys


    point_icp_accuracies, point_agent_counts = calculate_accuracies(point_icp_json_path)
    box_icp_accuracies, box_agent_counts = calculate_accuracies(box_icp_json_path)

    # Ensure both lists of agent counts are aligned for plotting
    # We'll use the union of agent counts and sort them
    all_agent_counts = sorted(list(set(point_agent_counts + box_agent_counts)))

    # Re-map accuracies to the unified agent counts, filling with 0 if an agent count is missing in one dataset
    # This is a more robust way to handle potentially different sets of agent counts in the two files.
    
    # Create dictionaries for easy lookup
    point_acc_map = dict(zip(point_agent_counts, point_icp_accuracies))
    box_acc_map = dict(zip(box_agent_counts, box_icp_accuracies))

    # Create aligned accuracy lists
    aligned_point_accuracies = [point_acc_map.get(count, 0) for count in all_agent_counts]
    aligned_box_accuracies = [box_acc_map.get(count, 0) for count in all_agent_counts]


    plt.figure(figsize=(10, 6))
    
    plt.plot(all_agent_counts, aligned_point_accuracies, marker='o', linestyle='--', label='Point ICP')
    plt.plot(all_agent_counts, aligned_box_accuracies, marker='s', linestyle='-', label='Box ICP')
    
    plt.xlabel("Number of Agents")
    plt.ylabel("ICP Accuracy")
    plt.title("ICP Accuracy: Box vs. Point")
    plt.legend()
    plt.grid(True)
    plt.xticks(all_agent_counts) # Ensure all agent counts are shown as ticks
    plt.ylim(0, 1.05) # Accuracy is between 0 and 1

    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    print("Starting whale paper plots")
    plot_box_icp_vs_point_icp("pybullet_env/icp/whale_data/eval_data_json/icp_accuracy_vs_agents_conf_0.3_vary_heights_False_use_point_icp_True_blur_0.json", "pybullet_env/icp/whale_data/eval_data_json/icp_accuracy_vs_agents_conf_0.3_vary_heights_False_use_point_icp_False_blur_0.json")

    # Example usage: Call the functions you want to run
    # plot_accuracy_vs_agents_by_whale_count()
    # plot_model_accuracy_vs_whale_count()
    # plot_icp_accuracy_vs_agents("path/to/your_point_icp_data.json", "pybullet_env/icp/whale_data/eval_data_json/icp_accuracy_vs_agents_conf_0.3_vary_heights_False_use_point_icp_False_blur_35.json", "actual_comparison_plot.png")