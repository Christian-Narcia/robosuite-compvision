#!/usr/bin/env python3

import os
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_scalar_data(logdir, scalar_tag):
    """
    Loads data for the given scalar tag from a TensorBoard event log directory.
    Returns two lists: steps and values.
    """
    event_acc = EventAccumulator(logdir)
    event_acc.Reload()
    
    # Check if the scalar tag is available
    available_tags = event_acc.Tags().get('scalars', [])
    if scalar_tag not in available_tags:
        raise ValueError(
            f"Scalar tag '{scalar_tag}' not found in {logdir}. "
            f"Available tags: {available_tags}"
        )
    
    # Extract all scalar events under 'scalar_tag'
    scalar_events = event_acc.Scalars(scalar_tag)
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    return steps, values

def gather_runs_and_aggregate(logdir, scalar_key):
    """
    1. Finds all subdirectories in `logdir` that start with 'PPO_'.
    2. Loads the scalar data from each run.
    3. Aggregates the data by step, collecting values from each run that has that step.
    4. Returns:
       - sorted_steps: A sorted list of unique steps
       - means: The mean scalar value across runs at each step
       - mins: The minimum scalar value across runs at each step
       - maxs: The maximum scalar value across runs at each step
    """
    subdirs = [
        os.path.join(logdir, d)
        for d in os.listdir(logdir) 
        if os.path.isdir(os.path.join(logdir, d)) and d.startswith("PPO_")
    ]

    aggregated_data = {}  # step -> list of values

    for sub in subdirs:
        # Look for any event files in this subdirectory
        event_files = glob.glob(
            os.path.join(sub, "**", "events.out.tfevents.*"), 
            recursive=True
        )
        if not event_files:
            print(f"No TensorBoard event files found in {sub}. Skipping.")
            continue

        for ef in event_files:
            try:
                steps, values = load_scalar_data(ef, scalar_key)
            except ValueError as e:
                # If scalar_key is not found, skip
                print(e)
                continue

            # Populate aggregated_data
            for s, v in zip(steps, values):
                if s not in aggregated_data:
                    aggregated_data[s] = []
                aggregated_data[s].append(v)

    # Convert the aggregated data into sorted lists
    sorted_steps = sorted(aggregated_data.keys())
    means, mins, maxs = [], [], []
    for step in sorted_steps:
        vals = aggregated_data[step]
        means.append(sum(vals) / len(vals))
        mins.append(min(vals))
        maxs.append(max(vals))

    return sorted_steps, means, mins, maxs

def plot_aggregated_scalar(logdir, scalar_key):
    """
    Gathers runs from `logdir` (subdirs starting with 'PPO_'), then plots both:
      - min–max range (fill_between)
      - mean curve
    The label for each run is based on the logdir's folder name.
    """
    sorted_steps, means, mins, maxs = gather_runs_and_aggregate(logdir, scalar_key)
    
    # Use the directory name as a label to distinguish multiple calls
    run_label = os.path.basename(os.path.normpath(logdir))

    # Fill between min and max
    plt.fill_between(sorted_steps, mins, maxs, alpha=0.2)
    
    # Plot the mean curve
    plt.plot(sorted_steps, means, label=f'{run_label}')


if __name__ == "__main__":
    # Create a new figure
    plt.figure(figsize=(13, 6))

    # First directory
    # logdir1 = "../cluster-robo-04_12/runs/ImgPPO_reachImgSimplified_sb3_simplified_v4/"
    # # Second directory
    # logdir2 = "../cluster-robo-04_12/runs/ImgPPO_reachImgSimplified_sb3_simplifiedv3/"

    # logdir3 = "../cluster-robo-04_12/runs/PPO_reachImgSimplified_sb3_simplifiedv3/"

    # First directory
    logdir1 = "../cluster-robo-04_12/runs/ImgPPO_reachImgSimplified_sb3_simplified_v5/"
    
    # Provide the scalar you want to track (e.g. 'rollout/ep_rew_mean')
    scalar_key = "rollout/ep_rew_mean"

    # Plot from the first directory
    plot_aggregated_scalar(logdir1, scalar_key)
    # Plot from the second directory
    # plot_aggregated_scalar(logdir2, scalar_key)

    # plot_aggregated_scalar(logdir3, scalar_key)

    # Customize plot
    plt.title(f"Reward performance Simplified v5 from Image only")
    plt.xlabel("Steps")
    plt.ylabel(scalar_key)
    plt.grid(True)
    plt.legend()

    # Finally show the figure
    plt.show()
