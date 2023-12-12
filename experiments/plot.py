import pandas as pd
import matplotlib.pyplot as plt
import argparse,os

def plot_and_save(csv_file_path):
    # Extract the file name and directory from the CSV file path
    file_name, _ = os.path.splitext(csv_file_path)
    plot_save_path = f"{file_name}_combined.png"
    print(plot_save_path)
    # Read the CSV data
    data = pd.read_csv(csv_file_path)

    data = data.groupby('ep')[['1', '2', '5', '6']].mean().reset_index()
    data.columns = ['ep', '1', '2', '5', '6']
    
    ep_col = data['ep']
    reward_cols = data[['1', '2', '5', '6']].mean(axis=1)

    # Saving the calculated mean values to a new CSV
    plt.figure(figsize=(10, 6))
    # for i in range(reward_cols.shape[1]+1):
    #     plt.plot(ep_col, reward_cols.iloc[:, i], label=f"TS: {i + 1}")
    plt.plot(ep_col, reward_cols, label=f"SAC Rewards Combined")
    plt.legend()

    # Save the plot as a PNG image
    plt.savefig(plot_save_path)
    plt.close()

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path to the CSV file")
args = parser.parse_args()

# Check if the CSV file path is provided
if not args.file:
    print("Please provide the CSV file path using the -f flag.")
    exit()

# Plot and save the graph
plot_and_save(args.file)
