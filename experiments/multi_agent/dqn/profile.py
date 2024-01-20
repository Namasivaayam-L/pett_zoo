import pstats, os

# Analyze profiling results
profiling_directory = "experiments/multi_agent/dqn/profiling"
profiling_files = [os.path.join(profiling_directory, file) for file in os.listdir(profiling_directory) if file.endswith(".prof")]
for profile in profiling_files:
    # profiling_output_path = f"experiments/multi_agent/dqn/profiling/{profile}.prof"  # Change to the appropriate episode number
    stats = pstats.Stats(profile)
    stats.sort_stats('cumulative','s').print_stats(50)  # Print the top 20 functions by cumulative time
# import pstats
# import os

# def compare_profiling_stats(profiling_files, target_functions):
#     # Create a Stats object for each profiling file
#     stats_list = [pstats.Stats(file) for file in profiling_files]

#     # Sort statistics by cumulative time
#     stats_list.sort(key=lambda x: x.total_tt, reverse=True)

#     # Get common function statistics across all runs
#     common_stats = pstats.Stats(stats_list[0].stats)
#     for stats in stats_list[1:]:
#         common_stats.add(stats)

#     print("Common function statistics across all runs:")
#     common_stats.strip_dirs().sort_stats('cumulative').print_stats(20)  # Adjust the number as needed

#     # Print differences in function statistics for specific functions
#     for i, stats in enumerate(stats_list[1:], start=2):
#         print(f"\nDifferences in function statistics between run 1 and run {i} for target functions:")
#         stats.strip_dirs().sort_stats('cumulative').print_stats(20, target_functions)  # Adjust the number as needed

# if __name__ == "__main__":
#     # Specify the directory containing .prof files
#     profiling_directory = "experiments/multi_agent/dqn/profiling"

#     # Get a list of .prof files in the directory
#     profiling_files = [os.path.join(profiling_directory, file) for file in os.listdir(profiling_directory) if file.endswith(".prof")]

#     # Specify the target functions to focus on
#     target_functions = ['function_name_1', 'function_name_2', '...']

#     # Compare profiling statistics for specific functions
#     compare_profiling_stats(profiling_files, target_functions)
