import matplotlib.pyplot as plt


def visualize_on_surface_and_in_air(file_path):
    # Loading the .svc file content
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting data and storing in lists
    x_coords, y_coords, times, flags = [], [], [], []
    for line in lines:
        if line[0] != '#':  # Ignoring comment lines
            data = line.split()
            if len(data) >= 4:  # Making sure the line contains enough data
                x_coords.append(float(data[0]))
                y_coords.append(float(data[1]))
                times.append(float(data[2]))
                flags.append(int(data[3]))

    # Creating a figure for the visualization
    plt.figure(figsize=(10, 6))

    # Initializing variables to store the start index of each continuous movement segment
    start_idx = 0

    # Iterating over rows in the dataframe to plot each segment based on the flag values
    for idx in range(len(flags)):
        if idx > 0 and flags[idx - 1] != flags[idx]:
            # If there is a change in the flag value, plot the segment and update the start index
            color = 'b' if flags[idx - 1] == 1 else 'r'
            plt.plot(x_coords[start_idx:idx], y_coords[start_idx:idx], color=color)
            plt.scatter(x_coords[start_idx:idx], y_coords[start_idx:idx], color=color, edgecolor='none', s=10)
            start_idx = idx

    # Plotting the last segment
    color = 'blue' if flags[start_idx] == 1 else 'red'
    plt.plot(x_coords[start_idx:], y_coords[start_idx:], color=color)
    plt.scatter(x_coords[start_idx:], y_coords[start_idx:], color=color, edgecolor='none', s=10)

    # Setting titles and labels
    plt.title('Handwriting Visualization with On-surface (Blue) and In-air (Red) Movements')
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.legend(['On-surface (line)', 'On-surface (points)', 'In-air (line)', 'In-air (points)'])
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    svc_file_path = 'all_in_one\\00006.svc'

    visualize_on_surface_and_in_air(svc_file_path)
