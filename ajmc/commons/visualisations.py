from typing import List, Dict

import matplotlib.pyplot as plt


def draw_lineplot(data_dict: Dict[str, List[float]],
                  x_label: str = "X-axis",
                  y_label: str = "Y-axis",
                  title: str = "Line Plot",
                  show: bool = True,
                  output_path: str = None):
    """
    Draw a line plot from a dictionary of lists.

    Args:
        data_dict (dict): A dictionary where keys are labels for different lines, and values are lists of data points.
        x_label (str, optional): Label for the X-axis. Default is "X-axis".
        y_label (str, optional): Label for the Y-axis. Default is "Y-axis".
        title (str, optional): Title of the plot. Default is "Line Plot".

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each line from the dictionary
    for label, values in data_dict.items():
        plt.plot(values, label=label)

    # Set X and Y labels and plot title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add a legend to distinguish lines
    plt.legend()
    plt.grid()

    # Show the plot
    if show:
        plt.show()

    if output_path is not None:
        plt.savefig(output_path, dpi=300)

    plt.close()
