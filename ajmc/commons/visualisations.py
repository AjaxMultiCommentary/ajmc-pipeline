from typing import List, Dict, Optional, Union

import matplotlib.pyplot as plt


def draw_lineplot(data_dict: Dict[str, List[float]],
                  x_values: Optional[List[float]] = None,
                  x_label: str = "X-axis",
                  y_label: str = "Y-axis",
                  title: str = "Line Plot",
                  show: bool = True,
                  output_path: Union[str, 'Path'] = None):
    """
    Draw a line plot from a dictionary of lists.

    Args:
        data_dict: A dictionary where keys are labels for different lines, and values are lists of data points.
        x_values: A list of X-axis values. If None, the indices of the lists in data_dict are used.
        x_label: Label for the X-axis. Default is "X-axis".
        y_label: Label for the Y-axis. Default is "Y-axis".
        title: Title of the plot. Default is "Line Plot".
        show: Whether to show the plot. Default is True.
        output_path: Path to save the plot. If None, the plot is not saved. Default is None.

    Returns:
        None
    """
    # Create a new figure
    plt.figure(figsize=(10, 6))

    # Plot each line from the dictionary
    for label, values in data_dict.items():
        if x_values is None:
            plt.plot(values, label=label)
        else:
            plt.plot(x_values, values, label=label)

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

# data = {'coucou': [1, 2, 8, 4, 5], 'salut': [2, 4, 6, 1, 10]}
# draw_lineplot(data, x_values=[1000, 1001, 1002, 1003, 1004], show=True, output_path='/Users/sven/Desktop/test.png')
