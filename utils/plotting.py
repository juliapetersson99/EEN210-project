import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

def split_dataframe_into_sliding_windows(df, window_size):

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    windowed_dfs = []
    start = df.index.min()
    end = df.index.max()

    current_start = start
    while current_start < end:
        print(current_start)
        current_end = current_start + pd.Timedelta(window_size)
        window = df[current_start:current_end]
        if not window.empty:
            windowed_dfs.append(window.reset_index())
        current_start += pd.Timedelta(window_size)

    return windowed_dfs


def plot_full_sequence(data_frame):

    colors = ["b", "g", "r"]
    styles = ["-", "--", "-."]
    index_x = ["x", "y", "z"]

    X = data_frame["timestamp"]
    gyroscope = data_frame[["gyroscope_x", "gyroscope_y", "gyroscope_z"]]
    accelerometer = data_frame[["acceleration_x", "acceleration_y", "acceleration_z"]]

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Full sequence plot")
    ax[0].set_title("Gyroscope")
    ax[1].set_title("Accelerometer")
    ax[0].set_xlabel("Time")
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("dps (degrees per second)")
    ax[1].set_ylabel("m/s^2 (g)")

    for i in range(3):
        ax[0].plot(
            X,
            gyroscope.iloc[:, i],
            color=colors[i],
            linestyle=styles[i],
            label="gyroscope_" + index_x[i],
        )
        ax[1].plot(
            X,
            accelerometer.iloc[:, i],
            color=colors[i],
            linestyle=styles[i],
            label="accelerometer_" + index_x[i],
        )

    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("plots/full_sequence_numpy.png")
    plt.close()


labels = ["falling", "walking", "running", "sitting", "standing", "laying", "recover"]
label_colors = {
    "falling": "#e41a1c",  # Red
    "walking": "#377eb8",  # Blue
    "running": "#4daf4a",  # Green
    "sitting": "#984ea3",  # Purple
    "standing": "#ff7f00",  # Orange
    "laying": "#ffff33",  # Yellow
    "recover": "#a65628",  # Brown
}


def plot_with_labels(data_frame, plt_name):

    colors = ["b", "g", "r", "tab:blue", "tab:green", "tab:red"]
    styles = ["-", "-", "-", "--", "--", "--"]
    index_x = ["x", "y", "z", "x_avg", "y_avg", "z_avg"]

    X = data_frame["timestamp"]
    gyroscope = data_frame[["gyroscope_x", "gyroscope_y", "gyroscope_z"]]
    accelerometer = data_frame[["acceleration_x", "acceleration_y", "acceleration_z"]]

    for col in gyroscope.columns:
        gyroscope[col + "_avg"] = gyroscope[col].rolling(window=20).mean()

    for col in accelerometer.columns:
        accelerometer[col + "_avg"] = accelerometer[col].rolling(window=20).mean()

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle("Full sequence plot")
    ax[0].set_title("Gyroscope")
    ax[1].set_title("Accelerometer")
    ax[0].set_xlabel("Time")
    ax[1].set_xlabel("Time")
    ax[0].set_ylabel("dps (degrees per second)")
    ax[1].set_ylabel("m/s^2 (g)")

    for i in range(6):
        ax[0].plot(
            X,
            gyroscope.iloc[:, i],
            color=colors[i],
            linestyle=styles[i],
            label="gyroscope_" + index_x[i],
        )
        ax[1].plot(
            X,
            accelerometer.iloc[:, i],
            color=colors[i],
            linestyle=styles[i],
            label="accelerometer_" + index_x[i],
        )

    labels = data_frame["label"]
    change_points = labels.ne(labels.shift()).cumsum()
    intervals = data_frame.groupby(change_points).agg(
        {"timestamp": ["first", "last"], "label": "first"}
    )
    intervals.columns = ["start", "end", "label"]

    # shade the different intervals with label
    for _, row in intervals.iterrows():
        # for each of the subplots
        for sub_ax in ax:
            sub_ax.axvspan(
                row["start"],
                row["end"],
                color=label_colors[row["label"]],
                alpha=0.2,  # Visibiulity
                zorder=0,
            )  # order of plots in the z-direction

    present_labels = intervals["label"].unique()
    patches = [
        mpatches.Patch(color=label_colors[label], label=label.capitalize(), alpha=0.3)
        for label in label_colors
        if label in present_labels
    ]
    # Add label legend to the figure
    fig.legend(
        handles=patches,
        title="Activity Labels",
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(0.85, 0.95),  # Adjusted to fit within the figure
        frameon=False,
    )

    # Add legends for sensor data to each subplot
    ax[0].legend(loc="upper left", fontsize=10)
    ax[1].legend(loc="upper left", fontsize=10)
    plt.tight_layout(
        rect=[0, 0, 0.85, 0.95]
    )  # Leave space on the right for the Activity Labels legend, nededs ot be in
    plt.savefig(plt_name + ".png", bbox_inches="tight")
    plt.close()
