import  matplotlib.pyplot as plt
import pandas as pd

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

    colors = ['b', 'g', 'r']
    styles = ['-', '--', '-.']
    index_x= ['x', 'y', 'z']


    X = data_frame['timestamp']
    gyroscope = data_frame[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']]
    accelerometer = data_frame[['acceleration_x', 'acceleration_y', 'acceleration_z']]


    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle('Full sequence plot')
    ax[0].set_title('Gyroscope')
    ax[1].set_title('Accelerometer')
    ax[0].set_xlabel('Time')
    ax[1].set_xlabel('Time')
    ax[0].set_ylabel('dps (degrees per second)')
    ax[1].set_ylabel('m/s^2 (g)')
    
    for i in range(3):
        ax[0].plot(X, gyroscope.iloc[:, i], color=colors[i], linestyle=styles[i], label='gyroscope_' + index_x[i])
        ax[1].plot(X, accelerometer.iloc[:, i], color=colors[i], linestyle=styles[i], label='accelerometer_' + index_x[i])

    ax[0].legend()
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('plots/full_sequence_numpy.png')
    plt.close()
    











   
