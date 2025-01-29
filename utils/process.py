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



def add_missing_labels(data_frame):
    data_frame['label'] = data_frame['label'].fillna(method='ffill')
    return data_frame   


