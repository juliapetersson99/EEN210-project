from collections import deque
import numpy as np
import pandas as pd


class RollingStats:
    def __init__(
        self,
        window_size: int,
        columns: list[str] = [
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ],
    ):
        self.columns = columns
        self.window_size = window_size
        self.data_deque = deque()

        # store stats for each column
        self.sum = np.zeros(len(columns))
        self.sum_sq = np.zeros(len(columns))
        self.min = np.zeros(len(columns))
        self.max = np.zeros(len(columns))

    def update(self, new_row: float):
        """
        Add a new data point into the rolling window.
        Remove the oldest data point if we're over capacity.
        """
        # Add new sample
        self.data_deque.append(new_row)
        self.sum += new_row
        self.sum_sq += new_row * new_row
        self.min = np.minimum(self.min, new_row)
        self.max = np.maximum(self.max, new_row)

        # Pop oldest if over window size
        if len(self.data_deque) > self.window_size:
            old_value = self.data_deque.popleft()
            self.sum -= old_value
            self.sum_sq -= old_value * old_value
            # Update min and max
            self.min = np.min(self.data_deque, axis=0)
            self.max = np.max(self.data_deque, axis=0)

    def mean(self) -> np.ndarray:
        """
        Returns the rolling mean of the current window.
        """
        current_size = len(self.data_deque)
        if current_size == 0:
            return 0.0
        return self.sum / current_size

    def mean_labeled(self) -> dict:
        """
        Returns a labeled rolling mean of the current window.
        """
        # Add column labels to the mean, add _mean suffix
        return {f"{col}_mean": mean for col, mean in zip(self.columns, self.mean())}

    def variance(self) -> np.ndarray:
        """
        Returns the rolling sample variance of the current window.
        """
        current_size = len(self.data_deque)
        if current_size < 2:
            return np.zeros(len(self.columns))
        # sample variance = (sum of x^2 - (sum of x)^2 / n ) / (n-1)
        return (self.sum_sq - (self.sum / current_size) ** 2) / (current_size - 1)

    def std(self) -> np.ndarray:
        """
        Returns a rolling standard deviation of the current window.
        """
        # small correction term for numerical stability (avoid sqrt of negative number)
        mu = 0.000001
        return (
            # sqrt of variance
            (self.variance() + mu).pow(1.0 / 2)
            if len(self.data_deque) > 1
            else np.zeros(len(self.columns))
        )

    def std_labeled(self) -> dict:
        """
        Returns a labeled rolling standard deviation of the current window.
        """
        # Add column labels to the std, add _std suffix
        return {f"{col}_std": std for col, std in zip(self.columns, self.std())}

    def min_labeled(self) -> dict:
        """
        Returns a labeled rolling minimum of the current window.
        """
        # Add column labels to the min, add _min suffix
        return {f"{col}_min": min_val for col, min_val in zip(self.columns, self.min)}

    def max_labeled(self) -> dict:
        """
        Returns a labeled rolling maximum of the current window.
        """
        # Add column labels to the max, add _max suffix
        return {f"{col}_max": max_val for col, max_val in zip(self.columns, self.max)}

    def size(self) -> int:
        """
        Returns the current size of the window.
        """
        return len(self.data_deque)

    def clear(self):
        """
        Clear the window.
        """
        self.data_deque.clear()
        self.sum = np.zeros(len(self.columns))
        self.sum_sq = np.zeros(len(self.columns))
        self.min = np.zeros(len(self.columns))
        self.max = np.zeros(len(self.columns))


class LabelMemory:
    """
    Class to keep track of the probabilities of previous labels in a rolling window.
    """

    def __init__(self, window_size: int, labels: list[str]):
        self.window_size = window_size
        self.labels = labels
        self.data_deque = deque()
        self.label_counts = {label: 0 for label in labels}

    def update(self, new_label: pd.Series):
        """
        Add a new label probability Series into the rolling window.
        Remove the oldest label if we're over capacity.
        """
        # Reindex the Series to ensure correct order
        new_label = new_label.reindex(self.labels)

        # Add new sample
        self.data_deque.append(new_label)

        # find the label with the highest probability
        max_label = new_label.idxmax()
        # update the counts
        self.label_counts[max_label] += 1

        # Pop oldest if over window size
        if len(self.data_deque) > self.window_size:
            old_value = self.data_deque.popleft()
            # remove the label with the highest probability
            max_label = old_value.idxmax()
            self.label_counts[max_label] -= 1

    def mode(self) -> str:
        """
        Returns the most common label of the current window.
        """
        # find the label with the highest count
        max_label = max(self.label_counts, key=self.label_counts.get)
        return max_label

    def averaged_current_label(self) -> pd.Series:
        """
        Returns the averaged probability of the current window with higher weighting the most recent one.
        """
        N = len(self.data_deque)
        if N == 0:
            return pd.Series({label: 0.0 for label in self.labels})

        # Calculate linearly weighted average of probabilities
        weights = 2 * np.arange(1, N + 1) / (N * (N + 1))

        # Convert deque of Series to DataFrame for weight calculation
        df = pd.DataFrame(list(self.data_deque))

        # Use pandas weighted average - multiply by weights and sum
        weighted_result = df.mul(weights, axis=0).sum()

        return weighted_result


class RollingStatsPandas:
    """
    Class to calculate rolling statistics using Pandas DataFrame.
    Provides the same functionality as RollingStats but with Pandas backend.
    """

    def __init__(
        self,
        window_size: int,
        columns: list[str] = [
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ],
    ):
        self.columns = columns
        self.window_size = window_size
        self.df = pd.DataFrame(columns=columns)

    def update(self, new_row: np.ndarray):
        """
        Add a new data point into the rolling window.
        Remove the oldest data point if we're over capacity.
        """
        # Add new sample as a row at the end of the DataFrame
        self.df.loc[len(self.df)] = new_row

        # Remove oldest row if over window size
        if len(self.df) > self.window_size:
            self.df = self.df.iloc[1:].reset_index(drop=True)

    def mean(self) -> np.ndarray:
        """
        Returns the rolling mean of the current window.
        """
        if len(self.df) == 0:
            return np.zeros(len(self.columns))
        return self.df.mean().values

    def mean_labeled(self) -> dict:
        """
        Returns a labeled rolling mean of the current window.
        """
        if len(self.df) == 0:
            return {f"{col}_mean": 0.0 for col in self.columns}
        means = self.df.mean()
        return {f"{col}_mean": val for col, val in means.items()}

    def variance(self) -> np.ndarray:
        """
        Returns the rolling sample variance of the current window.
        """
        if len(self.df) < 2:
            return np.zeros(len(self.columns))
        return self.df.var().values

    def std(self) -> np.ndarray:
        """
        Returns a rolling standard deviation of the current window.
        """
        if len(self.df) < 2:
            return np.zeros(len(self.columns))
        return self.df.std().values

    def std_labeled(self) -> dict:
        """
        Returns a labeled rolling standard deviation of the current window.
        """
        if len(self.df) < 2:
            return {f"{col}_std": 0.0 for col in self.columns}
        stds = self.df.std()
        return {f"{col}_std": val for col, val in stds.items()}

    def min_labeled(self) -> dict:
        """
        Returns a labeled rolling minimum of the current window.
        """
        if len(self.df) == 0:
            return {f"{col}_min": 0.0 for col in self.columns}
        mins = self.df.min()
        return {f"{col}_min": val for col, val in mins.items()}

    def max_labeled(self) -> dict:
        """
        Returns a labeled rolling maximum of the current window.
        """
        if len(self.df) == 0:
            return {f"{col}_max": 0.0 for col in self.columns}
        maxs = self.df.max()
        return {f"{col}_max": val for col, val in maxs.items()}

    def size(self) -> int:
        """
        Returns the current size of the window.
        """
        return len(self.df)

    def clear(self):
        """
        Clear the window.
        """
        self.df = pd.DataFrame(columns=self.columns)
