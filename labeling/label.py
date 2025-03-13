import dash
from plotly import graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import io

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Initialize the Dash app
app = dash.Dash(__name__)

person_map = {
    "Julia": "julia",
    "Sara": "sara",
    "Jakob": "jakob",
    "Marten": "marten",
}

app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select a CSV File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=False,
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id="new-label",
                    options=[
                        {"label": "Falling", "value": "falling"},
                        {"label": "Walking", "value": "walking"},
                        {"label": "Running", "value": "running"},
                        {"label": "Sitting", "value": "sitting"},
                        {"label": "Standing", "value": "standing"},
                        {"label": "Laying", "value": "laying"},
                        {"label": "Recover", "value": "recover"},
                        {"label": "Delete Label", "value": "delete"},
                    ],
                    placeholder="Select a new label",
                ),
                dcc.Dropdown(
                    id="person-select",
                    options=[
                        {"label": x[0], "value": x[1]} for x in person_map.items()
                    ],
                    placeholder="Person",
                ),
                html.Button("Download CSV", id="btn-download"),
                dcc.Download(id="download-dataframe-csv"),
            ],
            style={
                "textAlign": "center",
                "margin": "10px",
                "display": "grid",
                "grid-template-columns": "2fr 1fr 100px",
            },
        ),
        dcc.Graph(
            id="line-plot",
            config={"scrollZoom": True},
            style={"height": "80vh"},
            figure={},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="accelerometer-3d-plot",
                            config={"scrollZoom": True},
                            style={"height": "80vh"},
                            figure={},
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="gyroscope-3d-plot",
                            config={"scrollZoom": True},
                            style={"height": "80vh"},
                            figure={},
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
            ]
        ),
    ]
)


def calculate_position_velocity(df):
    """Calculate position and velocity from acceleration data using integration."""
    if "acceleration_y" not in df.columns:
        return df

    # Ensure timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Set initial position based on labels if available
    initial_position = 0.0
    if "label" in df.columns:
        position_mapping = {
            "standing": 1,
            "walking": 1,
            "running": 1,
            "sitting": 0,
            "falling": -1,
            "laying": -1,
            "recovering": -1,
        }
        first_valid_label = (
            df["label"].dropna().iloc[0] if not df["label"].dropna().empty else None
        )
        initial_position = (
            position_mapping.get(str(first_valid_label).lower(), 0)
            if first_valid_label
            else 0
        )
        print(f"Initial position: {initial_position}")

    # Calculate time differences as a series (not stored in df)
    time_diffs = df["timestamp"].diff().dt.total_seconds()
    # Handle first value
    time_diffs.iloc[0] = time_diffs.iloc[1:].median() if len(df) > 1 else 0.01

    # Apply a moving average filter to smooth out the acceleration noise
    window_size = 5  # Adjust window size based on your data sampling rate
    acceleration_filtered = (
        df["acceleration_x"].rolling(window=window_size, center=True).mean()
    )

    # Fill NaN values at the edges due to rolling window
    acceleration_filtered = acceleration_filtered.fillna(df["acceleration_x"])

    # Remove gravity component and apply a stronger damping factor
    damping_factor = 0.05  # Much stronger damping to decrease sensitivity
    ay_adjusted = (
        acceleration_filtered - acceleration_filtered.loc[:20].mean()
    ) * damping_factor

    # Add a noise threshold filter to ignore small accelerations
    noise_threshold = 0.05  # Adjust based on your sensor's noise level
    ay_adjusted = ay_adjusted.apply(lambda x: 0 if abs(x) < noise_threshold else x)

    # Set initial values
    df["velocity"] = 0.0
    df["relative_position"] = initial_position

    # Increase velocity decay for stronger stability
    velocity_decay = 0.95  # Stronger decay to reduce drift

    # Apply integration step by step
    for i in range(1, len(df)):
        if i >= len(df) - 1:  # Skip the last row
            continue

        dt = time_diffs.iloc[i]
        if dt <= 0:
            dt = 0.01

        # Apply decay to velocity (stronger for near-zero velocities)
        prev_velocity = df["velocity"].iloc[i - 1] * velocity_decay
        if abs(prev_velocity) < 0.01:  # Extra damping for small velocities
            prev_velocity = 0

        # Calculate velocity with additional noise filtering
        if abs(ay_adjusted.iloc[i]) > noise_threshold / 5:  # Secondary noise filter
            avg_accel = (ay_adjusted.iloc[i - 1] + ay_adjusted.iloc[i]) / 2
            new_velocity = prev_velocity + avg_accel * dt
        else:
            new_velocity = (
                prev_velocity  # Maintain velocity if acceleration is just noise
            )

        # Clip velocity
        new_velocity = max(min(new_velocity, 20), -20)
        df.loc[df.index[i], "velocity"] = new_velocity

        # Calculate position with less sensitivity to velocity
        position_damping = 0.8  # Damping factor specifically for position updates
        avg_velocity = (prev_velocity + new_velocity) / 2 * position_damping
        new_position = df["relative_position"].iloc[i - 1] + avg_velocity * dt

        # Clip position to [-1, 1]
        new_position = max(min(new_position, 2), -2)
        df.loc[df.index[i], "relative_position"] = new_position

    return df


def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Calculate position and velocity using the dedicated function
    df = calculate_position_velocity(df)

    return df


def create_label_blocks(input):
    # df = df.dropna(subset=["label"])
    df = input.copy()
    df["prev_label"] = df["label"].shift(1)
    df["next_label"] = df["label"].shift(-1)

    label_blocks = df[
        ((df["label"] != df["prev_label"]) | (df["label"] != df["next_label"]))
        & pd.notnull(df["label"])
        | pd.isnull(df["label"])
        != pd.isnull(df["next_label"])
    ].copy()
    label_blocks.iloc[0, label_blocks.columns.get_loc("timestamp")] = df[
        "timestamp"
    ].iloc[0]
    label_blocks["start_time"] = label_blocks["timestamp"]
    label_blocks["end_time"] = (
        label_blocks["timestamp"].shift(-1).fillna(df["timestamp"].iloc[-1])
    )

    label_blocks = label_blocks[label_blocks["label"] != label_blocks["prev_label"]]
    label_blocks = label_blocks[["start_time", "end_time", "label"]]

    label_blocks["start_time"] = pd.to_datetime(label_blocks["start_time"])
    label_blocks["end_time"] = pd.to_datetime(label_blocks["end_time"])

    return label_blocks.reset_index(drop=True)


def standard_scale_features(df):
    features = [
        "acceleration_x",
        "acceleration_y",
        "acceleration_z",
        "gyroscope_x",
        "gyroscope_y",
        "gyroscope_z",
    ]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[features] = scaler.fit_transform(df[features])
    return df


# Global variables to store label blocks and edits
edit_blocks = pd.DataFrame()
edits = pd.DataFrame()


@app.callback(
    [
        Output("line-plot", "figure", allow_duplicate=True),
        Output("accelerometer-3d-plot", "figure"),
        Output("gyroscope-3d-plot", "figure"),
    ],
    [Input("upload-data", "contents")],
    prevent_initial_call=True,
)
def update_figure_on_upload(contents):
    global edits, edit_blocks
    edit_blocks = pd.DataFrame()
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update

    df = parse_contents(contents)
    edits = df.copy()  # Store the original DataFrame for edits

    df_scaled = standard_scale_features(df.copy())

    # Create main figure with both original sensor data and calculated position/velocity
    fig = px.line(
        df_scaled,
        x="timestamp",
        y=[
            "acceleration_x",
            "acceleration_y",
            "acceleration_z",
            "gyroscope_x",
            "gyroscope_y",
            "gyroscope_z",
        ],
        title="Sensor Data",
    )

    # Add relative position and velocity as separate traces
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["relative_position"],
            mode="lines",
            name="Relative Position",
            line=dict(width=2, color="green"),
        )
    )

    if "velocity" in df.columns:
        # Use a smaller scaling factor to better visualize velocity
        scaled_velocity = df["velocity"] / 10  # Adjust scaling for better visualization
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=scaled_velocity,
                mode="lines",
                name="Velocity (scaled)",
                line=dict(width=1.5, color="purple", dash="dash"),
            )
        )

    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(
        clickmode="select",
        dragmode="select",
        yaxis=dict(
            title="Sensor Values", showgrid=True, gridwidth=1, gridcolor="LightGray"
        ),
        xaxis=dict(
            title="Timestamp", showgrid=True, gridwidth=1, gridcolor="LightGray"
        ),
    )

    # Create a color map for labels
    color_map = {}

    if "label" in df.columns:
        # fill in weird gaps
        df["label"] = df["label"].fillna(df["label"].shift(1))

        label_blocks = create_label_blocks(df)
        print(label_blocks)
        edits = df.copy()  # Store the original DataFrame for edits
        color_map = {
            label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, label in enumerate(df["label"].dropna().unique())
        }
        for _, row in label_blocks.iterrows():
            if row["label"] is not None and not pd.isna(row["label"]):
                fig.add_vrect(
                    x0=row["start_time"],
                    x1=row["end_time"],
                    fillcolor=color_map[row["label"]],
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    annotation_text=row["label"],
                    annotation_position="top left",
                    annotation=dict(textangle=-90),
                )
    accel_fig = go.Figure()
    gyro_fig = go.Figure()

    # Create 3D scatter plot for accelerometer data with label-based coloring
    if "label" in df.columns:
        # Create a numeric encoding of labels for coloring
        unique_labels = df["label"].dropna().unique()
        label_to_num = {label: i for i, label in enumerate(unique_labels)}

        # Fill NA labels with a placeholder value for visualization
        df_plot = df.copy()
        df_plot["label_numeric"] = df_plot["label"].map(label_to_num)
        df_plot["label_numeric"] = df_plot["label_numeric"].fillna(
            -1
        )  # -1 for unlabeled

        # Create the accelerometer scatter plot with label coloring
        accel_fig = px.scatter_3d(
            df_plot,
            x="acceleration_y",
            y="acceleration_x",
            z="acceleration_z",
            color="label",
            color_discrete_map=color_map,
            title="Acceleration 3D Scatter",
            labels={"color": "Activity"},
            # opacity=0.2,
        )
        # accel_fig.update_traces(marker_size=2.5)

        # Create the gyroscope scatter plot with label coloring
        gyro_fig = px.scatter_3d(
            df_plot,
            x="gyroscope_y",
            y="gyroscope_x",
            z="gyroscope_z",
            color="label",
            color_discrete_map=color_map,
            title="Gyroscope 3D Scatter",
            labels={"color": "Activity"},
            opacity=0.2,
        )
        gyro_fig.update_traces(marker_size=2.5)
    return fig, accel_fig, gyro_fig


@app.callback(
    Output("line-plot", "figure", allow_duplicate=True),
    [Input("line-plot", "selectedData")],
    [State("new-label", "value"), State("line-plot", "figure")],
    prevent_initial_call=True,
)
def update_figure_on_label(selected_data, new_label, existing_figure):
    global edit_blocks, edits
    if existing_figure is None or new_label is None or selected_data is None:
        return dash.no_update

    if "range" in selected_data and selected_data["range"].get("x") is not None:
        start_time = selected_data["range"]["x"][0]
        end_time = selected_data["range"]["x"][1]

        if new_label == "delete":
            # Remove the label from the edits DataFrame
            edits.loc[
                (edits["timestamp"] >= start_time) & (edits["timestamp"] <= end_time),
                "label",
            ] = None
        else:
            # Add a row to the global label blocks
            new_row = pd.DataFrame(
                {
                    "start_time": [start_time],
                    "end_time": [end_time],
                    "label": [new_label],
                }
            )
            edit_blocks = pd.concat([edit_blocks, new_row], ignore_index=True)

            # Update the edits DataFrame
            edits.loc[
                (edits["timestamp"] >= start_time) & (edits["timestamp"] <= end_time),
                "label",
            ] = new_label

    if "rangeslider" in existing_figure["layout"]["xaxis"]:
        if "yaxis" in existing_figure["layout"]["xaxis"]["rangeslider"]:
            del existing_figure["layout"]["xaxis"]["rangeslider"]["yaxis"]

    # Reuse the existing figure
    fig = go.Figure(existing_figure)

    if not edit_blocks.empty:
        color_map = {
            label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, label in enumerate(edit_blocks["label"].unique())
        }
        fig.update_layout(shapes=[])

        for _, row in edit_blocks.iterrows():
            fig.add_vrect(
                x0=row["start_time"],
                x1=row["end_time"],
                fillcolor=color_map[row["label"]],
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text=row["label"],
                annotation_position="top left",
                annotation=dict(textangle=-90),
            )
    return fig


@app.callback(
    Output("download-dataframe-csv", "data"),
    [
        Input("btn-download", "n_clicks"),
        State("upload-data", "filename"),
        State("person-select", "value"),
    ],
    prevent_initial_call=True,
)
def download_csv(n_clicks, filename, selected_person):
    global edits

    export_df = edits.copy()
    export_df["prev_label_block"] = None

    label_blocks = create_label_blocks(edits)
    prev_label = None

    # Sort label_blocks by start_time
    label_blocks = label_blocks.sort_values(by="start_time")

    for _, row in label_blocks.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        label = row["label"]

        if label is None or pd.isna(label):
            duration = (end_time - start_time).total_seconds()
            # remove longer sections with no label
            if duration > 0.5:
                export_df = export_df[
                    ~(
                        (export_df["timestamp"] >= start_time)
                        & (export_df["timestamp"] <= end_time)
                    )
                ]
        else:
            # assign the previous label
            export_df.loc[
                (export_df["timestamp"] >= start_time)
                & (export_df["timestamp"] <= end_time),
                "prev_label_block",
            ] = prev_label
            prev_label = label

    # fill NaN labels with the next label to fix weird gaps
    export_df["label"] = export_df["label"].fillna(export_df["label"].shift(1))

    # Add the selected person column
    if selected_person:
        export_df["person"] = selected_person

    if filename:
        clean_filename = filename.rsplit(".", 1)[0] + "_clean.csv"
    else:
        clean_filename = "labeled_data_clean.csv"
    return dcc.send_data_frame(export_df.to_csv, clean_filename, index=False)


@app.callback(Output("new-label", "value"), [Input("new-label", "value")])
def update_label(new_label):
    return new_label


@app.callback(
    Output("person-select", "value"),
    [Input("upload-data", "filename")],
    prevent_initial_call=True,
)
def update_person_select(filename):
    global person_map
    if filename:
        for person in person_map.values():
            if person in filename.lower():
                return person
    return dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True)
