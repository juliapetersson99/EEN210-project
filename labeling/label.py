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
    ]
)


def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))

    df["timestamp"] = pd.to_datetime(df["timestamp"])

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
    Output("line-plot", "figure", allow_duplicate=True),
    [Input("upload-data", "contents")],
    prevent_initial_call=True,
)
def update_figure_on_upload(contents):
    global edits, edit_blocks
    edit_blocks = pd.DataFrame()
    if contents is None:
        return dash.no_update

    df = parse_contents(contents)
    edits = df.copy()  # Store the original DataFrame for edits

    df = standard_scale_features(df)

    fig = px.line(
        df,
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

    if "label" in df.columns:
        label_blocks = create_label_blocks(df)
        print(label_blocks)
        edits = df.copy()  # Store the original DataFrame for edits
        color_map = {
            label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, label in enumerate(label_blocks["label"].unique())
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
    return fig


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
                "prev_label",
            ] = prev_label
            prev_label = label

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
