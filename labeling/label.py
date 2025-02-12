import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import io

import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# Initialize the Dash app
app = dash.Dash(__name__)

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
                    ],
                    placeholder="Select a new label",
                ),
            ],
            style={"textAlign": "center", "margin": "10px"},
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
    return df


def create_label_blocks(df):
    df = df.dropna(subset=["label"])
    df["prev_label"] = df["label"].shift(1)
    df["next_label"] = df["label"].shift(-1)

    label_blocks = df[
        (df["label"] != df["prev_label"]) | (df["label"] != df["next_label"])
    ].copy()
    label_blocks["start_time"] = label_blocks["timestamp"]
    label_blocks["end_time"] = (
        label_blocks["timestamp"].shift(-1).fillna(df["timestamp"].iloc[-1])
    )

    label_blocks = label_blocks[label_blocks["label"] != label_blocks["prev_label"]]
    label_blocks = label_blocks[["start_time", "end_time", "label"]]

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


@app.callback(
    Output("line-plot", "figure"),
    [Input("upload-data", "contents"), Input("line-plot", "selectedData")],
    [State("new-label", "value")],
)
def update_output(contents, relayout_data, new_label):
    if contents is None:
        return dash.no_update

    df = parse_contents(contents)
    df = standard_scale_features(df)

    print(relayout_data)

    if (
        new_label is not None
        and relayout_data
        and "range" in relayout_data
        and relayout_data["range"].get("x") is not None
    ):
        start_time = relayout_data["range"]["x"][0]
        end_time = relayout_data["range"]["x"][1]
        df.loc[
            (df["timestamp"] >= start_time) & (df["timestamp"] <= end_time), "label"
        ] = new_label

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
        color_map = {
            label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, label in enumerate(label_blocks["label"].unique())
        }
        for _, row in label_blocks.iterrows():
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


@app.callback(Output("new-label", "value"), [Input("new-label", "value")])
def update_label(new_label):
    return new_label


if __name__ == "__main__":
    app.run_server(debug=True)
