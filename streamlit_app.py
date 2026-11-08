from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "powerplant_data.csv"
MODEL_PATH = APP_DIR / "models" / "best_model.pt"
FEATURE_COLUMNS = ["AT", "V", "AP", "RH"]
FEATURE_LABELS = {
    "AT": "Ambient temperature (°C)",
    "V": "Exhaust vacuum (cm Hg)",
    "AP": "Ambient pressure (mbar)",
    "RH": "Relative humidity (%)",
}


class PowerPlantANN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)


@st.cache_data(show_spinner=False)
def load_training_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data was not found at {DATA_PATH}.")

    data = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in [*FEATURE_COLUMNS, "PE"] if column not in data]
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {', '.join(missing_columns)}.")
    return data


@st.cache_resource(show_spinner="Loading the trained model...")
def load_predictor() -> tuple[PowerPlantANN, StandardScaler]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model was not found at {MODEL_PATH}.")

    data = load_training_data()
    features = data.loc[:, FEATURE_COLUMNS]
    target = data["PE"]
    train_features, _test_features, _train_target, _test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )
    scaler = StandardScaler()
    scaler.fit(train_features)

    model = PowerPlantANN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model, scaler


def predict(
    model: PowerPlantANN,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> np.ndarray:
    scaled_values = scaler.transform(features.loc[:, FEATURE_COLUMNS]).astype(np.float32)
    with torch.inference_mode():
        predictions = model(torch.from_numpy(scaled_values))
    return predictions.squeeze(1).numpy()


st.set_page_config(
    page_title="Power plant output estimator",
    page_icon=":material/bolt:",
    layout="centered",
)

st.title("Power plant output estimator")
st.write(
    "Estimate net electrical output from ambient conditions using the trained "
    "artificial neural network."
)

try:
    training_data = load_training_data()
    model, scaler = load_predictor()
except (FileNotFoundError, ValueError, RuntimeError) as error:
    st.error(f"The prediction service could not be initialized: {error}")
    st.stop()

with st.container(border=True):
    st.subheader("Estimate one operating point")
    st.caption("Enter the four ambient measurements, then run the model.")

    with st.form("single_prediction"):
        first_row = st.columns(2)
        second_row = st.columns(2)
        input_columns = [first_row[0], first_row[1], second_row[0], second_row[1]]
        inputs: dict[str, float] = {}

        for column, feature in zip(input_columns, FEATURE_COLUMNS):
            with column:
                values = training_data[feature]
                inputs[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    value=float(values.median()),
                    step=0.1,
                    format="%.2f",
                    help=(
                        f"Training data range: {values.min():.2f} to "
                        f"{values.max():.2f}."
                    ),
                )

        submitted = st.form_submit_button(
            "Estimate output",
            type="primary",
            icon=":material/bolt:",
        )

    if submitted:
        input_data = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
        estimate = float(predict(model, scaler, input_data)[0])
        st.metric("Estimated net electrical output", f"{estimate:.2f} MW")
        outside_training_range = []
        for feature, value in inputs.items():
            minimum = training_data[feature].min()
            maximum = training_data[feature].max()
            if value < minimum or value > maximum:
                outside_training_range.append(
                    f"- **{FEATURE_LABELS[feature]}:** {value:.2f} "
                    f"(training range: {minimum:.2f} to {maximum:.2f})"
                )
        if outside_training_range:
            st.warning(
                "These inputs are outside the training data range, so the "
                "estimate may be less reliable:\n\n"
                + "\n".join(outside_training_range)
            )

st.subheader("Estimate from a CSV")
st.caption(
    "Upload a CSV with columns AT, V, AP, and RH. Additional columns are kept "
    "in the downloadable results."
)
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
    except (
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as error:
        st.error(f"The uploaded file could not be read as a CSV: {error}")
    else:
        missing_columns = [
            column for column in FEATURE_COLUMNS if column not in batch_data.columns
        ]
        if missing_columns:
            st.error(f"CSV is missing required columns: {', '.join(missing_columns)}.")
        elif batch_data.columns.duplicated().any():
            st.error("CSV column names must be unique.")
        elif batch_data.empty:
            st.error("The uploaded CSV has no rows to predict.")
        else:
            batch_features = batch_data.loc[:, FEATURE_COLUMNS].apply(
                pd.to_numeric,
                errors="coerce",
            )
            if batch_features.isna().any().any():
                st.error("All feature values must be numeric and non-empty.")
            elif not np.isfinite(batch_features.to_numpy(dtype=np.float64)).all():
                st.error("Feature values must be finite numbers.")
            else:
                results = batch_data.copy()
                results["Predicted PE (MW)"] = predict(
                    model,
                    scaler,
                    batch_features,
                )
                st.dataframe(results.head(100), hide_index=True)
                st.caption(f"Showing up to 100 of {len(results):,} predictions.")
                st.download_button(
                    "Download predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="power_plant_predictions.csv",
                    mime="text/csv",
                    icon=":material/download:",
                )

with st.expander("About this model"):
    st.write(
        "The app loads the best checkpoint produced by the regression notebook. "
        "It uses the same 80/20 split (random state 42) to recreate the "
        "training-only StandardScaler used before the model was trained."
    )
