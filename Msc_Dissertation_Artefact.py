"""
NASA RUL PREDICTIVE MAINTENANCE DASHBOARD
=========================================
"""

# IMPORT THE TOOLS (LIBRARIES) WE NEED.

import streamlit as st
import pandas as pd
import numpy as np   
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# LOAD THE RAW DATA FROM FILES
# @st.cache_data tells Streamlit: "if this function is called again
# with the same inputs, don't redo the work - just reuse the answer
# from last time." This makes the app faster.

@st.cache_data
def load_data(fd_type="FD001", data_dir=""):

    # Read the training data, the test data, and the "true answers"
    # for the test data (the real remaining useful life, RUL).
    train_data = pd.read_csv(f"{data_dir}train_{fd_type}.txt", sep=r"\s+", header=None)
    test_data = pd.read_csv(f"{data_dir}test_{fd_type}.txt", sep=r"\s+", header=None)
    true_rul_data = pd.read_csv(f"{data_dir}RUL_{fd_type}.txt", header=None)

    column_names = (
        ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    train_data.columns = column_names
    test_data.columns = column_names
    true_rul_data.columns = ["true_RUL"]

    return train_data, test_data, true_rul_data


# CREATE THE TARGET WE WANT THE MODEL TO PREDICT

def create_target(train_data, failure_window=30):
    # For each engine, find the last cycle it was seen in the training data. Then subtract the current cycle from that to get the
    # remaining useful life (RUL) for each row. This is the number of cycles left before the engine fails.
    last_cycle_per_engine = train_data.groupby("engine_id")["cycle"].transform("max")
    train_data["RUL"] = last_cycle_per_engine - train_data["cycle"]

    # If an engine has 30 or fewer cycles left, we label it as
    # "about to fail" (1). Otherwise it's "healthy for now" (0).
    train_data["failure"] = (train_data["RUL"] <= failure_window).astype(int)

    return train_data


# FEATURE ENGINEERING

def create_features(data):

    sensor_columns = [f"sensor_{i}" for i in range(1, 22)]

    for sensor in sensor_columns:

        # Rolling average of the last 5 readings for this sensor,
        # calculated separately for each engine.
        rolling_mean_column = f"{sensor}_rm"
        data[rolling_mean_column] = (
            data.groupby("engine_id")[sensor]
            .rolling(window=5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Rolling standard deviation (spread/variation) of the last
        # 5 readings, also per engine.
        rolling_std_column = f"{sensor}_rs"
        data[rolling_std_column] = (
            data.groupby("engine_id")[sensor]
            .rolling(window=5, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    # The very first reading of each engine has no "standard
    # deviation" yet (there's nothing to compare it to), so it
    # comes out blank. We fill any blanks with 0.
    data.fillna(0, inplace=True)

    return data



#  TRAIN THE TWO MODELS

def train_models(train_data):

    feature_columns = [
        column for column in train_data.columns
        if "sensor" in column or "op_setting" in column
    ]

    X = train_data[feature_columns]     # the inputs
    y = train_data["failure"]           # the answer we want to predict

    # Machine learning models work best when all the numbers are on
    # a similar scale. StandardScaler rescales every column so it
    # has a mean of 0 and a standard deviation of 1.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # MODEL 1: Decision Tree

    decision_tree_model = DecisionTreeClassifier(
        random_state=42,           # makes results repeatable
        class_weight="balanced"
    )
    decision_tree_model.fit(X_scaled, y)

    # MODEL 2: Random Forest

    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    random_forest_model.fit(X_scaled, y)

    # We store both trained models in a dictionary so we can loop
    # over them easily later.
    trained_models = {
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }

    return trained_models, scaler, feature_columns


# DRAW A CONFUSION MATRIX

def plot_confusion_matrix(matrix, title="Confusion Matrix"):

    figure, axis = plt.subplots()

    sns.heatmap(
        matrix,
        annot=True,      # write the numbers on the chart
        fmt="d",          # show them as whole numbers
        cmap="Blues",
        cbar=False,
        ax=axis
    )

    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(title)

    st.pyplot(figure)


# EXPLAIN THE CONFUSION MATRIX IN PLAIN ENGLISH

def explain_confusion_matrix(matrix):

    # A confusion matrix always unpacks into these 4 numbers, in
    # this order:
    true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

    st.subheader(" Confusion Matrix Breakdown")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("True Negatives (TN)", true_negatives)
    col2.metric("False Positives (FP)", false_positives)
    col3.metric("False Negatives (FN)", false_negatives)
    col4.metric("True Positives (TP)", true_positives)

    st.markdown("""
    ### Interpretation

    - **True Negatives (TN)** = correctly predicted NO failure
      → Engine was healthy and model confirmed it is safe.

    - **False Positives (FP)** = false alarm
      → Model predicted failure, but engine was actually fine.
      → Leads to unnecessary maintenance cost.

    - **False Negatives (FN)** = missed failure ⚠️
      → Model predicted NO failure, but engine actually failed.
      → Most dangerous error in predictive maintenance.

    - **True Positives (TP)** = correctly detected failure
      → Model successfully identified failing engine.
    """)


# SCORE ONE TRAINED MODEL AGAINST THE TEST DATA
# This takes one trained model, makes predictions on the unseen
# test engines, and works out how good those predictions were.

def evaluate_model(model, X_test_scaled, test_data, true_rul_data, feature_columns):

    # Ask the model: for every row, do you think this is a failure
    # (0 or 1), and how confident are you (a probability from 0-1)?
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    test_data = test_data.copy()
    test_data["pred"] = predictions
    test_data["prob"] = probabilities

    # Each engine has many rows (one per cycle). We say an engine is
    # predicted "failing" if ANY of its rows were predicted failing
    # (that's what .max() does here, since predictions are 0 or 1).
    engine_level_prediction = test_data.groupby("engine_id")["pred"].max()
    engine_level_probability = test_data.groupby("engine_id")["prob"].max()

    results = pd.DataFrame({
        "engine_id": engine_level_prediction.index,
        "predicted_failure": engine_level_prediction.values,
        "max_prob": engine_level_probability.values
    })

    # Attach the REAL remaining useful life for each engine, so we
    # can check our predictions against the truth.
    true_rul_data = true_rul_data.copy()
    true_rul_data["engine_id"] = np.arange(1, len(true_rul_data) + 1)

    results = results.merge(true_rul_data, on="engine_id")
    results["true_failure"] = (results["true_RUL"] <= 30).astype(int)

    # Standard scikit-learn scoring tools:
    report = classification_report(
        results["true_failure"],
        results["predicted_failure"],
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        results["true_failure"],
        results["predicted_failure"]
    )

    # AUC needs both classes (failure and non-failure) to be present
    # to be calculated, so we guard against an error if that's not
    # the case.
    try:
        auc_score = roc_auc_score(results["true_failure"], results["max_prob"])
    except Exception:
        auc_score = None

    return results, report, matrix, auc_score


# RUN THE FULL PIPELINE FOR ONE DATASET (e.g. FD001)
# This ties together steps 2-8 for a single chosen sub-dataset,
# training AND scoring both models.

def run_pipeline(fd_type):

    train_data, test_data, true_rul_data = load_data(fd_type)

    train_data = create_target(train_data)
    train_data = create_features(train_data)

    test_data = create_features(test_data)

    # Train both models on the training data.
    trained_models, scaler, feature_columns = train_models(train_data)

    X_test_scaled = scaler.transform(test_data[feature_columns])

    all_outputs = {}
    for model_name, model in trained_models.items():
        results, report, matrix, auc_score = evaluate_model(
            model, X_test_scaled, test_data, true_rul_data, feature_columns
        )
        all_outputs[model_name] = {
            "results": results,
            "report": report,
            "cm": matrix,
            "auc": auc_score
        }

    return all_outputs, trained_models, feature_columns


#  BUILD A SUMMARY TABLE ACROSS ALL 4 DATASETS

# This repeats the whole pipeline for FD001, FD002, FD003 and FD004,
# for BOTH models, and puts one row of key numbers per combination
# into a single summary table.

def compute_summary():

    dataset_names = ["FD001", "FD002", "FD003", "FD004"]
    summary_rows = []

    progress_bar = st.progress(0, text="Generating summary...")
    status_text = st.empty()

    for step_number, fd_type in enumerate(dataset_names):
        status_text.info(f"Processing {fd_type}...")

        all_outputs, _, _ = run_pipeline(fd_type)

        for model_name, output in all_outputs.items():
            results = output["results"]
            matrix = np.array(output["cm"])

            actual_values = results["true_failure"]
            predicted_values = results["predicted_failure"]

            actual_failure_rate = actual_values.mean() * 100
            predicted_failure_rate = predicted_values.mean() * 100
            rate_difference = predicted_failure_rate - actual_failure_rate

            true_negatives, false_positives, false_negatives, true_positives = matrix.ravel()

            total_predictions = true_positives + true_negatives + false_positives + false_negatives
            accuracy = (
                (true_positives + true_negatives) / total_predictions
                if total_predictions > 0 else 0.0
            )

            false_negative_rate = (
                false_negatives / (false_negatives + true_positives)
                if (false_negatives + true_positives) > 0 else 0.0
            )
            false_positive_rate = (
                false_positives / (false_positives + true_negatives)
                if (false_positives + true_negatives) > 0 else 0.0
            )

            summary_rows.append({
                "Dataset": fd_type,
                "Model": model_name,
                "Accuracy": f"{accuracy:.3f}",
                "AUC": f"{output['auc']:.3f}" if output["auc"] is not None else "N/A",
                "Actual failure rate": f"{actual_failure_rate:.2f}%",
                "Predicted failure rate": f"{predicted_failure_rate:.2f}%",
                "Deviation (pp)": f"{rate_difference:.2f}",
                "FN rate": f"{false_negative_rate:.2%}",
                "FP rate": f"{false_positive_rate:.2%}"
            })

        progress_bar.progress((step_number + 1) / len(dataset_names), text=f"Processed {fd_type}")

    status_text.empty()
    progress_bar.empty()

    return pd.DataFrame(summary_rows)

# THE ACTUAL WEBPAGE (STREAMLIT UI)

# Everything above this line just DEFINES functions - none of it
# runs yet. From here down is where the app actually starts doing
# things and drawing the page.

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("🔧 Predictive Maintenance Dashboard (NASA CMAPSS)")
st.caption(
    "Compares a single Decision Tree against a Random Forest ensemble, "
    "trained on identical engineered features, so the effect of "
    "ensembling can be read directly from the results below."
)

# --- Sidebar controls (the panel on the left) ---
st.sidebar.header("Controls")

selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    ["FD001", "FD002", "FD003", "FD004"]
)

run_button_clicked = st.sidebar.button("🚀 Run Pipeline (Decision Tree vs Random Forest)")
summary_button_clicked = st.sidebar.button("📊 Generate Summary Table (All Datasets, Both Models)")


# --- What happens when the "Run Pipeline" button is clicked ---
if run_button_clicked:

    with st.spinner(f"Training Decision Tree and Random Forest for {selected_dataset}..."):
        try:
            all_outputs, trained_models, feature_columns = run_pipeline(selected_dataset)
            st.success("Pipeline completed successfully for both models!")

            # Show the two models' headline numbers side by side.
            st.subheader("⚖️ Decision Tree vs Random Forest — Head-to-Head")
            comparison_columns = st.columns(2)

            for column, model_name in zip(comparison_columns, all_outputs.keys()):
                output = all_outputs[model_name]
                report = output["report"]

                with column:
                    st.markdown(f"#### {model_name}")
                    st.metric("Accuracy", f"{report['accuracy']:.3f}")
                    st.metric("AUC", f"{output['auc']:.3f}" if output["auc"] is not None else "N/A")
                    st.metric("Recall (failure class)", f"{report.get('1', {}).get('recall', 0):.3f}")
                    st.metric("Precision (failure class)", f"{report.get('1', {}).get('precision', 0):.3f}")

            # Show the full detail for each model, one after another.
            for model_name, output in all_outputs.items():
                st.markdown("---")

                st.subheader(f"📊 {model_name} — Classification Report")
                st.json(output["report"])

                st.subheader(f"📉 {model_name} — Confusion Matrix")
                plot_confusion_matrix(np.array(output["cm"]), title=f"{model_name} Confusion Matrix")
                explain_confusion_matrix(np.array(output["cm"]))

                st.subheader(f"📦 {model_name} — Prediction Results")
                st.dataframe(output["results"])

                csv_data = output["results"].to_csv(index=False).encode("utf-8")
                st.download_button(
                    f"⬇ Download {model_name} Predictions",
                    csv_data,
                    f"predictions_{model_name.replace(' ', '_')}.csv",
                    "text/csv",
                    key=f"dl_{model_name}"
                )

        except Exception as error:
            st.error(f"❌ Error: {error}")


# --- What happens when the "Generate Summary Table" button is clicked ---
if summary_button_clicked:

    try:
        summary_table = compute_summary()
        st.success("Summary generated successfully!")

        st.subheader("📊 Summary Table Across All Datasets and Both Models")
        st.dataframe(summary_table, use_container_width=True)

        summary_csv = summary_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Summary",
            data=summary_csv,
            file_name="summary_table_dt_vs_rf.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"❌ Error generating summary: {error}")
