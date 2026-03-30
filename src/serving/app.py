"""Gradio Marketing Manager Dashboard for Churn Prediction.

Four-tab interface:
  1. Import Data — load and preview customer CSV
  2. Batch Predict — score all customers, filter by risk
  3. Model Training — retrain and track via MLflow
  4. Single Prediction — per-customer prediction with SHAP explanations
"""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from src.config import settings
from src.data.loader import load_raw_data
from src.features.schema import CATEGORICAL_VALUES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-loaded pipeline (avoids crash on import if model artifacts missing)
# ---------------------------------------------------------------------------
_pipeline = None


def _get_pipeline():
    """Lazy-load the inference pipeline (which needs model artifacts)."""
    global _pipeline
    if _pipeline is None:
        from pipelines.inference_pipeline import InferencePipeline
        _pipeline = InferencePipeline()
    return _pipeline


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 1 — Import Data
# ═══════════════════════════════════════════════════════════════════════════

def import_data(csv_path: str):
    """Load raw data from the given CSV path.

    Returns (status_html, preview_df, summary_html, row_count_for_state).
    """
    try:
        df = load_raw_data(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        error_html = f"""
        <div style="padding:20px; background:linear-gradient(135deg,#ff416c,#ff4b2b);
                    border-radius:12px; color:white; text-align:center;">
            <h3 style="margin:0 0 8px">❌ Import Failed</h3>
            <p style="margin:0; opacity:.9">{exc}</p>
        </div>"""
        return error_html, None, "", None

    churn_rate = df["Churn"].map({"Yes": 1, "No": 0}).mean() * 100

    status_html = f"""
    <div style="padding:20px; background:linear-gradient(135deg,#11998e,#38ef7d);
                border-radius:12px; color:white; text-align:center;">
        <h3 style="margin:0 0 8px">✅ Data Loaded Successfully</h3>
        <p style="margin:0; opacity:.9">{len(df):,} customers · {len(df.columns)} columns</p>
    </div>"""

    summary_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap;">
        <div style="flex:1; min-width:160px; background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px; padding:20px; color:white; text-align:center;">
            <div style="font-size:2em; font-weight:700">{len(df):,}</div>
            <div style="opacity:.8; font-size:.85em">Total Customers</div>
        </div>
        <div style="flex:1; min-width:160px; background:linear-gradient(135deg,#f093fb,#f5576c);
                    border-radius:12px; padding:20px; color:white; text-align:center;">
            <div style="font-size:2em; font-weight:700">{churn_rate:.1f}%</div>
            <div style="opacity:.8; font-size:.85em">Churn Rate</div>
        </div>
        <div style="flex:1; min-width:160px; background:linear-gradient(135deg,#4facfe,#00f2fe);
                    border-radius:12px; padding:20px; color:white; text-align:center;">
            <div style="font-size:2em; font-weight:700">{len(df.columns)}</div>
            <div style="opacity:.8; font-size:.85em">Features</div>
        </div>
    </div>"""

    preview = df.head(25)
    return status_html, preview, summary_html, df


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 2 — Batch Predict
# ═══════════════════════════════════════════════════════════════════════════

def run_batch_prediction(stored_df, risk_filter: str):
    """Run batch prediction on the stored data."""
    if stored_df is None:
        return (
            '<div style="padding:20px; background:#2d2d2d; border-radius:12px; '
            'text-align:center; color:#ff9f43;">'
            "<h3>⚠️ No Data Loaded</h3>"
            "<p>Go to <b>Import Data</b> tab and load a CSV first.</p></div>",
            None,
            "",
        )

    pipeline = _get_pipeline()
    result_df = pipeline.predict_batch(stored_df)

    # Build display table with key columns
    display_cols = [
        "customerID", "risk_level", "churn_probability",
        "Contract", "tenure", "MonthlyCharges", "TotalCharges",
        "InternetService", "PaymentMethod",
    ]
    display_cols = [c for c in display_cols if c in result_df.columns]
    display_df = result_df[display_cols].copy()
    display_df["churn_probability"] = (display_df["churn_probability"] * 100).round(1)
    display_df = display_df.rename(columns={"churn_probability": "churn_prob_%"})
    display_df = display_df.sort_values("churn_prob_%", ascending=False)

    # Filter
    if risk_filter and risk_filter != "All":
        display_df = display_df[display_df["risk_level"] == risk_filter]

    # Risk distribution
    risk_counts = result_df["risk_level"].value_counts()
    high = int(risk_counts.get("High", 0))
    med = int(risk_counts.get("Medium", 0))
    low = int(risk_counts.get("Low", 0))

    stats_html = f"""
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-bottom:8px;">
        <div style="flex:1; min-width:140px; background:linear-gradient(135deg,#eb3349,#f45c43);
                    border-radius:12px; padding:16px; color:white; text-align:center;">
            <div style="font-size:1.8em; font-weight:700">{high}</div>
            <div style="opacity:.85; font-size:.85em">🔴 High Risk</div>
        </div>
        <div style="flex:1; min-width:140px; background:linear-gradient(135deg,#f7971e,#ffd200);
                    border-radius:12px; padding:16px; color:white; text-align:center;">
            <div style="font-size:1.8em; font-weight:700">{med}</div>
            <div style="opacity:.85; font-size:.85em">🟡 Medium Risk</div>
        </div>
        <div style="flex:1; min-width:140px; background:linear-gradient(135deg,#11998e,#38ef7d);
                    border-radius:12px; padding:16px; color:white; text-align:center;">
            <div style="font-size:1.8em; font-weight:700">{low}</div>
            <div style="opacity:.85; font-size:.85em">🟢 Low Risk</div>
        </div>
        <div style="flex:1; min-width:140px; background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px; padding:16px; color:white; text-align:center;">
            <div style="font-size:1.8em; font-weight:700">{len(result_df):,}</div>
            <div style="opacity:.85; font-size:.85em">📊 Total Scored</div>
        </div>
    </div>"""

    info = f"Showing **{len(display_df):,}** customers" + (
        f" (filtered: **{risk_filter}** risk)" if risk_filter and risk_filter != "All" else ""
    )

    return stats_html, display_df, info


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 3 — Model Training
# ═══════════════════════════════════════════════════════════════════════════

def train_new_model(csv_path: str, progress=gr.Progress()):
    """Trigger a full training pipeline run."""
    from pipelines.train_pipeline import run_with_callback

    log_lines: list[str] = []

    def _cb(msg: str):
        log_lines.append(msg)

    progress(0.0, desc="Starting training pipeline…")
    try:
        results = run_with_callback(data_path=csv_path, callback=_cb)
    except Exception as exc:
        log_lines.append(f"\n❌ ERROR: {exc}")
        return "\n".join(log_lines), ""

    progress(1.0, desc="Done!")

    result_html = f"""
    <div style="padding:20px; background:linear-gradient(135deg,#11998e,#38ef7d);
                border-radius:12px; color:white;">
        <h3 style="margin:0 0 12px">✅ Training Complete</h3>
        <table style="width:100%; color:white; font-size:0.95em;">
            <tr><td style="padding:4px 8px; opacity:.8">Run ID</td>
                <td style="padding:4px 8px; font-family:monospace">{results['run_id'][:12]}…</td></tr>
            <tr><td style="padding:4px 8px; opacity:.8">F1 Score</td>
                <td style="padding:4px 8px; font-weight:700">{results['f1']:.3f}</td></tr>
            <tr><td style="padding:4px 8px; opacity:.8">ROC-AUC</td>
                <td style="padding:4px 8px; font-weight:700">{results['roc_auc']:.3f}</td></tr>
            <tr><td style="padding:4px 8px; opacity:.8">Accuracy</td>
                <td style="padding:4px 8px; font-weight:700">{results['accuracy']:.3f}</td></tr>
            <tr><td style="padding:4px 8px; opacity:.8">Threshold</td>
                <td style="padding:4px 8px; font-weight:700">{results['threshold']:.2f}</td></tr>
            <tr><td style="padding:4px 8px; opacity:.8">Feature Store</td>
                <td style="padding:4px 8px">{results['feature_store_version']}</td></tr>
        </table>
    </div>"""

    return "\n".join(log_lines), result_html


# ═══════════════════════════════════════════════════════════════════════════
#  TAB 4 — Single Prediction with SHAP
# ═══════════════════════════════════════════════════════════════════════════

def predict_single(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service,
    online_security, online_backup, device_protection,
    tech_support, streaming_tv, streaming_movies,
    contract, paperless_billing, payment_method,
    monthly_charges, total_charges,
):
    """Run churn prediction with SHAP explanations."""
    customer: dict[str, Any] = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen else 0,
        "Partner": "Yes" if partner else "No",
        "Dependents": "Yes" if dependents else "No",
        "tenure": int(tenure),
        "PhoneService": "Yes" if phone_service else "No",
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": "Yes" if paperless_billing else "No",
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": str(total_charges),
    }

    pipeline = _get_pipeline()
    explained = pipeline.predict_single_with_explanation(customer)
    res = explained.result

    # --- Risk badge ---
    color_map = {"Low": ("#11998e", "#38ef7d"), "Medium": ("#f7971e", "#ffd200"), "High": ("#eb3349", "#f45c43")}
    icon_map = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    c1, c2 = color_map.get(res.risk_level, ("#667eea", "#764ba2"))
    icon = icon_map.get(res.risk_level, "⚪")

    result_html = f"""
    <div style="padding:24px; background:linear-gradient(135deg,{c1},{c2});
                border-radius:16px; color:white; text-align:center; margin-bottom:16px;">
        <div style="font-size:3em; margin-bottom:8px">{icon}</div>
        <div style="font-size:2.2em; font-weight:800">{res.churn_probability:.1%}</div>
        <div style="font-size:1.1em; opacity:.9; margin-top:4px">
            {res.risk_level} Risk · {"Will Churn" if res.churn_prediction else "Likely Stays"}
        </div>
        <div style="font-size:.8em; opacity:.7; margin-top:8px">threshold: {res.threshold:.0%}</div>
    </div>"""

    # --- SHAP contributions table (top 10) ---
    top = explained.feature_contributions[:10]
    shap_rows = ""
    for item in top:
        val = item["shap_value"]
        bar_color = "#f45c43" if val > 0 else "#38ef7d"
        direction = "↑ increases churn" if val > 0 else "↓ decreases churn"
        bar_width = min(abs(val) / 0.05 * 100, 100)  # scale for visibility

        # Make feature name human-readable
        feat_label = item["feature"].replace("_", " ").title()

        shap_rows += f"""
        <tr>
            <td style="padding:8px 12px; border-bottom:1px solid #333; font-weight:500">
                {feat_label}
            </td>
            <td style="padding:8px 12px; border-bottom:1px solid #333; font-family:monospace; font-size:.9em">
                {item['feature_value']:.2f}
            </td>
            <td style="padding:8px 12px; border-bottom:1px solid #333; width:40%">
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="background:{bar_color}; height:8px; border-radius:4px;
                                width:{bar_width:.0f}%; min-width:4px; transition:width 0.3s"></div>
                    <span style="font-size:.78em; opacity:.7; white-space:nowrap">{val:+.4f}</span>
                </div>
            </td>
            <td style="padding:8px 12px; border-bottom:1px solid #333; font-size:.82em; opacity:.85">
                {direction}
            </td>
        </tr>"""

    shap_html = f"""
    <div style="background:#1a1a2e; border-radius:12px; padding:20px; color:#e0e0e0;">
        <h3 style="margin:0 0 16px; color:white;">🔬 Why This Prediction?</h3>
        <p style="margin:0 0 12px; font-size:.9em; opacity:.7">
            Top factors influencing the churn prediction (SHAP analysis):
        </p>
        <table style="width:100%; border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:2px solid #444;">
                    <th style="padding:8px 12px; text-align:left; opacity:.7; font-weight:600">Feature</th>
                    <th style="padding:8px 12px; text-align:left; opacity:.7; font-weight:600">Value</th>
                    <th style="padding:8px 12px; text-align:left; opacity:.7; font-weight:600">Impact</th>
                    <th style="padding:8px 12px; text-align:left; opacity:.7; font-weight:600">Effect</th>
                </tr>
            </thead>
            <tbody>{shap_rows}</tbody>
        </table>
    </div>"""

    # --- Text summary of top 3 drivers ---
    drivers = explained.feature_contributions[:3]
    summary_parts = []
    for i, d in enumerate(drivers, 1):
        feat = d["feature"].replace("_", " ").title()
        if d["shap_value"] > 0:
            summary_parts.append(f"**{i}. {feat}** (value: {d['feature_value']:.2f}) — pushes toward churn")
        else:
            summary_parts.append(f"**{i}. {feat}** (value: {d['feature_value']:.2f}) — reduces churn risk")

    summary_md = "### 📋 Key Risk Drivers\n\n" + "\n\n".join(summary_parts)

    return result_html, shap_html, summary_md


# ═══════════════════════════════════════════════════════════════════════════
#  BUILD GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.gradio-container {
    max-width: 1200px !important;
    background: #0f0f1a !important;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95em !important;
    padding: 12px 24px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* Primary action buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.4) !important;
}

/* Secondary buttons */
.gr-button-secondary {
    border: 2px solid #667eea !important;
    color: #667eea !important;
    font-weight: 600 !important;
}

/* Dataframe styling */
.dataframe-container { border-radius: 12px !important; overflow: hidden !important; }

/* MLflow iframe container */
.mlflow-frame-wrap iframe {
    border-radius: 12px;
    border: 2px solid #333;
}

.header-banner {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 32px 24px;
    border-radius: 16px;
    margin-bottom: 8px;
    text-align: center;
}
"""

# --- Determine MLflow UI url for iframe ---
_mlflow_tracking = settings.mlflow.tracking_uri
if _mlflow_tracking.startswith("http"):
    MLFLOW_UI_URL = _mlflow_tracking
else:
    MLFLOW_UI_URL = "http://localhost:5000"


with gr.Blocks(
    title="Churn Manager — Marketing Dashboard",
) as app:

    # ── Header ──────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-banner">
        <h1 style="margin:0; color:white; font-size:2em; font-weight:800;
                   letter-spacing:-0.5px;">
            📊 Churn Prediction Dashboard
        </h1>
        <p style="margin:8px 0 0; color:rgba(255,255,255,.65); font-size:1.05em;">
            Marketing Manager Interface · Import Data · Predict · Train · Analyse
        </p>
    </div>
    """)

    # ── Shared state ────────────────────────────────────────────────────
    stored_data = gr.State(value=None)

    # ════════════════════════════════════════════════════════════════════
    with gr.Tabs() as tabs:

        # ── TAB 1: Import Data ──────────────────────────────────────────
        with gr.Tab("📂 Import Data", id="tab-import"):
            gr.Markdown("### Load customer data from CSV")
            with gr.Row():
                csv_path_input = gr.Textbox(
                    value=str(settings.data.raw_path),
                    label="CSV File Path",
                    placeholder="path/to/Telco-Customer-Churn.csv",
                    scale=4,
                    interactive=True,
                )
                import_btn = gr.Button("📥 Load Data", variant="primary", scale=1)

            import_status = gr.HTML("")
            import_summary = gr.HTML("")
            import_preview = gr.Dataframe(
                label="Data Preview (first 25 rows)",
                interactive=False,
                wrap=True,
            )

            import_btn.click(
                fn=import_data,
                inputs=[csv_path_input],
                outputs=[import_status, import_preview, import_summary, stored_data],
            )

        # ── TAB 2: Batch Predict ────────────────────────────────────────
        with gr.Tab("📊 Batch Predict", id="tab-batch"):
            gr.Markdown("### Predict which customers churn next quarter")
            with gr.Row():
                risk_dropdown = gr.Dropdown(
                    choices=["All", "High", "Medium", "Low"],
                    value="All",
                    label="Filter by Risk Level",
                    scale=2,
                )
                batch_btn = gr.Button("🚀 Run Batch Prediction", variant="primary", scale=1)

            batch_stats = gr.HTML("")
            batch_info = gr.Markdown("")
            batch_table = gr.Dataframe(
                label="Customer Churn Predictions",
                interactive=False,
                wrap=True,
            )

            batch_btn.click(
                fn=run_batch_prediction,
                inputs=[stored_data, risk_dropdown],
                outputs=[batch_stats, batch_table, batch_info],
            )
            # Also allow re-filtering without re-running prediction
            risk_dropdown.change(
                fn=run_batch_prediction,
                inputs=[stored_data, risk_dropdown],
                outputs=[batch_stats, batch_table, batch_info],
            )

        # ── TAB 3: Model Training ──────────────────────────────────────
        with gr.Tab("🏋️ Model Training", id="tab-train"):
            gr.Markdown("### Train a new model and track experiments")
            with gr.Row():
                train_csv_path = gr.Textbox(
                    value=str(settings.data.raw_path),
                    label="Training Data Path",
                    scale=4,
                    interactive=True,
                )
                train_btn = gr.Button("🏋️ Train New Model", variant="primary", scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    train_result = gr.HTML(
                        '<div style="padding:20px; background:#1a1a2e; border-radius:12px; '
                        'text-align:center; color:#888; font-size:.95em;">'
                        "No training run yet — click <b>Train New Model</b> to start.</div>"
                    )
                with gr.Column(scale=1):
                    train_logs = gr.Textbox(
                        label="Training Logs",
                        lines=14,
                        max_lines=30,
                        interactive=False,
                    )

            train_btn.click(
                fn=train_new_model,
                inputs=[train_csv_path],
                outputs=[train_logs, train_result],
            )

            gr.Markdown("---")
            gr.Markdown("### MLflow Experiment Tracker")
            gr.HTML(
                "<p style='opacity:0.8; font-size:0.95em;'>"
                "MLflow secures its web interface by blocking iframe embeds "
                "(<code>X-Frame-Options: SAMEORIGIN</code>). <br/>"
                "Click the button below to safely open the tracking UI in a new tab."
                "</p>"
            )
            gr.Button(
                "📈 Open MLflow UI (New Tab)",
                link=MLFLOW_UI_URL,
                variant="secondary",
                size="lg",
            )

        # ── TAB 4: Single Prediction ────────────────────────────────────
        with gr.Tab("🔍 Single Prediction", id="tab-single"):
            gr.Markdown("### Predict churn for a single customer")

            with gr.Row():
                # Column 1 — Demographics
                with gr.Column():
                    gr.Markdown("#### 👤 Demographics")
                    gender = gr.Dropdown(CATEGORICAL_VALUES["gender"], label="Gender", value="Male")
                    senior_citizen = gr.Checkbox(label="Senior Citizen (65+)", value=False)
                    partner = gr.Checkbox(label="Has Partner", value=False)
                    dependents = gr.Checkbox(label="Has Dependents", value=False)

                    gr.Markdown("#### 📞 Phone & Internet")
                    phone_service = gr.Checkbox(label="Phone Service", value=True)
                    multiple_lines = gr.Dropdown(
                        CATEGORICAL_VALUES["MultipleLines"], label="Multiple Lines", value="No"
                    )
                    internet_service = gr.Dropdown(
                        CATEGORICAL_VALUES["InternetService"], label="Internet Service", value="Fiber optic"
                    )

                # Column 2 — Add-ons
                with gr.Column():
                    gr.Markdown("#### 🛡️ Add-on Services")
                    online_security = gr.Dropdown(
                        CATEGORICAL_VALUES["OnlineSecurity"], label="Online Security", value="No"
                    )
                    online_backup = gr.Dropdown(
                        CATEGORICAL_VALUES["OnlineBackup"], label="Online Backup", value="No"
                    )
                    device_protection = gr.Dropdown(
                        CATEGORICAL_VALUES["DeviceProtection"], label="Device Protection", value="No"
                    )
                    tech_support = gr.Dropdown(
                        CATEGORICAL_VALUES["TechSupport"], label="Tech Support", value="No"
                    )
                    streaming_tv = gr.Dropdown(
                        CATEGORICAL_VALUES["StreamingTV"], label="Streaming TV", value="No"
                    )
                    streaming_movies = gr.Dropdown(
                        CATEGORICAL_VALUES["StreamingMovies"], label="Streaming Movies", value="No"
                    )

                # Column 3 — Billing
                with gr.Column():
                    gr.Markdown("#### 💳 Billing & Contract")
                    contract = gr.Dropdown(
                        CATEGORICAL_VALUES["Contract"], label="Contract", value="Month-to-month"
                    )
                    paperless_billing = gr.Checkbox(label="Paperless Billing", value=True)
                    payment_method = gr.Dropdown(
                        CATEGORICAL_VALUES["PaymentMethod"], label="Payment Method", value="Electronic check"
                    )
                    tenure = gr.Slider(0, 72, value=1, step=1, label="Tenure (months)")
                    monthly_charges = gr.Slider(18, 120, value=70.0, step=0.05, label="Monthly Charges ($)")
                    total_charges = gr.Number(value=70.0, label="Total Charges ($)")

            predict_btn = gr.Button("🔍 Predict Churn", variant="primary", size="lg")

            with gr.Row():
                single_result = gr.HTML(
                    '<div style="padding:24px; background:#1a1a2e; border-radius:16px; '
                    'text-align:center; color:#888;">Enter customer details above and click '
                    '<b>Predict Churn</b></div>'
                )

            with gr.Row():
                with gr.Column(scale=2):
                    shap_output = gr.HTML("")
                with gr.Column(scale=1):
                    driver_summary = gr.Markdown("")

            predict_btn.click(
                fn=predict_single,
                inputs=[
                    gender, senior_citizen, partner, dependents, tenure,
                    phone_service, multiple_lines, internet_service,
                    online_security, online_backup, device_protection,
                    tech_support, streaming_tv, streaming_movies,
                    contract, paperless_billing, payment_method,
                    monthly_charges, total_charges,
                ],
                outputs=[single_result, shap_output, driver_summary],
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def main():
    app.launch(
        server_name=settings.serving.host,
        server_port=settings.serving.port,
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.gray,
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
