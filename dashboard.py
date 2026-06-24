import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import torch

st.set_page_config(page_title="LLM Router Dashboard", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Model Results", "Dataset Explorer"])

def load_metrics(mode, seed=42):
    path = f"checkpoints/metrics_{mode}_seed{seed}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def load_history_from_ckpt(mode, seed=42):
    path = f"checkpoints/mf_{mode}_seed{seed}.pt"
    if os.path.exists(path):
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            return ckpt.get("history", [])
        except Exception as e:
            st.error(f"Error loading checkpoint: {e}")
    return []

if page == "Model Results":
    st.title("📊 Model Results")
    
    mode = st.radio("Select Routing Mode", ["binary", "multi"])
    
    metrics = load_metrics(mode)
    history = load_history_from_ckpt(mode)
    
    if metrics:
        st.subheader("Overall Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Accuracy", f"{metrics.get('avg_acc', 0):.4f}")
        col2.metric("Gain@Random", f"{metrics.get('gain_random', 0):.4f}")
        col3.metric("Gap@Oracle", f"{metrics.get('gap_oracle', 0):.4f}")
        if 'routing_acc' in metrics:
            col4.metric("Routing Accuracy", f"{metrics.get('routing_acc', 0):.4f}")
            
        st.subheader("Per-Dataset Breakdown")
        df_breakdown = pd.DataFrame(metrics.get("breakdown", []))
        st.dataframe(df_breakdown, use_container_width=True)
    else:
        st.warning("Metrics file not found. Please wait for `python run.py` to finish or run it manually.")
        
    if history:
        st.subheader("Training History")
        df_hist = pd.DataFrame(history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['train_loss'], name="Train Loss"))
        fig.add_trace(go.Scatter(x=df_hist.index, y=df_hist['test_loss'], name="Test Loss"))
        fig.update_layout(title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
        
elif page == "Dataset Explorer":
    st.title("🗂 Dataset Explorer")
    bench_dir = "bench-release"
    if not os.path.exists(bench_dir):
        st.warning(f"Directory '{bench_dir}' not found.")
    else:
        datasets = sorted([d for d in os.listdir(bench_dir) if os.path.isdir(os.path.join(bench_dir, d))])
        if not datasets:
            st.info("No datasets found.")
        else:
            selected_dataset = st.selectbox("Select Dataset", datasets)
            
            test_dir = os.path.join(bench_dir, selected_dataset, "test")
            if os.path.exists(test_dir):
                models = sorted([m for m in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, m))])
                if not models:
                    st.info("No models found for this dataset.")
                else:
                    selected_model = st.selectbox("Select Model", models)
                    
                    model_dir = os.path.join(test_dir, selected_model)
                    files = [f for f in os.listdir(model_dir) if f.endswith(".json")]
                    if files:
                        try:
                            with open(os.path.join(model_dir, files[0]), "r", encoding="utf-8") as f:
                                # Data could be a list of dicts, or jsonlines
                                content = f.read().strip()
                                if content.startswith('['):
                                    data = json.loads(content)
                                else:
                                    data = [json.loads(line) for line in content.split('\n') if line]
                                    
                            st.write(f"Loaded **{len(data)}** queries from `{files[0]}`")
                            
                            if data:
                                df = pd.DataFrame(data)
                                st.dataframe(df, use_container_width=True)
                                
                                st.subheader("Query Preview")
                                sample_idx = st.selectbox("Select Query to Preview (Index)", range(len(df)))
                                st.json(df.iloc[sample_idx].to_dict())
                        except Exception as e:
                            st.error(f"Error loading JSON: {e}")
                    else:
                        st.info("No JSON files found in the model directory.")
            else:
                st.info("No 'test' directory found for this dataset.")
