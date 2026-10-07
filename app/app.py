# 1. Import required libraries

from pathlib import Path
import streamlit as st
import joblib as  jl
import pandas as pd
import numpy as np

Basedir = Path(__file__).resolve().parent.parent
MODEL_DIR = Basedir / "app"

# 2. Load the trained model and saved objects
model = jl.load(MODEL_DIR / "kmeans_model.pkl")
scaler= jl.load(MODEL_DIR / "scaler.pkl")
cluster_labels = jl.load(MODEL_DIR / "cluster_labels.pkl")

# 3. Page settings

st.set_page_config(page_title="Online Retail Customer Segmentation", page_icon="📊", layout="wide") 
st.title("Online Retail Customer Segmentation")
st.markdown("##### Identify customer segments using a trained K-Means model")
st.markdown("---")

# 4. Load the cleaned dataset
rfm = pd.read_csv(Basedir / "data" / "customer_rfm.csv", index_col="CustomerID")

# 5. User input for Customer ID
customer_id = st.number_input("Enter Customer ID", min_value=0, value=12347, step=1)

# 6. Display RFM metrics and predict customer segment
if st.button("Find Segment"):
    if float(customer_id) not in rfm.index:
        st.error("Customer ID not found.")

    else:
        row = rfm.loc[float(customer_id)]
        recency = int(row["Recency"])
        frequency = int(row["Frequency"])
        monetary = row["Monetary"]

        # Display RFM metrics
        col1, col2, col3=st.columns(3)
        col1.metric("Recency", f"{recency} days")
        col2.metric("Frequency", f"{frequency} orders")
        col3.metric("Monetary", f"£{monetary:,.2f}")

        #Prepare RFM values for the model
        rfm_values = np.array([recency,frequency,monetary]).reshape(1, -1)

        # Same preprocessing used during training 
        rfm_log = np.log1p(rfm_values)
        rfm_scaled = scaler.transform(rfm_log)

        # Predict cluster
        cluster = model.predict(rfm_scaled)[0]

        segment = cluster_labels[cluster]
        st.success(f"Customer Segment: {segment}")
        st.markdown("---")

        # 7. Display customer summary and recommendations
        summaries = {
        "High-Value / Loyal":
        f"Customer {customer_id} is recently active, frequently purchasing and has high spending, indicating strong customer value.",

        "Mid-Value / Potential":
        f"Customer {customer_id} shows moderate engagement and spending with potential to increase purchase frequency.",

        "Low-Value / At Risk":
        f"Customer {customer_id} has low purchase frequency, lower spending and a longer period since their last purchase."
        }
        st.markdown("### Customer Summary")
        st.write(summaries[segment])

        st.markdown("---")
        recommendations = {
        "High-Value / Loyal": [
            "Prioritize customer retention",
            "Offer loyalty rewards and exclusive benefits",
            "Use personalized product recommendations",
            "Provide early access of new products"
        ],

        "Mid-Value / Potential": [
            "Use personalized offers to encourage repeat purchases",
            "Promote relevant product bundles",
            "Use cross-selling to increase purchase frequency"
        ],

        "Low-Value / At Risk": [
            "Run targeted re-engagement campaigns",
            "Use win-back offers to encourage another purchase",
            "Use email reminders to bring customers back"
        ]
        }                
        
        st.subheader("Recommended Actions")
        for recommendation in recommendations[segment]:
            st.write(f"- {recommendation}")

          