
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(
    page_title="FoodWise AI",
    page_icon="🍽️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "foodwise_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "foodwise_model.pkl")

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found. Please upload and save the dataset.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please train and save the model.")
    st.stop()

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    if "Month" not in data.columns:
        data["Month"] = data["Date"].dt.month

    if "Day_Number" not in data.columns:
        data["Day_Number"] = data["Date"].dt.day

    return data

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

features = [
    "Day_of_Week",
    "Is_Weekend",
    "Weather",
    "Temperature_C",
    "Special_Event",
    "Promotion_Active",
    "Estimated_Footfall",
    "Reservations",
    "Category",
    "Selling_Price_INR",
    "Previous_Day_Sales_Units",
    "7_Day_Average_Sales_Units",
    "Month",
    "Day_Number"
]

st.sidebar.title("🍽️ FoodWise AI")
st.sidebar.caption("Smart Food Demand & Waste Management")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home Dashboard",
        "AI Demand Prediction",
        "Smart Kitchen Planner",
        "Food Waste Intelligence",
        "Sales Analytics",
        "What-If Simulator",
        "AI Recommendations",
        "Download Reports"
    ]
)

st.title("🍽️ FoodWise AI")
st.caption("AI-Powered Food Demand Prediction and Waste Management System")

if page == "Home Dashboard":

    st.header("Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Demand", f"{df['Demand_Units'].sum():,.0f}")
    col3.metric("Food Waste", f"{df['Food_Waste_kg'].sum():,.2f} kg")
    col4.metric("Total Revenue", f"₹{df['Revenue_INR'].sum():,.0f}")

    st.subheader("Category-wise Demand")

    category_data = (
        df.groupby("Category")["Demand_Units"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        category_data,
        x="Category",
        y="Demand_Units",
        color="Category",
        title="Food Demand by Category"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Demand Trend")

    trend = (
        df.groupby("Date")["Demand_Units"]
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        trend,
        x="Date",
        y="Demand_Units",
        title="Daily Food Demand Trend"
    )

    st.plotly_chart(fig2, width="stretch")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), width="stretch")


elif page == "AI Demand Prediction":

    st.header("🤖 AI Food Demand Prediction")

    st.write("Enter business conditions to estimate expected food demand.")

    col1, col2 = st.columns(2)

    with col1:
        day = st.selectbox(
            "Day of the Week",
            sorted(df["Day_of_Week"].dropna().unique())
        )

        weather = st.selectbox(
            "Weather",
            sorted(df["Weather"].dropna().unique())
        )

        temperature = st.number_input(
            "Temperature (°C)", 0.0, 50.0, 28.0
        )

        special_event = st.selectbox(
            "Special Event",
            sorted(df["Special_Event"].dropna().unique())
        )

        category = st.selectbox(
            "Food Category",
            sorted(df["Category"].dropna().unique())
        )

        month = st.slider("Month", 1, 12, 6)
        day_number = st.slider("Day of Month", 1, 31, 15)

    with col2:
        is_weekend = st.selectbox("Is Weekend?", [0, 1])
        promotion = st.selectbox("Promotion Active?", [0, 1])

        footfall = st.number_input(
            "Estimated Footfall", min_value=0, value=150
        )

        reservations = st.number_input(
            "Reservations", min_value=0, value=30
        )

        price = st.number_input(
            "Selling Price (₹)", min_value=1.0, value=150.0
        )

        previous_sales = st.number_input(
            "Previous Day Sales", min_value=0, value=40
        )

        average_sales = st.number_input(
            "7-Day Average Sales", min_value=0.0, value=45.0
        )

    if st.button("Predict Food Demand", type="primary"):

        input_data = pd.DataFrame([{
            "Day_of_Week": day,
            "Is_Weekend": is_weekend,
            "Weather": weather,
            "Temperature_C": temperature,
            "Special_Event": special_event,
            "Promotion_Active": promotion,
            "Estimated_Footfall": footfall,
            "Reservations": reservations,
            "Category": category,
            "Selling_Price_INR": price,
            "Previous_Day_Sales_Units": previous_sales,
            "7_Day_Average_Sales_Units": average_sales,
            "Month": month,
            "Day_Number": day_number
        }], columns=features)

        prediction = max(0, int(round(model.predict(input_data)[0])))

        st.success("Demand prediction completed!")

        st.metric("Predicted Food Demand", f"{prediction} units")

        st.info(
            f"The model estimates approximately {prediction} units "
            f"of demand for the {category} category."
        )


elif page == "Smart Kitchen Planner":

    st.header("👨‍🍳 Smart Kitchen Planner")

    category = st.selectbox(
        "Select Food Category",
        sorted(df["Category"].dropna().unique()),
        key="kitchen_category"
    )

    demand = st.number_input(
        "Expected Demand (Units)",
        min_value=0,
        value=50
    )

    buffer = st.slider(
        "Safety Buffer (%)",
        0,
        30,
        10
    )

    if st.button("Calculate Preparation Quantity"):

        preparation = int(np.ceil(demand * (1 + buffer / 100)))

        col1, col2, col3 = st.columns(3)

        col1.metric("Expected Demand", f"{demand} units")
        col2.metric("Safety Buffer", f"{buffer}%")
        col3.metric("Recommended Preparation", f"{preparation} units")

        st.success(
            f"Recommended preparation for {category}: "
            f"{preparation} units."
        )


elif page == "Food Waste Intelligence":

    st.header("♻️ Food Waste Intelligence")

    total_waste = df["Food_Waste_kg"].sum()
    total_waste_cost = df["Waste_Cost_INR"].sum()

    col1, col2 = st.columns(2)

    col1.metric("Total Food Waste", f"{total_waste:.2f} kg")
    col2.metric("Total Waste Cost", f"₹{total_waste_cost:,.2f}")

    waste_data = (
        df.groupby("Category")
        .agg(
            Food_Waste_kg=("Food_Waste_kg", "sum"),
            Waste_Cost_INR=("Waste_Cost_INR", "sum")
        )
        .reset_index()
    )

    st.subheader("Category-wise Food Waste")

    fig = px.bar(
        waste_data,
        x="Category",
        y="Food_Waste_kg",
        color="Category",
        title="Food Waste by Category"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Waste Cost Analysis")

    fig2 = px.pie(
        waste_data,
        names="Category",
        values="Waste_Cost_INR",
        title="Category-wise Waste Cost"
    )

    st.plotly_chart(fig2, width="stretch")

    st.subheader("Detailed Waste Report")
    st.dataframe(waste_data, width="stretch")


elif page == "Sales Analytics":

    st.header("📊 Sales Analytics")

    total_revenue = df["Revenue_INR"].sum()
    total_demand = df["Demand_Units"].sum()
    average_revenue = df["Revenue_INR"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue", f"₹{total_revenue:,.2f}")
    col2.metric("Total Demand", f"{total_demand:,.0f} units")
    col3.metric("Average Revenue", f"₹{average_revenue:,.2f}")

    sales_data = (
        df.groupby("Category")
        .agg(
            Revenue_INR=("Revenue_INR", "sum"),
            Demand_Units=("Demand_Units", "sum")
        )
        .reset_index()
    )

    st.subheader("Revenue by Food Category")

    fig = px.bar(
        sales_data,
        x="Category",
        y="Revenue_INR",
        color="Category",
        title="Category-wise Revenue"
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Demand Distribution")

    fig2 = px.pie(
        sales_data,
        names="Category",
        values="Demand_Units",
        title="Demand Share by Category"
    )

    st.plotly_chart(fig2, width="stretch")

    st.subheader("Sales Performance Table")
    st.dataframe(sales_data, width="stretch")


elif page == "What-If Simulator":

    st.header("🔮 What-If Demand Simulator")

    st.write(
        "Compare model-based demand estimates under two "
        "different promotional scenarios."
    )

    category = st.selectbox(
        "Select Food Category",
        sorted(df["Category"].dropna().unique()),
        key="sim_category"
    )

    category_data = df[df["Category"] == category]

    base_row = category_data.sort_values("Date").iloc[-1]

    input_data = pd.DataFrame([{
        column: base_row[column] for column in features
    }], columns=features)

    without_promotion = input_data.copy()
    with_promotion = input_data.copy()

    without_promotion["Promotion_Active"] = 0
    with_promotion["Promotion_Active"] = 1

    if st.button("Run Simulation", type="primary"):

        demand_without = max(
            0, model.predict(without_promotion)[0]
        )

        demand_with = max(
            0, model.predict(with_promotion)[0]
        )

        difference = demand_with - demand_without

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Without Promotion",
            f"{demand_without:.0f} units"
        )

        col2.metric(
            "With Promotion",
            f"{demand_with:.0f} units"
        )

        col3.metric(
            "Demand Difference",
            f"{difference:+.0f} units"
        )

        comparison = pd.DataFrame({
            "Scenario": ["Without Promotion", "With Promotion"],
            "Predicted Demand": [demand_without, demand_with]
        })

        fig = px.bar(
            comparison,
            x="Scenario",
            y="Predicted Demand",
            color="Scenario",
            title="Promotion Scenario Comparison"
        )

        st.plotly_chart(fig, width="stretch")

        st.info(
            "These are model-based scenario estimates. "
            "They do not establish that promotions directly cause "
            "a change in demand."
        )


elif page == "AI Recommendations":

    st.header("💡 FoodWise AI Recommendations")

    waste_data = (
        df.groupby("Category")
        .agg(
            Waste=("Food_Waste_kg", "sum"),
            Waste_Cost=("Waste_Cost_INR", "sum"),
            Demand=("Demand_Units", "sum")
        )
        .reset_index()
    )

    highest_waste = waste_data.loc[waste_data["Waste"].idxmax()]
    highest_cost = waste_data.loc[waste_data["Waste_Cost"].idxmax()]
    highest_demand = waste_data.loc[waste_data["Demand"].idxmax()]

    st.subheader("Smart Business Insights")

    st.warning(
        f"{highest_waste['Category']} has the highest recorded "
        f"food waste: {highest_waste['Waste']:.2f} kg. "
        "Review its preparation quantities."
    )

    st.error(
        f"{highest_cost['Category']} has the highest waste cost: "
        f"₹{highest_cost['Waste_Cost']:.2f}. "
        "Monitor its ingredient usage."
    )

    st.success(
        f"{highest_demand['Category']} has the highest demand. "
        "Ensure sufficient preparation."
    )

    st.subheader("Category-wise Recommendations")

    for _, row in waste_data.iterrows():

        st.write(
            f"**{row['Category']}** | "
            f"Demand: {row['Demand']:.0f} units | "
            f"Waste: {row['Waste']:.2f} kg | "
            f"Waste Cost: ₹{row['Waste_Cost']:.2f}"
        )

        st.caption(
            "Recommendation: Review demand patterns and adjust "
            "preparation quantities according to expected demand."
        )


elif page == "Download Reports":

    st.header("📥 Download FoodWise Reports")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(20), width="stretch")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Complete Dataset (CSV)",
        data=csv_data,
        file_name="FoodWise_AI_Dataset.csv",
        mime="text/csv"
    )

    summary = (
        df.groupby("Category")
        .agg(
            Total_Demand=("Demand_Units", "sum"),
            Total_Waste=("Food_Waste_kg", "sum"),
            Total_Revenue=("Revenue_INR", "sum"),
            Total_Waste_Cost=("Waste_Cost_INR", "sum")
        )
        .reset_index()
    )

    summary_csv = summary.to_csv(index=False).encode("utf-8")

    st.subheader("Category Summary")

    st.dataframe(summary, width="stretch")

    st.download_button(
        label="Download Category Summary (CSV)",
        data=summary_csv,
        file_name="FoodWise_Category_Summary.csv",
        mime="text/csv"
    )

st.sidebar.divider()
st.sidebar.caption("FoodWise AI | Academic Integrated Project")
