# import streamlit as st
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# # Preloaded dataset
# @st.cache_data
# def load_data():
#     # Replace with your dataset file path or DataFrame
#     file_path = 'comprehensive_mutual_funds_data.csv'
#     data = pd.read_csv(file_path)
#     return data

# # Train and predict function
# def predict_funds(data, timeframe, risk_level):
#     # Filter required columns
#     required_columns = ['returns_1yr', 'returns_3yr', 'returns_5yr', 'risk_level', 'scheme_name', 'sharpe', 'expense_ratio', 'fund_size_cr']
#     filtered_data = data.dropna(subset=required_columns).reset_index(drop=True)

#     # Define features and target
#     X = filtered_data[['returns_1yr', 'returns_3yr', 'returns_5yr', 'risk_level']]
#     y = filtered_data[timeframe]

#     # Train-Test Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Train Random Forest Model
#     rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     rf_model.fit(X_train, y_train)

#     # Predict on test set
#     y_pred = rf_model.predict(X_test)

#     # Add predictions to the test dataset for ranking
#     test_results = X_test.copy()
#     test_results['Predicted Returns'] = y_pred
#     test_results = test_results.reset_index()
#     test_results['scheme_name'] = filtered_data.loc[test_results['index'], 'scheme_name'].values
#     test_results['sharpe'] = filtered_data.loc[test_results['index'], 'sharpe'].values
#     test_results['expense_ratio'] = filtered_data.loc[test_results['index'], 'expense_ratio'].values
#     test_results['fund_size_cr'] = filtered_data.loc[test_results['index'], 'fund_size_cr'].values

#     # Rank funds by predicted returns, breaking ties using expense ratio
#     ranked_results = test_results.sort_values(
#         by=['Predicted Returns', 'expense_ratio'], ascending=[False, True]
#     ).head(3)

#     return ranked_results[['scheme_name', 'Predicted Returns', 'sharpe', 'expense_ratio', 'fund_size_cr']]

# # Streamlit App
# st.title("Mutual Fund Prediction Dashboard")
# st.subheader("Analyze and Predict Mutual Fund Returns Based on Historical Data")

# # Load the preloaded dataset
# data = load_data()

# # User Inputs
# st.sidebar.header("User Inputs")
# timeframe = st.sidebar.selectbox("Select Investment Timeframe", options=["1 Year", "3 Years", "5 Years"])
# risk_level = st.sidebar.slider("Select Risk Level", min_value=1, max_value=6, value=3)

# # Map timeframe to column names
# timeframe_map = {
#     "1 Year": "returns_1yr",
#     "3 Years": "returns_3yr",
#     "5 Years": "returns_5yr"
# }
# selected_timeframe = timeframe_map[timeframe]

# # Predict and Display Results
# if st.sidebar.button("Run Prediction"):
#     with st.spinner("Predicting top mutual funds..."):
#         top_funds = predict_funds(data, selected_timeframe, risk_level)

#     st.success("Prediction Complete!")
#     st.write("### Top 3 Mutual Funds Based on Predicted Returns")
#     st.dataframe(top_funds)


import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Preloaded dataset
@st.cache_data
def load_data():
    file_path = 'comprehensive_mutual_funds_data.csv'
    data = pd.read_csv(file_path)
    return data

# Train and predict function
def predict_funds(data, timeframe, risk_level):
    required_columns = ['returns_1yr', 'returns_3yr', 'returns_5yr', 'risk_level', 'scheme_name', 'sharpe', 'expense_ratio', 'fund_size_cr']
    filtered_data = data.dropna(subset=required_columns).reset_index(drop=True)

    X = filtered_data[['returns_1yr', 'returns_3yr', 'returns_5yr', 'risk_level']]
    y = filtered_data[timeframe]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    test_results = X_test.copy()
    test_results['Predicted Returns'] = y_pred
    test_results = test_results.reset_index()
    test_results['scheme_name'] = filtered_data.loc[test_results['index'], 'scheme_name'].values
    test_results['sharpe'] = filtered_data.loc[test_results['index'], 'sharpe'].values
    test_results['expense_ratio'] = filtered_data.loc[test_results['index'], 'expense_ratio'].values
    test_results['fund_size_cr'] = filtered_data.loc[test_results['index'], 'fund_size_cr'].values
    test_results.drop(columns=['index'], inplace=True)

    ranked_results = test_results.sort_values(
        by=['Predicted Returns', 'expense_ratio'], ascending=[False, True]
    ).head(3)

    return ranked_results[['scheme_name', 'Predicted Returns', 'sharpe', 'expense_ratio', 'fund_size_cr']]

# Streamlit App
st.set_page_config(
    page_title="Mutual Fund Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📈 Mutual Fund Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Mutual Fund Prediction Dashboard**! Use this tool to predict and analyze the best mutual funds for your investment timeframe and risk level. 🧠
    """
)

# Load the preloaded dataset
data = load_data()

# Sidebar Inputs
st.sidebar.header("🔧 Configure Your Inputs")
timeframe = st.sidebar.selectbox("📆 Select Investment Timeframe", options=["1 Year", "3 Years", "5 Years"])
risk_level = st.sidebar.slider("⚖️ Select Risk Level", min_value=1, max_value=6, value=3)

# Map timeframe to column names
timeframe_map = {
    "1 Year": "returns_1yr",
    "3 Years": "returns_3yr",
    "5 Years": "returns_5yr"
}
selected_timeframe = timeframe_map[timeframe]

# Prediction Button
if st.sidebar.button("🚀 Run Prediction"):
    with st.spinner("Predicting the best mutual funds..."):
        top_funds = predict_funds(data, selected_timeframe, risk_level)

    st.success("✅ Prediction Complete!")
    st.markdown("### 🎯 Top 3 Mutual Funds Based on Predicted Returns")
    st.dataframe(
        top_funds.style.format({
            "Predicted Returns": "{:.2f}",
            "expense_ratio": "{:.2f}",
            "fund_size_cr": "{:,.0f}"
        })
    )

