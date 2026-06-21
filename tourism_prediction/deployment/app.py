import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# --- Page Setup & Configurations ---
st.set_page_config(
    page_title="Visit with Us - Prediction Engine",
    page_icon="✈️",
    layout="wide"
)

# --- Remote Artifact Connection Layer ---
# Downloads your optimized model binary directly from your model hub repository
REPO_ID = "mshahban/tourism-package-model"

@st.cache_resource
def load_production_pipeline():
    downloaded_file = hf_hub_download(repo_id=REPO_ID, filename="best_tourism_package_model_v1.joblib")
    return joblib.load(downloaded_file)

try:
    predictive_engine = load_production_pipeline()
except Exception as error_msg:
    st.error(f"Failed to fetch model from hub. Error detail: {error_msg}")
    st.stop()

# --- Main Dashboard Header ---
st.title("🧳 Wellness Tourism Package - Purchase Propensity Predictor")
st.markdown("---")

# --- Interactive Information Sidebar ---
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1507525428034-b723cf961d3e")
    st.header("💡 Live Profile Metrics")
    income_input = st.number_input("Gross Monthly Income (INR)", min_value=0, max_value=200000, value=25000, step=500)

    # Value added dynamic visual widget cards
    st.metric(label="Estimated Annual Income", value=f"₹{income_input * 12:,}")
    st.info("Ensure all questionnaire fields in the central tabs are fully populated before running prediction calculations.")

# --- Central Data Input Tabs ---
tab_profile, tab_sales = st.tabs(["👤 Customer Profile Data", "📞 Marketing & Pitch Interaction"])

with tab_profile:
    col_a, col_b = st.columns(2)
    with col_a:
        gender = st.selectbox("Customer Gender", ["Female", "Male"])
        age_group = st.selectbox("Demographic Age Interval", ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65'])
        marital_status = st.selectbox("Current Marital Status", ["Single", "Married", "Divorced"])
    with col_b:
        occupation = st.selectbox("Professional Sector / Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Organizational Seniority Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        city_tier = st.radio("Residency City Classification Tier", [1, 2, 3], horizontal=True)

with tab_sales:
    col_c, col_d = st.columns(2)
    with col_c:
        typeof_contact = st.selectbox("Lead Source / Acquisition Method", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Holiday Tier Product Variant Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        duration_of_pitch = st.slider("Duration of Sales Pitch Presentation (Minutes)", min_value=0, max_value=120, value=15)
        own_car = st.radio("Personal Automobile Ownership?", ["No", "Yes"], horizontal=True)
        passport = st.radio("Valid International Passport Holder?", ["No", "Yes"], horizontal=True)
    with col_d:
        num_person_visiting = st.slider("Total Intended Travelers Group Size", 1, 5, 2)
        num_children_visiting = st.slider("Number of Accompanying Children (Under Age 5)", 0, 3, 0)
        num_followups = st.slider("Total Sales Follow-up Iterations Completed", 1, 6, 3)
        num_trips = st.slider("Annual Vacation Frequency Rate", 1, 10, 2)
        pitch_satisfaction = st.select_slider("Pitch Presentation Satisfaction Rating", options=[1, 2, 3, 4, 5], value=3)
        preferred_stars = st.select_slider("Preferred Accommodation Star Rating", options=[3, 4, 5], value=3)

st.markdown("---")

# --- Inference Data Assembly Pipeline ---
if st.button("🔮 Evaluate Purchase Probability", type="primary", use_container_width=True):

    # 1. Enforce strict binary label mapping matching prep.py expectations
    gender_encoded = 0 if gender == "Female" else 1
    contact_encoded = 1 if typeof_contact == "Self Enquiry" else 0

    # 2. Assemble localized numeric structure schema
    runtime_features = {
        'TypeofContact': contact_encoded,
        'CityTier': city_tier,
        'DurationOfPitch': float(duration_of_pitch),
        'Gender': gender_encoded,
        'NumberOfPersonVisiting': num_person_visiting,
        'NumberOfFollowups': float(num_followups),
        'PreferredPropertyStar': float(preferred_stars),
        'NumberOfTrips': float(num_trips),
        'Passport': 1 if passport == "Yes" else 0,
        'PitchSatisfactionScore': pitch_satisfaction,
        'OwnCar': 1 if own_car == "Yes" else 0,
        'NumberOfChildrenVisiting': float(num_children_visiting),
        'MonthlyIncome': float(income_input)
    }

    # 3. Handle categorical layout expansions programmatically
    dummy_matrices = {
        "Occupation": (["Free Lancer", "Large Business", "Salaried", "Small Business"], occupation),
        "ProductPitched": (["Basic", "Deluxe", "King", "Standard", "Super Deluxe"], product_pitched),
        "MaritalStatus": (["Divorced", "Married", "Single"], marital_status),
        "Designation": (["AVP", "Executive", "Manager", "Senior Manager", "VP"], designation),
        "AgeGroup": (['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65'], age_group)
    }

    for feature_prefix, (allowed_labels, selected_val) in dummy_matrices.items():
        for category_label in allowed_labels:
            runtime_features[f"{feature_prefix}_{category_label}"] = 1 if selected_val == category_label else 0

    # Format vector dictionary into single row pandas structure
    inference_dataframe = pd.DataFrame([runtime_features])

    # --- Prediction Model Execution Block ---
    try:
        raw_prediction = predictive_engine.predict(inference_dataframe)[0]

        st.subheader("📋 Analytical Conversion Assessment")
        if raw_prediction == 1:
            st.success("🎯 **Target Lead:** The predictive analytics engine signals a **HIGH PROBABILITY** of conversion. This customer is likely to **BUY** the Wellness Tourism Package.")
            st.balloons()
        else:
            st.error("📉 **Cold Lead:** The predictive analytics engine signals a **LOW PROBABILITY** of conversion. This customer is likely to **REJECT** the package proposal.")

    except Exception as calculation_error:
        st.error(f"An unexpected data structure shape mismatch occurred during core execution: {calculation_error}")
