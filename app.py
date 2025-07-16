import streamlit as st
import numpy as np

st.title("AKI Risk Calculator (ICU Ridge Model)")

st.write("""
This tool estimates the probability of acute kidney injury (AKI) in ICU patients using your 
final ridge logistic regression model, based on eICU data. 
It incorporates CKD history, nephrotoxins, APACHE severity, and key exposures.
""")

# Inputs
ckd_combined = st.checkbox("CKD (by eGFR or documented history)", value=False)
vol_overload = st.checkbox("Volume overload (>+3L positive balance before AKI)", value=False)
dysnatremia = st.checkbox("Dysnatremia (Na <135 or >145 before AKI)", value=False)
hyperlactatemia = st.checkbox("Hyperlactatemia (Lactate >2 before AKI)", value=False)
hypotension = st.checkbox("Hypotension (MAP <65 before AKI)", value=False)
nephrotoxin = st.checkbox("Nephrotoxin exposure before AKI", value=False)

apache_std = st.slider("APACHE score (standardized)", min_value=-3.0, max_value=3.0, step=0.1, value=0.0,
    help="Standardized APACHE score: (patient APACHE - cohort mean) / SD. Rough guide: -1 = low, 0 = avg, +1 = high.")

# Prediction
def predict_aki_ridge(ckd_combined, vol_overload, dysnatremia, hyperlactatemia,
                      hypotension, nephrotoxin, apache_std):
    intercept = 0.099
    beta_ckd = 0.781
    beta_vol = -0.928
    beta_dysnatremia = -0.643
    beta_hyperlact = -0.337
    beta_hypotension = -1.259
    beta_nephrotoxin = 0.371
    beta_apache = 0.759

    lp = (
        intercept
        + beta_ckd * ckd_combined
        + beta_vol * vol_overload
        + beta_dysnatremia * dysnatremia
        + beta_hyperlact * hyperlactatemia
        + beta_hypotension * hypotension
        + beta_nephrotoxin * nephrotoxin
        + beta_apache * apache_std
    )
    prob = 1 / (1 + np.exp(-lp))
    return prob

# Compute
prob = predict_aki_ridge(
    int(ckd_combined),
    int(vol_overload),
    int(dysnatremia),
    int(hyperlactatemia),
    int(hypotension),
    int(nephrotoxin),
    apache_std
)

st.success(f"✅ Predicted probability of AKI: **{prob:.2%}**")
