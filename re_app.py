import streamlit as st
import pandas as pd
import json
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

base_dir=os.path.dirname(__file__)

# Loading dfm dataframe
@st.cache_data
def load_data():
    data_path=os.path.join(base_dir,'data','dfm_model_inputs.parquet')
    df=pd.read_parquet(data_path)
    return df
# Loading the feature list
def load_feature_list():
    fl_path=os.path.join(base_dir,'data','feature_list.json')
    with open(fl_path,'r') as f:
        feat_list=json.load(f)
    return feat_list

# Loading the saved models
@st.cache_resource
def load_models():
    models_dir=os.path.join(base_dir,'data','saved_models')
    model_r=joblib.load(os.path.join(models_dir,'xgb_model_r.joblib'))
    model_p=joblib.load(os.path.join(models_dir,'rfr_model_price.joblib'))
    model_c=joblib.load(os.path.join(models_dir,'xgb_model_clf.joblib'))
    return model_c,model_r,model_p

# Loading the encoders
def load_encoders():
    enc_dir=os.path.join(base_dir,'data','encoders')
    
    files={'le_investment_type':'le_investment_type.joblib',
           'te_city':'te_city.joblib',
           'te_state':'te_state.joblib'}

    encoders={}
    for key,fname in files.items():
        path=os.path.join(enc_dir,fname)
        try:
            encoders[key]=joblib.load(path)
        except FileNotFoundError:
            encoders[key]=None
        except Exception as e:
            st.warning(f'Could not load encoder {fname}: {e}')
            encoders[key]=None

    return encoders

dfm=load_data()
model_c,model_r,model_p=load_models()
feature_list=load_feature_list()
encoders=load_encoders()
prop_map={1:'Apartment', 2:'Independent House', 3:'Villa'}

# Sidebar navigation
with st.sidebar:
    st.title("AI driven Real Estate Clarity.")
    st.header("Smarter Property Picks. Backed by Data.")

page=st.sidebar.radio('Go to:',('Analytics Dashboard','Investment Advisor'))

if page=='Analytics Dashboard':
    st.header('Analytics Dashboard')

    col1,col2,col3=st.columns(3)
    with col1:
        st.metric("Avg. Price (lakhs)",f"{dfm['Price_in_Lakhs'].mean():.2f}")
    with col2:
        st.metric("Median Price (lakhs)",f"{dfm['Price_in_Lakhs'].median():.2f}")
    with col3:
        st.metric("Avg. Appreciation Rate (r)",f"{(dfm['r'].mean())*100:.2f}%")

    st.markdown('---')

    # Size vs Price (line plot on binned data)
    st.markdown("### Trend: Property Size vs Price")

    dfm_bins = dfm.copy()
    dfm_bins['Size_bin']=pd.cut(dfm_bins['Size_in_SqFt'],bins=50)
    dfm_bins['Size_mid']=dfm_bins['Size_bin'].apply(lambda x:x.mid)
    size_trend=dfm_bins.groupby('Size_mid')['Price_in_Lakhs'].median().reset_index()
    st.bar_chart(size_trend,x='Size_mid',y='Price_in_Lakhs')

    st.markdown('---')

    # Median price and price per sq.ft. by BHK
    st.markdown('#### Median Price by BHK')

    price_by_bhk=dfm.groupby('BHK')['Price_in_Lakhs'].median().reset_index().sort_values('BHK')
    st.bar_chart(price_by_bhk,x='BHK',y='Price_in_Lakhs')

    st.markdown('---')

    st.markdown('#### Mean Price per sq.ft. by BHK')
    pps_by_bhk=dfm.groupby('BHK')['Price_per_SqFt'].mean().reset_index().sort_values('BHK')
    st.bar_chart(pps_by_bhk,x='BHK',y='Price_per_SqFt')

    st.markdown('---')

    # Price per sq.ft by Property Type
    st.markdown('#### Average Price per sq.ft by Property Type')

    # Handle encoded values if needed
    if pd.api.types.is_integer_dtype(dfm['Property_Type']):
        temp=dfm.copy()
        temp['Property_Type_Label']=temp['Property_Type'].map(prop_map)
        grp=temp.groupby('Property_Type_Label')['Price_per_SqFt'].mean().reset_index()
        st.bar_chart(grp,x='Property_Type_Label',y='Price_per_SqFt')
    else:
        grp=dfm.groupby('Property_Type')['Price_per_SqFt'].mean().reset_index()
        st.bar_chart(grp,x='Property_Type',y='Price_per_SqFt')

    st.markdown('---')

    # Appreciation rate by investment type
    st.markdown('#### Average Appreciation Rate by Investment Type')

    r_by_inv=dfm.groupby('investment_type')['r'].mean().reset_index()

    st.bar_chart(r_by_inv,x='investment_type', y='r')

    st.markdown('---')

    # Average Price and Price per sq.ft. of Property by State
    st.markdown('### Average Property Price per sq.ft. by State')
    pps_by_state=dfm.groupby('State')['Price_per_SqFt'].mean().reset_index().sort_values('Price_per_SqFt')
    st.bar_chart(pps_by_state,x='State',y='Price_per_SqFt')

    st.markdown('### Average Property Price by State')
    price_by_state=dfm.groupby('State')['Price_in_Lakhs'].mean().reset_index().sort_values('Price_in_Lakhs')
    st.bar_chart(price_by_state,x='State',y='Price_in_Lakhs')

    st.markdown('---')

    # Number of Different Types of Properties in each State
    st.markdown('### Number of Different Types of Properties in each State')
    tmp=dfm.copy()
    tmp['prop_label']=tmp['Property_Type'].map(prop_map)
    prop_count=pd.crosstab(tmp['State'],tmp['prop_label'])
    fig,ax=plt.subplots(figsize=(10,max(6,len(prop_count)*0.15)))
    sns.heatmap(prop_count,fmt='d',cmap='plasma',annot=True,ax=ax)
    ax.set_title('Count of Property Type by State')
    ax.set_ylabel('State')
    ax.set_xlabel('Property Type')
    plt.xticks(rotation=45,ha='right')
    plt.tight_layout()
    st.pyplot(fig)

elif page == 'Investment Advisor':
    st.header("Investment Advisor")
    st.markdown("### Your Real Estate decisions, powered by AI")
    st.markdown("Choose your specifications from the options below:")

    STATES = ["Andhra Pradesh", "Assam", "Bihar", "Chhattisgarh", "Delhi", "Gujarat", "Haryana", "Jharkhand",
        "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Odisha", "Punjab", "Rajasthan", "Tamil Nadu",
        "Telangana", "Uttar Pradesh", "Uttarakhand", "West Bengal"]
    
    MASTER_CITIES = ["Ahmedabad", "Amritsar", "Bangalore", "Bhopal", "Bhubaneswar", "Bilaspur", "Chennai", "Coimbatore",
        "Cuttack", "Dehradun", "Durgapur", "Dwarka", "Faridabad", "Gaya", "Gurgaon", "Guwahati", "Haridwar",
        "Hyderabad", "Indore", "Jaipur", "Jamshedpur", "Jodhpur", "Kochi", "Kolkata", "Lucknow", "Ludhiana",
        "Mangalore", "Mumbai", "Mysore", "Nagpur", "New Delhi", "Noida", "Patna", "Pune", "Raipur", "Ranchi",
        "Silchar", "Surat", "Trivandrum", "Vijayawada", "Vishakhapatnam", "Warangal"]

    # Property type mapping (dfm stores numeric codes 1/2/3)
    prop_map = {1: 'Apartment', 2: 'Independent House', 3: 'Villa'}
    inverse_prop_map = {v: k for k, v in prop_map.items()}

    # Helper: safely call a saved encoder (target/ordinal)
    def apply_encoder_safe(enc, col_name, value):
        if enc is None:
            return value
        try:
            df_in = pd.DataFrame({col_name: [value]})
            res = enc.transform(df_in)
            if isinstance(res, pd.DataFrame):
                val = res.iloc[0, 0]
            elif isinstance(res, pd.Series):
                val = res.iat[0]
            else:
                try:
                    if hasattr(res, "_len_") and not isinstance(res, (str, bytes)):
                        val = res[0]
                    else:
                        val = res
                except Exception:
                    val = res
            try:
                return pd.to_numeric(val, errors='coerce')
            except Exception:
                return val
        except Exception:
            # fallback: return original value
            return value

    # UI inputs: state first, then city list depends on state
    col1, col2 = st.columns(2)
    with col1:
        default_state = dfm['State'].mode().iat[0] if ('State' in dfm.columns and not dfm['State'].mode().empty) else STATES[0]
        user_state = st.selectbox("State", options=STATES, index=STATES.index(default_state) if default_state in STATES else 0, key="state_select")
        # city options filtered by dfm rows in that state
        available_cities = sorted(dfm[dfm['State'] == user_state]['City'].unique().tolist())
        if not available_cities:
            # fallback to master list
            available_cities = MASTER_CITIES
        # default city: first available
        user_city = st.selectbox("City", options=available_cities, index=0 if len(available_cities) > 0 else 0, key="city_select")

        # Property type shown as labels; we'll convert to numeric code before predicting
        user_prop_label = st.selectbox("Property Type", options=list(inverse_prop_map.keys()), index=0, key="prop_type")
        user_bhk = st.slider("BHK", min_value=1, max_value=5, value=2, step=1, key="bhk")

    with col2:
        user_size = st.slider("Size_in_SqFt", min_value=500, max_value=5000, value=900, step=50, key="size")
        user_price_per_sqft = st.slider("Price_per_SqFt (lakhs per sq.ft)", min_value=0.01, max_value=0.99, value=0.10, step=0.01, format="%.2f", key="pps")
        user_nearby_hosp = st.slider("Nearby_Hospitals (count)", min_value=1, max_value=10, value=2, step=1, key="hosp")
        user_age = st.slider("Age_of_Property (years)", min_value=2, max_value=35, value=5, step=1, key="age")
        user_amenities = st.multiselect("Amenities (choose any)", options=["Pool", "Clubhouse", "Gym", "Garden", "Playground"], default=[], key="amen")

    st.markdown("---")

    # Build one-row input using feature_list & dfm defaults (median/mode)
    def build_input_row():
        base = {}
        # if feature_list available (which was loaded earlier), use it. otherwise fallback to dfm columns
        cols = feature_list if ('feature_list' in globals() and feature_list) else list(dfm.columns)
        for c in cols:
            if c in dfm.columns:
                # numeric -> median, categorical -> mode
                if pd.api.types.is_numeric_dtype(dfm[c]):
                    base[c] = float(dfm[c].median())
                else:
                    base[c] = dfm[c].mode().iat[0] if not dfm[c].mode().empty else ""
            else:
                # unknown column -> zero default
                base[c] = 0
        # override with user values (use exact feature names present in your dfm)
        base.update({
            "State": user_state,
            "City": user_city,
            "Property_Type": int(inverse_prop_map[user_prop_label]),  # numeric code as in dfm
            "BHK": int(user_bhk),
            "Size_in_SqFt": float(user_size),
            "Price_per_SqFt": float(user_price_per_sqft),
            "Age_of_Property": int(user_age),
            "Nearby_Hospitals": int(user_nearby_hosp),
            "Pool": 1 if "Pool" in user_amenities else 0,
            "Clubhouse": 1 if "Clubhouse" in user_amenities else 0,
            "Gym": 1 if "Gym" in user_amenities else 0,
            "Garden": 1 if "Garden" in user_amenities else 0,
            "Playground": 1 if "Playground" in user_amenities else 0
        })
        return base

    if st.button("Predict"):
        # build input and DataFrame
        base_row = build_input_row()
        X = pd.DataFrame([base_row])

        # Using encoders that were loaded in the encoders dictionary
        te_state = encoders.get("te_state") if isinstance(encoders, dict) else None
        te_city = encoders.get("te_city") if isinstance(encoders, dict) else None
        # corrected key to match loader
        le_inv = encoders.get("le_investment_type") if isinstance(encoders, dict) else None

        # State
        if "State" in X.columns:
            if te_state is not None:
                out_state = apply_encoder_safe(te_state, "State", X.at[0, "State"])
                
                state_num = pd.to_numeric(out_state, errors='coerce')
                if pd.notna(state_num):
                    X.loc[:, "State"] = state_num
                else:
                    # fallback to dfm encoded representation
                    match = dfm[dfm["State"] == X.at[0, "State"]]
                    if len(match) > 0:
                        X.loc[:, "State"] = match["State"].iloc[0]
                    else:
                        try:
                            X.loc[:, "State"] = pd.to_numeric(dfm["State"].median())
                        except Exception:
                            X.loc[:, "State"] = dfm["State"].mode().iat[0] if not dfm["State"].mode().empty else X.at[0, "State"]
            else:
                match = dfm[dfm["State"] == X.at[0, "State"]]
                if len(match) > 0:
                    X.loc[:, "State"] = match["State"].iloc[0]
                else:
                    # fallback to dfm mode/median
                    if pd.api.types.is_numeric_dtype(dfm["State"]):
                        X.loc[:, "State"] = dfm["State"].median()
                    else:
                        X.loc[:, "State"] = dfm["State"].mode().iat[0] if not dfm["State"].mode().empty else X.at[0, "State"]

        # City
        if "City" in X.columns:
            if te_city is not None:
                out_city = apply_encoder_safe(te_city, "City", X.at[0, "City"])
                city_num = pd.to_numeric(out_city, errors='coerce')
                if pd.notna(city_num):
                    X.loc[:, "City"] = city_num
                else:
                    match = dfm[dfm["City"] == X.at[0, "City"]]
                    if len(match) > 0:
                        X.loc[:, "City"] = match["City"].iloc[0]
                    else:
                        try:
                            X.loc[:, "City"] = pd.to_numeric(dfm["City"].median())
                        except Exception:
                            X.loc[:, "City"] = dfm["City"].mode().iat[0] if not dfm["City"].mode().empty else X.at[0, "City"]
            else:
                match = dfm[dfm["City"] == X.at[0, "City"]]
                if len(match) > 0:
                    X.loc[:, "City"] = match["City"].iloc[0]
                else:
                    if pd.api.types.is_numeric_dtype(dfm["City"]):
                        X.loc[:, "City"] = dfm["City"].median()
                    else:
                        X.loc[:, "City"] = dfm["City"].mode().iat[0] if not dfm["City"].mode().empty else X.at[0, "City"]

        # Ensured that all feature_list columns exist and are in order
        cols_expected = feature_list if ('feature_list' in globals() and feature_list) else list(dfm.columns)
        for c in cols_expected:
            if c not in X.columns:
                # fill from dfm defaults
                if c in dfm.columns:
                    X[c] = dfm[c].median() if pd.api.types.is_numeric_dtype(dfm[c]) else (dfm[c].mode().iat[0] if not dfm[c].mode().empty else "")
                else:
                    X[c] = 0
        X = X[cols_expected]

        # Coerce numeric columns and fill nans with dfm median
        for c in X.columns:
            if c in dfm.columns and pd.api.types.is_numeric_dtype(dfm[c]):
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(dfm[c].median())

        # FINAL SAFETY CHECK: ensure State and City are numeric (models were trained on numeric target encodings for these two columns)
        for col in ["State", "City"]:
            if col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    coerced = pd.to_numeric(X[col], errors='coerce')
                    if coerced.isna().any():
                        st.error(f"Input column '{col}' could not be coerced to numeric encoded values. Check encoders and dfm.")
                        # stop further execution of predictions
                        st.stop()
                    else:
                        X[col] = coerced

        # Run models
        pred_r = pred_p = pred_c_label = None
        try:
            pred_r = float(model_r.predict(X)[0])
        except Exception as e:
            st.error(f"Appreciation (r) model error: {e}")
        try:
            pred_p = float(model_p.predict(X)[0])
        except Exception as e:
            st.error(f"Price model error: {e}")
        try:
            pred_c_raw = model_c.predict(X)[0]
            # decode investment label
            if le_inv is not None and hasattr(le_inv, "inverse_transform"):
                try:
                    pred_c_label = le_inv.inverse_transform([pred_c_raw])[0]
                except Exception:
                    pred_c_label = str(pred_c_raw)
            else:
                # If classifier returns numeric codes (1/2/3), map using prop_map
                try:
                    pred_int = int(pred_c_raw)
                    pred_c_label = prop_map.get(pred_int, str(pred_c_raw))
                except Exception:
                    pred_c_label = str(pred_c_raw)
        except Exception as e:
            st.error(f"Investment classifier error: {e}")

        # Display results
        if pred_p is not None and pred_r is not None:
            future_price_5y = pred_p * ((1 + pred_r) ** 5)
            profit = future_price_5y - pred_p

            c1, c2, c3 = st.columns(3)
            c1.metric("Appreciation rate (r)", f"{(pred_r)*100:.2f}%")
            c2.metric("Current Price", f"₹{pred_p:.2f} lakhs")
            c3.metric("Estimated Price in 5 years", f"₹{future_price_5y:.2f} lakhs")

            st.markdown(f"Estimated Profit in 5 years: ₹{profit:.2f} lakhs")

            if pred_c_label is not None:
                st.write(f"Type of Investment: {pred_c_label}")

            # recommendations
            rec = []
            if profit >= 150:
                rec.append("High appreciation expected — very good buy for long-term hold.")
            elif profit >= 80:
                rec.append("Good appreciation expected — good buy for long-term hold.")
            elif profit >= 30:
                rec.append("Moderate appreciation — decent investment.")
            elif profit < 30:
                rec.append("Low appreciation — consider only if price is attractive.")
            else:
                rec.append("Very low or negative appreciation — risky for appreciation-focused investing.")

            if X.at[0, "Clubhouse"] == 1 or X.at[0, "Gym"] == 1 or X.at[0, "Pool"] == 1:
                rec.append("Has premium amenities which help resale/tenantability.")
            if X.at[0, "Garden"] == 1 or X.at[0, "Playground"] == 1:
                rec.append("Family-friendly amenities present — appeals to families.")
            
            st.markdown("Recommendations:")
            for r in rec:
                st.write("- " + r)
        else:
            st.info("Prediction incomplete — check model / encoders.")