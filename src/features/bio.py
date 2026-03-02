def build_bio_features(df):
    return df[[
        "Age",
        "Height (m)",
        "contract_years_left"
    ]]
