def build_market_features(df):
    return df[[
        "mv_mean",
        "mv_std",
        "mv_trend",
        "mv_max"
    ]]
