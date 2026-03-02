def build_transfer_features(df):
    return df[[
        "n_transfers",
        "total_fees",
        "avg_fee"
    ]]
