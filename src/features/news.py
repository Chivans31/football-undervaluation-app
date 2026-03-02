def build_sentiment_features(df):
    return df[[
        "sentiment_mean",
        "sentiment_std",
        "news_volume"
    ]]
