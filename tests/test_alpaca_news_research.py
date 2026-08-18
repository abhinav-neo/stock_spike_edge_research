import pandas as pd

from src.alpaca_news_research import event_news_features


def test_news_features_use_only_tagged_event_day_articles() -> None:
    events = pd.DataFrame([{"symbol": "ABC", "event_date": "2024-01-02"}])
    articles = {
        "2024-01-02": [
            {"symbols": ["ABC"], "headline": "ABC announces registered direct offering", "summary": ""},
            {"symbols": ["XYZ"], "headline": "Unrelated FDA trial", "summary": ""},
            {"symbols": ["ABC", "X", "Y", "Z"], "headline": "Stocks moving today", "summary": ""},
        ]
    }
    result = event_news_features(events, articles)
    assert result.loc[0, "news_article_count"] == 2
    assert result.loc[0, "news_specific_article_count"] == 1
    assert result.loc[0, "news_financing_count"] == 1
    assert result.loc[0, "news_market_mechanics_count"] == 1
