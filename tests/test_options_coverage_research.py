import pandas as pd

from src.options_coverage_research import AlpacaOptionsClient, coverage, write_assessment


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self, pages):
        self.headers = {}
        self.pages = iter(pages)

    def get(self, *args, **kwargs):
        return Response(next(self.pages))


def test_contract_pagination_and_spread_coverage() -> None:
    pages = [
        {"option_contracts": [{"expiration_date": "2024-03-15", "strike_price": "10"}], "next_page_token": "x"},
        {"option_contracts": [{"expiration_date": "2024-03-15", "strike_price": "5"}]},
    ]
    candidates = pd.DataFrame({"symbol": ["AAA"], "event_date": [pd.Timestamp("2024-02-10")]})
    result = coverage(candidates, AlpacaOptionsClient("key", "secret", Session(pages)))
    assert result.loc[0, "put_contracts"] == 2
    assert bool(result.loc[0, "spread_constructible"])
    assert "Not eligible" in write_assessment(result)


def test_precoverage_candidate_does_not_call_provider() -> None:
    candidates = pd.DataFrame({"symbol": ["AAA"], "event_date": [pd.Timestamp("2024-01-10")]})
    result = coverage(candidates, AlpacaOptionsClient("key", "secret", Session([])))
    assert not bool(result.loc[0, "data_available"])
