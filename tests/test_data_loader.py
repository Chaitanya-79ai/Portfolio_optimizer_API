from app.data_loader import get_available_tickers, load_data


def test_loads_expected_tickers_and_sheets():
    data = load_data()

    assert set(data) == {"fund_info", "fund_returns", "factor_returns"}
    assert get_available_tickers() == ["AGG", "GLD", "IEFA", "SPY", "VEA"]
    assert len(data["fund_info"]) == 5
    assert not data["fund_returns"].empty
    assert not data["factor_returns"].empty


def test_return_data_is_cleaned():
    data = load_data()

    fund_returns = data["fund_returns"]
    factor_returns = data["factor_returns"]

    assert fund_returns["date"].notna().all()
    assert fund_returns["total_return"].notna().all()
    assert factor_returns["date"].notna().all()
    assert factor_returns["total_return"].notna().all()
    assert set(factor_returns["index_ticker"].unique()) == {
        "Momentum Factor",
        "Size Factor",
        "Value Factor",
    }
