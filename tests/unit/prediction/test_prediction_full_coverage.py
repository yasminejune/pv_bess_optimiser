from __future__ import annotations

import json
import runpy
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from ors.services.prediction import data_pipeline, run_sample, train_script
from ors.services.prediction import hyperparameter_search as hs
from ors.services.prediction import prediction_model as pm
from ors.services.prediction import report_generator as rg


class DummyXGBRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.feature_importances_ = np.array([0.5, 0.3, 0.2])

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.fit_shape = (len(x_train), len(y_train))

    def predict(self, x_data: pd.DataFrame) -> np.ndarray:
        return np.full(len(x_data), 10.0, dtype=float)

    def save_model(self, path: Path) -> None:
        Path(path).write_text("dummy-model")


class DummyModel(DummyXGBRegressor):
    def __init__(self):
        super().__init__()
        self.feature_importances_ = np.array([0.6, 0.4, 0.2])


class FixedDateTime:
    @classmethod
    def now(cls) -> real_datetime:
        return real_datetime(2025, 1, 1, 12, 0, 0)


def _merged_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01 00:00:00", periods=8, freq="h"),
            "Price": [10.0, 11.0, 12.0, 13.0, 12.0, 11.0, 10.0, 9.0],
            "feature1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feature2": [True, False, True, False, True, False, True, False],
            "feature3": [0.1, 0.2, np.nan, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


def test_data_pipeline_load_and_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "Data"
    data_dir.mkdir()

    (data_dir / "price_data_rotated_2d.csv").write_text("timestamp,Price\n2025-01-01 00:00:00,10\n")
    (data_dir / "historical_hourly_2023_2025.csv").write_text(
        "timestamp_utc,temp\n2025-01-01 00:00:00,1\n"
    )
    (data_dir / "historical_daily_2023_2025.csv").write_text(
        "date_utc,sunrise,sunset,daylight_duration\n"
        "2025-01-01 00:00:00,2025-01-01 08:00:00,2025-01-01 16:00:00,28800\n"
    )

    monkeypatch.setattr(data_pipeline, "preprocess_raw_data", lambda *a, **kw: _merged_df())

    price, weather, sun = data_pipeline.load_source_data(tmp_path)
    assert "timestamp" in price.columns
    assert "timestamp_utc" in weather.columns
    assert "date_utc" in sun.columns

    merged = data_pipeline.build_merged_dataset(tmp_path)
    assert merged["Timestamp"].is_monotonic_increasing


def test_data_pipeline_missing_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="price_data_rotated_2d.csv"):
        data_pipeline.load_source_data(tmp_path)

    data_dir = tmp_path / "Data"
    data_dir.mkdir()
    (data_dir / "price_data_rotated_2d.csv").write_text("timestamp,Price\n2025-01-01,10\n")
    with pytest.raises(FileNotFoundError, match="historical_hourly_2023_2025.csv"):
        data_pipeline.load_source_data(tmp_path)

    (data_dir / "historical_hourly_2023_2025.csv").write_text("timestamp_utc\n2025-01-01\n")
    with pytest.raises(FileNotFoundError, match="historical_daily_2023_2025.csv"):
        data_pipeline.load_source_data(tmp_path)


def test_prediction_model_core_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _merged_df()

    assert pm.resolve_target_column(df, "Price") == "Price"
    assert pm.resolve_target_column(df.rename(columns={"Price": "price"}), "Price") == "price"
    with pytest.raises(ValueError, match="Target column"):
        pm.resolve_target_column(pd.DataFrame({"x": [1]}), "Price")

    features, target = pm.prepare_features(df, target_col="Price")
    assert "Timestamp" not in features.columns
    assert np.issubdtype(features["feature2"].dtype, np.integer)
    assert target.dtype == float

    infer_no_target = pm.prepare_features_for_inference(df.drop(columns=["Price"]))
    assert "Timestamp" not in infer_no_target.columns

    features_no_timestamp, _ = pm.prepare_features(
        df.drop(columns=["Timestamp"]), target_col="Price"
    )
    assert "feature1" in features_no_timestamp.columns

    infer_without_timestamp = pm.prepare_features_for_inference(df.drop(columns=["Timestamp"]))
    assert "feature1" in infer_without_timestamp.columns

    x_train, x_test, y_train, y_test = pm.time_based_split(features, target, test_size=0.25)
    assert len(x_train) == 6
    assert len(x_test) == 2

    monkeypatch.setitem(
        __import__("sys").modules,
        "xgboost",
        SimpleNamespace(XGBRegressor=DummyXGBRegressor),
    )

    model_default = pm.train_xgb_regressor(x_train, y_train)
    model_with_params = pm.train_xgb_regressor(x_train, y_train, params={"max_depth": 4})
    assert isinstance(model_default, DummyXGBRegressor)
    assert model_with_params.kwargs["max_depth"] == 4

    metrics = pm.evaluate_model(model_default, x_test, y_test)
    assert set(metrics.keys()) == {"mae", "rmse", "mape", "r2"}

    inf_metrics = pm.evaluate_model(model_default, x_test, pd.Series([0.0, 0.0]))
    assert inf_metrics["mape"] == float("inf")

    metrics_path = tmp_path / "metrics.json"
    preds_path = tmp_path / "predictions.csv"
    fi_path = tmp_path / "feature_importance.csv"

    pm.save_metrics(metrics, metrics_path)
    pm.save_predictions(
        df.loc[y_test.index, "Timestamp"], y_test, model_default.predict(x_test), preds_path
    )
    pm.save_feature_importance(model_default, features, fi_path)

    assert metrics_path.exists()
    assert preds_path.exists()
    assert fi_path.exists()

    monkeypatch.setattr(pm, "datetime", FixedDateTime)
    project_root = tmp_path
    first = pm.create_model_run_dir(project_root, "xgboost")
    second = pm.create_model_run_dir(project_root, "xgboost")
    third = pm.create_model_run_dir(project_root, "xgboost")
    assert first.exists()
    assert second.exists()
    assert third.exists()
    assert first != second

    preds = pm.predict_prices(model_default, df)
    assert len(preds) == len(df)


def test_prediction_model_training_wrappers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _merged_df()

    monkeypatch.setattr(pm, "datetime", FixedDateTime)
    monkeypatch.setattr(pm, "train_xgb_regressor", lambda x, y: DummyModel())

    run_dir = pm.train_and_save_from_dataframe(tmp_path, df, model_name="xgb_test", test_size=0.25)
    assert (run_dir / "model.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "feature_importance.csv").exists()
    assert (run_dir / "metadata.json").exists()

    monkeypatch.setattr(pm, "build_merged_dataset", lambda project_root: df)
    run_dir_2 = pm.train_and_save(tmp_path, model_name="xgb_test2", test_size=0.25)
    assert run_dir_2.exists()

    called = {"ok": False}

    def fake_train_and_save(project_root: Path) -> Path:
        called["ok"] = True
        return project_root

    monkeypatch.setattr(pm, "train_and_save", fake_train_and_save)
    monkeypatch.setattr(
        pm.Path, "resolve", lambda _self: tmp_path / "src" / "ors" / "prediction_model.py"
    )
    pm.main()
    assert called["ok"]


def test_report_generator_functions_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path
    (project_root / "Data").mkdir()
    (project_root / "src").mkdir()

    assert rg.find_project_root(project_root / "Data") == project_root
    with pytest.raises(FileNotFoundError):
        rg.find_project_root(Path(tmp_path.anchor) / "nonexistent")

    models_dir = project_root / "Prediction" / "Models"
    assert rg.find_latest_run_dir(models_dir) is None
    models_dir.mkdir(parents=True)
    assert rg.find_latest_run_dir(models_dir) is None

    run_a = models_dir / "run_a"
    run_b = models_dir / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    latest = rg.find_latest_run_dir(models_dir)
    assert latest in {run_a, run_b}

    metrics_path = tmp_path / "metrics.json"
    importance_path = tmp_path / "importance.csv"
    metadata_path = tmp_path / "metadata.json"
    preds_path = tmp_path / "preds.csv"

    metrics_path.write_text(json.dumps({"mae": 1.2, "rmse": 2.3, "mape": 3.4, "r2": 0.9}))
    importance_df = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.7, 0.3]})
    importance_df.to_csv(importance_path, index=False)
    metadata_path.write_text(json.dumps({"model_name": "xgb"}))

    preds_df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "Price_true": [10.0, 11.0],
            "Price_pred": [10.5, 10.8],
        }
    )
    preds_df.to_csv(preds_path, index=False)

    assert rg.load_metrics(tmp_path / "missing.json") == {}
    assert rg.load_feature_importance(tmp_path / "missing.csv").empty
    assert rg.load_model_metadata(None) == {}
    assert rg.load_metrics(metrics_path)["mae"] == 1.2
    assert not rg.load_feature_importance(importance_path).empty
    assert rg.load_model_metadata(metadata_path)["model_name"] == "xgb"

    out_pdf = tmp_path / "report.pdf"
    rg.build_report(
        project_root=project_root,
        output_path=out_pdf,
        preds_path=preds_path,
        metrics_path=metrics_path,
        importance_path=importance_path,
        model_metadata_path=metadata_path,
        test_size=0.5,
        model_name="xgb",
        merged_df=_merged_df(),
    )
    assert out_pdf.exists()

    preds_no_true = tmp_path / "preds_no_true.csv"
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
            "Price_pred": [10.5, 10.8],
        }
    ).to_csv(preds_no_true, index=False)

    out_pdf_2 = tmp_path / "report2.pdf"
    rg.build_report(
        project_root=project_root,
        output_path=out_pdf_2,
        preds_path=preds_no_true,
        metrics_path=metrics_path,
        importance_path=tmp_path / "missing_imp.csv",
        model_metadata_path=None,
        test_size=0.5,
        model_name="xgb",
        merged_df=_merged_df(),
    )
    assert out_pdf_2.exists()

    sparse_metrics_path = tmp_path / "sparse_metrics.json"
    sparse_metrics_path.write_text(json.dumps({"mae": 1.0}))
    out_pdf_3 = tmp_path / "report3.pdf"
    rg.build_report(
        project_root=project_root,
        output_path=out_pdf_3,
        preds_path=preds_path,
        metrics_path=sparse_metrics_path,
        importance_path=importance_path,
        model_metadata_path=metadata_path,
        test_size=0.5,
        model_name="xgb",
        merged_df=_merged_df(),
    )
    assert out_pdf_3.exists()

    pdf_manual = tmp_path / "manual_pages.pdf"
    with rg.PdfPages(pdf_manual) as pdf:
        rg.create_text_page(pdf, "Long Text", [f"line {i}" for i in range(40)])
        rg.create_table_page(pdf, "Table", pd.DataFrame({"a": [1], "b": [2]}))

        fig1 = rg.plot_actual_vs_predicted(preds_df)
        fig2 = rg.plot_residuals(preds_df.assign(Residual=[0.1, -0.2]))
        fig3 = rg.plot_residual_distribution(preds_df.assign(Residual=[0.1, -0.2]))
        fig4 = rg.plot_feature_importance(importance_df)
        for fig in [fig1, fig2, fig3, fig4]:
            pdf.savefig(fig)
            rg.plt.close(fig)

    assert pdf_manual.exists()

    monkeypatch.setattr(
        rg,
        "parse_args",
        lambda: SimpleNamespace(
            project_root=project_root,
            run_dir=run_a,
            output=None,
            preds=None,
            metrics=None,
            importance=None,
            metadata=None,
            test_size=0.5,
            model_name="xgb",
        ),
    )

    for file_name, content in [
        ("predictions.csv", preds_df.to_csv(index=False)),
        ("metrics.json", metrics_path.read_text()),
        ("feature_importance.csv", importance_df.to_csv(index=False)),
        ("metadata.json", metadata_path.read_text()),
    ]:
        (run_a / file_name).write_text(content)

    monkeypatch.setattr(rg, "build_merged_dataset", lambda _root: _merged_df())
    rg.main()
    assert (run_a / "model_report.pdf").exists()

    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        monkeypatch.setattr(
            rg,
            "parse_args",
            lambda: SimpleNamespace(
                project_root=project_root,
                run_dir=project_root / "missing_run",
                output=None,
                preds=None,
                metrics=None,
                importance=None,
                metadata=None,
                test_size=0.5,
                model_name="xgb",
            ),
        )
        rg.main()

    prediction_dir = project_root / "Prediction"
    prediction_dir.mkdir(exist_ok=True)
    (prediction_dir / "xgb_predictions.csv").write_text(
        preds_df[["Timestamp", "Price_pred"]].to_csv(index=False)
    )
    (prediction_dir / "xgb_metrics.json").write_text(metrics_path.read_text())
    (prediction_dir / "xgb_feature_importance.csv").write_text(importance_df.to_csv(index=False))

    monkeypatch.setattr(
        rg,
        "parse_args",
        lambda: SimpleNamespace(
            project_root=project_root,
            run_dir=None,
            output=None,
            preds=None,
            metrics=None,
            importance=None,
            metadata=None,
            test_size=0.5,
            model_name="xgb",
        ),
    )
    monkeypatch.setattr(rg, "find_latest_run_dir", lambda _models_dir: None)
    rg.main()
    assert (prediction_dir / "model_report.pdf").exists()

    out_pdf_4 = tmp_path / "report4.pdf"
    with pytest.raises((ValueError, IndexError)):
        rg.build_report(
            project_root=project_root,
            output_path=out_pdf_4,
            preds_path=preds_path,
            metrics_path=tmp_path / "missing_metrics.json",
            importance_path=importance_path,
            model_metadata_path=metadata_path,
            test_size=0.5,
            model_name="xgb",
            merged_df=_merged_df(),
        )


def test_report_generator_parse_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "prog",
            "--project-root",
            str(tmp_path),
            "--test-size",
            "0.3",
            "--model-name",
            "ABC",
        ],
    )
    args = rg.parse_args()
    assert args.project_root == tmp_path
    assert args.test_size == 0.3
    assert args.model_name == "ABC"


def test_hyperparameter_search_helpers_and_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = hs.default_param_grid()
    assert "n_estimators" in grid

    assert hs.is_better("rmse", 1.0, 2.0)
    assert hs.is_better("r2", 0.9, 0.8)
    assert hs.is_better("mae", 1.0, None)

    monkeypatch.setattr(
        hs,
        "parse_args",
        lambda: SimpleNamespace(
            project_root=tmp_path,
            model_name="xgb_search",
            test_size=0.25,
            score_metric="mape",
            max_evals=2,
            random_state=42,
        ),
    )
    monkeypatch.setattr(hs, "build_merged_dataset", lambda _root: _merged_df())

    def fake_train(_x_train, _y_train, random_state=42, params=None):
        _ = random_state
        _ = params
        return DummyXGBRegressor()

    monkeypatch.setattr(hs, "train_xgb_regressor", fake_train)
    monkeypatch.setattr(hs, "create_model_run_dir", lambda root, name: tmp_path / "run")
    monkeypatch.setattr(hs, "save_metrics", pm.save_metrics)
    monkeypatch.setattr(hs, "save_predictions", pm.save_predictions)
    monkeypatch.setattr(hs, "save_feature_importance", pm.save_feature_importance)

    (tmp_path / "run").mkdir(exist_ok=True)
    hs.main()

    assert (tmp_path / "run" / "search_results.csv").exists()
    assert (tmp_path / "run" / "model.json").exists()

    monkeypatch.setattr(hs, "ParameterGrid", lambda _grid: [])
    monkeypatch.setattr(
        hs, "train_xgb_regressor", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError)
    )
    monkeypatch.setattr(
        hs,
        "parse_args",
        lambda: SimpleNamespace(
            project_root=tmp_path,
            model_name="xgb_search",
            test_size=0.25,
            score_metric="mape",
            max_evals=0,
            random_state=42,
        ),
    )
    with pytest.raises(RuntimeError, match="No models were trained"):
        hs.main()


def test_hyperparameter_search_parse_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        [
            "prog",
            "--project-root",
            str(tmp_path),
            "--model-name",
            "m",
            "--test-size",
            "0.1",
            "--score-metric",
            "mae",
            "--max-evals",
            "3",
            "--random-state",
            "7",
        ],
    )
    args = hs.parse_args()
    assert args.project_root == tmp_path
    assert args.model_name == "m"
    assert args.test_size == 0.1
    assert args.score_metric == "mae"
    assert args.max_evals == 3
    assert args.random_state == 7


def test_run_sample_and_train_script_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_sample_called = {"ok": False}

    monkeypatch.setattr(
        run_sample.pm if hasattr(run_sample, "pm") else run_sample,
        "train_and_save_from_dataframe",
        lambda project_root, df, model_name, test_size: tmp_path / "run",
    )
    monkeypatch.setattr(run_sample.pd, "read_csv", lambda *args, **kwargs: _merged_df())

    def fake_train(project_root, df, model_name, test_size):
        run_sample_called["ok"] = True
        return tmp_path / "run"

    monkeypatch.setattr(run_sample, "train_and_save_from_dataframe", fake_train)
    run_sample.main()
    assert run_sample_called["ok"]

    monkeypatch.setattr(
        train_script,
        "parse_args",
        lambda: SimpleNamespace(project_root=tmp_path, model_name="xgb", test_size=0.2),
    )
    monkeypatch.setattr(train_script, "build_merged_dataset", lambda _root: _merged_df())
    monkeypatch.setattr(
        train_script,
        "train_and_save_from_dataframe",
        lambda project_root, df, model_name, test_size: tmp_path / "run2",
    )
    monkeypatch.setattr(train_script, "build_report", lambda **kwargs: None)

    train_script.main()


def test_script_mode_import_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = tmp_path
    _ = monkeypatch
    runpy.run_path(str(Path(run_sample.__file__)), run_name="coverage_script")
    runpy.run_path(str(Path(train_script.__file__)), run_name="coverage_script")
    runpy.run_path(str(Path(hs.__file__)), run_name="coverage_script")


def test_train_script_parse_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        __import__("sys"),
        "argv",
        ["prog", "--project-root", str(tmp_path), "--model-name", "x", "--test-size", "0.4"],
    )
    args = train_script.parse_args()
    assert args.project_root == tmp_path
    assert args.model_name == "x"
    assert args.test_size == 0.4
