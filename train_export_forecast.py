import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import joblib
except Exception:
    joblib = None


FEATURES = [
    'TimeIndex', 'Year', 'MonthNum', 'Quarter',
    'Month_sin', 'Month_cos',
    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_12',
    'Rolling_3m_mean', 'Rolling_6m_mean', 'Rolling_3m_std'
]
TARGET = 'Monthly_Sales'


@dataclass
class FoldResult:
    model_name: str
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    metrics: Dict[str, float]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # avoid division by zero
    denom = np.where(np.abs(y_true) < 1e-9, 1.0, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def build_model(model_name: str):
    if model_name == 'Linear Regression':
        return LinearRegression()
    if model_name == 'Random Forest':
        return RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    if model_name == 'Gradient Boosting':
        return GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    raise ValueError(f'Unknown model: {model_name}')


# NOTE: kept for future extension. Not used in current pipeline.
def make_future_features(history: List[float], last_month: pd.Timestamp, horizon: int):
    raise NotImplementedError



def add_lag_and_rolling(df_monthly: pd.DataFrame) -> pd.DataFrame:
    df_monthly = df_monthly.copy()

    df_monthly['Year'] = df_monthly['Month'].dt.year
    df_monthly['MonthNum'] = df_monthly['Month'].dt.month
    df_monthly['Quarter'] = df_monthly['Month'].dt.quarter
    df_monthly['TimeIndex'] = range(len(df_monthly))

    df_monthly['Month_sin'] = np.sin(2 * np.pi * df_monthly['MonthNum'] / 12)
    df_monthly['Month_cos'] = np.cos(2 * np.pi * df_monthly['MonthNum'] / 12)

    df_monthly['Lag_1'] = df_monthly['Monthly_Sales'].shift(1)
    df_monthly['Lag_2'] = df_monthly['Monthly_Sales'].shift(2)
    df_monthly['Lag_3'] = df_monthly['Monthly_Sales'].shift(3)
    df_monthly['Lag_12'] = df_monthly['Monthly_Sales'].shift(12)

    df_monthly['Rolling_3m_mean'] = df_monthly['Monthly_Sales'].shift(1).rolling(3).mean()
    df_monthly['Rolling_6m_mean'] = df_monthly['Monthly_Sales'].shift(1).rolling(6).mean()
    df_monthly['Rolling_3m_std'] = df_monthly['Monthly_Sales'].shift(1).rolling(3).std()

    df_model = df_monthly.dropna().copy()
    return df_model


def train_walk_forward(df_model: pd.DataFrame, model_names: List[str], horizon_test: int = None):
    # rolling-origin validation: choose fold size based on last 8 months like notebook default
    n = len(df_model)

    # If not provided, mimic notebook: test size ~20% of data
    if horizon_test is None:
        horizon_test = max(2, int(round(n * 0.20)))

    fold_results: List[FoldResult] = []

    # Starting train window leaving horizon_test at least
    min_train = n - horizon_test
    # We'll create multiple folds by stepping start of test earlier by horizon_test each time.
    # e.g., train up to 0.8, test next 0.2; then train up to 0.75, test next 0.2, etc.
    # Use step of 1 month for better averaging.
    step = 1
    last_test_start = n - horizon_test

    test_start = last_test_start
    while test_start - 12 > 0:  # keep enough for lags
        train_end = test_start
        test_end = test_start + horizon_test
        if test_end > n:
            break

        X_train = df_model.iloc[:train_end][FEATURES]
        y_train = df_model.iloc[:train_end][TARGET]
        X_test = df_model.iloc[test_start:test_end][FEATURES]
        y_test = df_model.iloc[test_start:test_end][TARGET]
        
        dates_test = df_model.iloc[test_start:test_end]['Month']

        for mname in model_names:
            model = build_model(mname)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(math.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))
            mape_val = mape(y_test.values, preds)

            fold_results.append(
                FoldResult(
                    model_name=mname,
                    train_start_idx=0,
                    train_end_idx=train_end,
                    test_start_idx=test_start,
                    test_end_idx=test_end,
                    metrics={
                        'MAE': mae,
                        'RMSE': rmse,
                        'R2': r2,
                        'MAPE': mape_val,
                    },
                )
            )

        test_start -= step
        if test_start < 12:
            break

    return fold_results


def aggregate_fold_metrics(fold_results: List[FoldResult]):
    by_model: Dict[str, List[FoldResult]] = {}
    for fr in fold_results:
        by_model.setdefault(fr.model_name, []).append(fr)

    agg = {}
    for model_name, lst in by_model.items():
        metrics = {}
        for key in ['MAE', 'RMSE', 'R2', 'MAPE']:
            vals = [x.metrics[key] for x in lst]
            metrics[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=0)),
            }
        agg[model_name] = metrics

    return agg


def fit_final_and_forecast(df_model: pd.DataFrame, best_model_name: str, horizon: int = 6, interval_bootstrap: bool = False, n_bootstrap: int = 200):
    # Fit on all data
    X = df_model[FEATURES]
    y = df_model[TARGET]

    model = build_model(best_model_name)
    model.fit(X, y)

    # Iterative forecast
    last_time_index = int(df_model['TimeIndex'].iloc[-1])
    history = df_model[TARGET].tolist()

    future_dates = pd.date_range(df_model['Month'].max(), periods=horizon + 1, freq='ME')[1:]

    preds = []
    for i, date in enumerate(future_dates):
        t_idx = last_time_index + i + 1
        month_num = int(date.month)
        quarter = int(date.quarter)
        year = int(date.year)

        feat_row = {
            'TimeIndex': t_idx,
            'Year': year,
            'MonthNum': month_num,
            'Quarter': quarter,
            'Month_sin': float(np.sin(2 * np.pi * month_num / 12)),
            'Month_cos': float(np.cos(2 * np.pi * month_num / 12)),
            'Lag_1': history[-1],
            'Lag_2': history[-2],
            'Lag_3': history[-3],
            'Lag_12': history[-12],
            'Rolling_3m_mean': float(np.mean(history[-3:])),
            'Rolling_6m_mean': float(np.mean(history[-6:])),
            'Rolling_3m_std': float(np.std(history[-3:], ddof=0)),
        }

        feat_df = pd.DataFrame([feat_row])
        pred = float(model.predict(feat_df[FEATURES])[0])
        preds.append(pred)
        history.append(pred)

    forecast = {
        'model': best_model_name,
        'horizon_months': horizon,
        'forecast_dates': [d.strftime('%b %Y') for d in future_dates],
        'forecast_values': preds,
    }

    # Basic intervals via bootstrap residuals (optional)
    if interval_bootstrap:
        # residual bootstrap from in-sample residuals
        in_sample_preds = model.predict(X)
        residuals = (y.values - in_sample_preds)
        if len(residuals) > 0:
            sims = []
            rng = np.random.default_rng(42)
            for _ in range(n_bootstrap):
                sim_history = df_model[TARGET].tolist()[:]
                sim_preds = []
                for i, date in enumerate(future_dates):
                    t_idx = last_time_index + i + 1
                    month_num = int(date.month)
                    quarter = int(date.quarter)
                    year = int(date.year)
                    feat_row = {
                        'TimeIndex': t_idx,
                        'Year': year,
                        'MonthNum': month_num,
                        'Quarter': quarter,
                        'Month_sin': float(np.sin(2 * np.pi * month_num / 12)),
                        'Month_cos': float(np.cos(2 * np.pi * month_num / 12)),
                        'Lag_1': sim_history[-1],
                        'Lag_2': sim_history[-2],
                        'Lag_3': sim_history[-3],
                        'Lag_12': sim_history[-12],
                        'Rolling_3m_mean': float(np.mean(sim_history[-3:])),
                        'Rolling_6m_mean': float(np.mean(sim_history[-6:])),
                        'Rolling_3m_std': float(np.std(sim_history[-3:], ddof=0)),
                    }
                    feat_df = pd.DataFrame([feat_row])
                    base_pred = float(model.predict(feat_df[FEATURES])[0])
                    eps = float(residuals[rng.integers(0, len(residuals))])
                    sim_pred = base_pred + eps
                    sim_preds.append(sim_pred)
                    sim_history.append(sim_pred)
                sims.append(sim_preds)

            sims_arr = np.array(sims)  # (n_bootstrap, horizon)
            lo = np.percentile(sims_arr, 10, axis=0).tolist()
            hi = np.percentile(sims_arr, 90, axis=0).tolist()
            forecast['intervals'] = {'p10': lo, 'p90': hi}

    return model, forecast


def main():
    csv_path = 'Sample - Superstore.csv'

    df = pd.read_csv(csv_path, encoding='latin-1')
    if 'Order Date' not in df.columns or 'Sales' not in df.columns:
        raise RuntimeError('Expected columns not found in CSV.')
    
    # Convert all currency values from USD to INR
    df['Sales'] = df['Sales'] * 83
    if 'Profit' in df.columns:
        df['Profit'] = df['Profit'] * 83

    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Ship Date'] = pd.to_datetime(df['Ship Date']) if 'Ship Date' in df.columns else df['Order Date']

    df = df.drop_duplicates().copy()

    df_monthly = (
        df.groupby(pd.Grouper(key='Order Date', freq='ME'))[['Sales']]
        .sum()
        .reset_index()
        .rename(columns={'Order Date': 'Month', 'Sales': 'Monthly_Sales'})
    )

    df_model = add_lag_and_rolling(df_monthly)

    model_names = ['Linear Regression', 'Random Forest', 'Gradient Boosting']

    fold_results = train_walk_forward(df_model, model_names=model_names, horizon_test=None)
    agg = aggregate_fold_metrics(fold_results)

    # Choose best by lowest mean MAPE
    best_model_name = min(agg.keys(), key=lambda m: agg[m]['MAPE']['mean'])

    # Fit final model on all data and forecast
    model, forecast = fit_final_and_forecast(df_model, best_model_name=best_model_name, horizon=6, interval_bootstrap=True, n_bootstrap=200)

    # Export historical monthly sales for charting
    historical = df_monthly[['Month', 'Monthly_Sales']].copy()
    historical['label'] = historical['Month'].dt.strftime('%b %y')

    out_forecast = {
        'features': FEATURES,
        'historical': {
            'labels': historical['Month'].dt.strftime('%b %Y').tolist(),
            'values': historical['Monthly_Sales'].astype(float).tolist(),
        },
        'forecast': forecast,
        'meta': {
            'best_model': best_model_name,
            'generated_by': 'train_export_forecast.py'
        }
    }

    out_metrics = {
        'walk_forward': {
            'fold_count': len(fold_results),
            'summary': agg,
        },
        'best_model': best_model_name,
        'generated_by': 'train_export_forecast.py'
    }

    with open('forecast.json', 'w', encoding='utf-8') as f:
        json.dump(out_forecast, f, ensure_ascii=False, indent=2)

    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump(out_metrics, f, ensure_ascii=False, indent=2)

    if joblib is not None:
        joblib.dump({'model': model, 'features': FEATURES, 'best_model_name': best_model_name}, 'model.joblib')

    print('✅ Exported forecast.json and metrics.json')


if __name__ == '__main__':
    main()

