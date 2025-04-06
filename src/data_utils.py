from pybaseball import batting_stats
import pandas as pd

def load_historical_data(up_to=2023, start=1900, feature_cols=[]):
    all_years = []
    for year in range(start, up_to + 1):
        try:
            df = batting_stats(year)
            df['Season'] = year
            all_years.append(df)
        except:
            continue
    return pd.concat(all_years)

def create_aggregates(df_history, feature_cols):
    valid_cols = [col for col in feature_cols if col in df_history.columns]
    agg = df_history.groupby("Name")[valid_cols].agg(['mean', 'std', 'max', 'sum'])
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg.reset_index()
