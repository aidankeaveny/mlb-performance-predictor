from pybaseball import batting_stats
import pandas as pd

feature_cols = [
    'G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SB', 'CS',
    'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'BB%', 'K%', 'BB/K', 'wOBA', 'wRC', 'wRC+', 'WAR', 'HR/FB',
    'LD%', 'GB%', 'FB%', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'EV', 'LA', 'Barrel%',
    'HardHit%', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'SwStr%',
    'Zone%', 'F-Strike%'
]

def build_y1_features(df_y1):
    y1 = df_y1[['Name'] + [col for col in feature_cols if col in df_y1.columns]].copy()
    y1.columns = [f"{col}_y1" if col != 'Name' else 'Name' for col in y1.columns]
    return y1

def create_aggregates(df_hist):
    valid_cols = [col for col in feature_cols if col in df_hist.columns]
    agg = (
        df_hist.groupby("Name")[valid_cols]
        .agg(['mean', 'std', 'max', 'sum'])
        .reset_index()
    )
    agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg.columns.values]
    return agg

def encode_positions(df, df_y1):
    if 'Pos' in df_y1.columns:
        pos_df = df_y1[['Name', 'Pos']].copy()
        pos_df['Pos'] = pos_df['Pos'].astype(str).fillna('UNK').str.split('-').str[0]
        pos_dummies = pd.get_dummies(pos_df['Pos'], prefix='Pos')
        pos_encoded = pd.concat([pos_df[['Name']], pos_dummies], axis=1)
        df = pd.merge(df, pos_encoded, on="Name", how="left")
    return df


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

def create_aggregates(df_history, feature_cols=feature_cols):
    valid_cols = [col for col in feature_cols if col in df_history.columns]
    agg = df_history.groupby("Name")[valid_cols].agg(['mean', 'std', 'max', 'sum'])
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    return agg.reset_index()
