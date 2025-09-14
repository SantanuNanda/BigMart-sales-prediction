"""
preprocess.py
Preprocess train and test CSVs and generate feature CSVs for modeling.
Usage:
    python preprocess.py --train path/to/train.csv --test path/to/test.csv --out_dir path/to/output_dir
"""
import argparse, os
import pandas as pd, numpy as np

def preprocess(train_path, test_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test_ids = test[['Item_Identifier','Outlet_Identifier']].copy()
    train['is_train'] = 1
    test['is_train'] = 0
    test['Item_Outlet_Sales'] = np.nan
    df = pd.concat([train, test], ignore_index=True, sort=False)
    # Standardize fat content
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat','low fat':'Low Fat','reg':'Regular'}).fillna('Unknown')
    # Impute Item_Weight
    df['Item_Weight'] = pd.to_numeric(df['Item_Weight'], errors='coerce')
    median_by_item = df.groupby('Item_Identifier')['Item_Weight'].median()
    global_wt_median = df['Item_Weight'].median()
    df['Item_Weight'] = df.apply(lambda r: median_by_item.get(r['Item_Identifier'], global_wt_median) if pd.isna(r['Item_Weight']) else r['Item_Weight'], axis=1)
    df['Item_Weight'].fillna(global_wt_median, inplace=True)
    # Item_Visibility
    df['Item_Visibility'] = pd.to_numeric(df['Item_Visibility'], errors='coerce')
    vis_median = df.loc[df['Item_Visibility']>0, 'Item_Visibility'].median()
    df['Item_Visibility'] = df['Item_Visibility'].replace(0, np.nan).fillna(vis_median)
    # Outlet years & prefix
    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    df['Item_Identifier_Prefix'] = df['Item_Identifier'].astype(str).str[:2]
    df['Outlet_Size'] = df['Outlet_Size'].fillna('Missing')
    df['Item_Type'] = df['Item_Type'].fillna('Other')
    df['Price_per_Weight'] = df['Item_MRP'] / (df['Item_Weight'].replace(0, np.nan))
    df['Price_per_Weight'] = df['Price_per_Weight'].fillna(df['Item_MRP'] / df['Item_Weight'].median())
    # Aggregates from TRAIN only
    agg_item = train.groupby('Item_Identifier')['Item_Outlet_Sales'].agg(['mean','median','count']).reset_index().rename(columns={'mean':'Item_mean_sales','median':'Item_median_sales','count':'Item_count'})
    agg_outlet = train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].agg(['mean','median','count']).reset_index().rename(columns={'mean':'Outlet_mean_sales','median':'Outlet_median_sales','count':'Outlet_count'})
    df = df.merge(agg_item, on='Item_Identifier', how='left')
    df = df.merge(agg_outlet, on='Outlet_Identifier', how='left')
    global_mean = train['Item_Outlet_Sales'].mean()
    for c in ['Item_mean_sales','Item_median_sales','Item_count','Outlet_mean_sales','Outlet_median_sales','Outlet_count']:
        if c in df.columns:
            df[c].fillna(global_mean if 'mean' in c or 'median' in c else 0, inplace=True)
    # relative visibility & interactions
    df['Visibility_rel_item'] = df['Item_Visibility'] / (df.groupby('Item_Identifier')['Item_Visibility'].transform('mean') + 1e-9)
    df['MRP_x_OutletYears'] = df['Item_MRP'] * df['Outlet_Years']
    df['MRP_div_Weight'] = df['Item_MRP'] / (df['Item_Weight'] + 1e-9)
    # final features list (cat kept as-is for CatBoost)
    features = [
        'Item_Weight','Item_Visibility','Item_MRP','Outlet_Years','Price_per_Weight',
        'Item_mean_sales','Item_median_sales','Item_count','Outlet_mean_sales','Outlet_median_sales','Outlet_count',
        'Visibility_rel_item','MRP_x_OutletYears','MRP_div_Weight',
        'Item_Identifier','Outlet_Identifier','Item_Fat_Content','Item_Type','Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Identifier_Prefix'
    ]
    df_final = df[features + ['is_train']].copy()
    train_df = df_final[df_final['is_train']==1].drop(columns=['is_train']).reset_index(drop=True)
    test_df  = df_final[df_final['is_train']==0].drop(columns=['is_train']).reset_index(drop=True)
    train_df.to_csv(os.path.join(out_dir,'train_features.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir,'test_features.csv'), index=False)
    print('Saved processed features to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out_dir', default='processed')
    args = parser.parse_args()
    preprocess(args.train, args.test, args.out_dir)
