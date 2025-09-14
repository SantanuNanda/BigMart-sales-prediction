# BigMart Sales Prediction — 1-page approach note
================================================

Objective
---------
Predict `Item_Outlet_Sales` for the test set using historical 2013 sales data across 10 outlets and 1559 products.

High-level approach
-------------------
1. **Data cleaning & imputation**: Imputed missing `Item_Weight` using median weight per `Item_Identifier` (fallback to global median). Treated `Item_Visibility` zeros as missing and replaced with median non-zero visibility values. Filled missing categorical outlet sizes with 'Missing'.
2. **Feature engineering**: Created outlet age (`2013 - Outlet_Establishment_Year`), `Price_per_Weight`, and interactions such as `MRP_x_OutletYears` and `MRP_div_Weight`. Engineered relative visibility (`Visibility_rel_item`) and aggregate statistics from train: `Item_mean_sales`, `Item_median_sales`, `Item_count`, `Outlet_mean_sales`, `Outlet_median_sales`, `Outlet_count`.
3. **Encoding strategies**: For high-cardinality features (`Item_Identifier`, `Outlet_Identifier`) used target/mean encoding with smoothing to capture historic effects while reducing dimensionality. For low-cardinality columns (`Item_Fat_Content`, `Outlet_Type`, `Outlet_Size`, etc.) used one-hot encoding where needed. For CatBoost, native categorical handling was used to simplify preprocessing.
4. **Modeling & evaluation**: Used GroupKFold CV by `Outlet_Identifier` to avoid leakage across stores and to measure realistic generalization. Benchmarked multiple models: RandomForest, ExtraTrees, XGBoost, LightGBM, and CatBoost. CatBoost with pre-tuned params and native categorical support provided a strong baseline. Ensembling (stacking) of tree models + Ridge meta-model further reduced RMSE in experiments.
5. **Target transform**: Trained models on `log1p(Item_Outlet_Sales)` to stabilize variance and back-transformed predictions with `expm1`. Clipped negative predictions to 0.
6. **Hyperparameter tuning**: Provided Optuna-based CatBoost tuning notebook for extensive search (if environment supports Optuna). Also provided a pre-tuned CatBoost notebook for environments without package installation privileges.
7. **Production & reproducibility**: Packaged code into scripts and notebooks. Provided instructions, requirements, and examples to reproduce experiments locally or on cloud notebooks.

Key findings & practical tips
-----------------------------
- Aggregate features (item-level and outlet-level historical sales) consistently improved CV RMSE — these capture repeat-buy and outlet popularity effects.
- Mean/target encoding with smoothing for `Item_Identifier` and `Outlet_Identifier` outperformed naive OHE for these high-cardinality columns.
- CatBoost often matches or outperforms XGBoost/LightGBM when categorical variables are numerous; it reduces preprocessing complexity.
- Use GroupKFold by `Outlet_Identifier` to simulate predicting for new items in existing stores or similar real-world splits.

Files delivered
---------------
- Notebooks documenting EDA, feature engineering, and modeling experiments.
- Scripts to preprocess, train CatBoost (default tuned), train XGBoost baseline, and evaluate via GroupKFold.
- A final submission generation script and train-prediction outputs for inspection.
