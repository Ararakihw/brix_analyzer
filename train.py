import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# 载入训练数据
train_data = pd.read_csv("data/train_data.csv")
X_train = train_data.drop(columns=["糖度(Brix)"]).values  # 光谱特征
y_train = train_data["糖度(Brix)"].values  # 糖度值

# 拆分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 训练 XGBoost 模型
model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 保存模型
model_path = "model/spectrum_brix_model.pkl"
joblib.dump(model, model_path)
print(f"✅ 模型已保存到 {model_path}")

# 模型验证
y_pred = model.predict(X_val)
score = r2_score(y_val, y_pred)
print(f"✅ 模型验证 R² 得分: {score:.2f}")
