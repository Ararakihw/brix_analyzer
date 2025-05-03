from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import pandas as pd

# 载入数据
train_data = pd.read_csv("data/train_data.csv")
X_train = train_data.drop(columns=["糖度(Brix)"]).values  # 光谱特征
y_train = train_data["糖度(Brix)"].values  # 糖度值

# 拆分数据集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 保存模型
joblib.dump(model, "model/other_model.pkl")
print("✅ 随机森林回归模型已保存")

# 验证模型
y_pred = model.predict(X_val)
score = r2_score(y_val, y_pred)
print(f"✅ 模型验证 R² 得分: {score:.2f}")
