import pandas as pd
import joblib


# 加载模型
model_path = "model/spectrum_brix_model.pkl"
model = joblib.load(model_path)
print(f"✅ 模型已加载：{model_path}")

# 载入预测数据
test_data = pd.read_csv("data/test_data.csv")
X_test = test_data.values  # 预测特征

# 进行预测
predicted_brix = model.predict(X_test)

# 保存预测结果
test_data['预测糖度(Brix)'] = predicted_brix
print(test_data)
test_data.to_csv("data/test_predictions.csv", index=False, encoding='utf-8-sig')
print("✅ 预测结果已保存到 data/test_predictions.csv")
