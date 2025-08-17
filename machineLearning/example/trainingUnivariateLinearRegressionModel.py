import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 设置中文字体（Windows用 SimHei，Mac可以用 Songti SC）
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示成方块

# 1. 读取数据
data = pd.read_csv("../dateSet/priceBySize_dataset.csv")

# 2. 准备特征和标签
X = data[["Size_sqm"]]   # 面积 (二维)
y = data["Price_wan"]    # 房价 (一维)

# 3. 建立模型并训练
model = LinearRegression()
model.fit(X, y)

# 4. 输出结果
print("模型系数（斜率 b1）：", model.coef_[0])
print("模型截距（b0）：", model.intercept_)

# 预测示例：输入一个房屋面积
test_size = 150
pred_price = model.predict([[test_size]])
print(f"预测 {test_size} 平方米的房价约为 {pred_price[0]:.2f} 万元")

# 5. 可视化
plt.scatter(X, y, color="blue", label="真实数据")
plt.plot(X, model.predict(X), color="red", label="回归直线")
plt.xlabel("面积 (平方米)")
plt.ylabel("房价 (万元)")
plt.legend()
plt.show()
