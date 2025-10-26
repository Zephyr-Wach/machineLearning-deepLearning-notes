import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 1. 读取数据
# ===============================
train_df = pd.read_csv("../dataSet/mnist_train_100.csv", header=None)
test_df  = pd.read_csv("../dataSet/mnist_test_10.csv", header=None)

# 特征与标签
X_train = train_df.iloc[:, 1:].values / 255.0
y_train = train_df.iloc[:, 0].values.reshape(-1, 1)
X_test  = test_df.iloc[:, 1:].values / 255.0
y_test  = test_df.iloc[:, 0].values.reshape(-1, 1)

# One-Hot 编码标签
encoder = OneHotEncoder(sparse_output=False)
Y_train = encoder.fit_transform(y_train)
Y_test  = encoder.transform(y_test)

# ===============================
# 2. 定义网络结构与超参数
# ===============================
input_size = 784   # 输入层节点数
hidden_size = 64   # 隐藏层节点数
output_size = 10   # 输出层节点数
lr = 0.1           # 学习率
epochs = 200       # 迭代次数

# ===============================
# 3. 初始化权重
# ===============================
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# ===============================
# 4. 激活函数
# ===============================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ===============================
# 5. 训练
# ===============================
for epoch in range(epochs):
    # 前向传播
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    # 损失函数（交叉熵）
    loss = -np.mean(np.sum(Y_train * np.log(a2 + 1e-8), axis=1))

    # 反向传播
    dz2 = a2 - Y_train
    dW2 = np.dot(a1.T, dz2) / X_train.shape[0]
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = np.dot(dz2, W2.T) * sigmoid_deriv(a1)
    dW1 = np.dot(X_train.T, dz1) / X_train.shape[0]
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # 更新权重
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

# ===============================
# 6. 测试
# ===============================
z1 = np.dot(X_test, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = softmax(z2)

preds = np.argmax(a2, axis=1)
true  = y_test.flatten()

accuracy = np.mean(preds == true)
print(f"\n测试集准确率: {accuracy * 100:.2f}%\n")

# ===============================
# 7. 显示测试结果
# ===============================
for i in range(len(true)):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"样本{i+1}: 真实={true[i]}, 预测={preds[i]}")
    plt.axis('off')
    plt.show()
