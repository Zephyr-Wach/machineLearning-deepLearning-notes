import matplotlib.pyplot as plt
import networkx as nx

# 设置中文字体，避免乱码
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False

def draw_neural_network(title, layers, weights, pos):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    G = nx.DiGraph()
    nodes = []
    edges = []
    labels = {}

    # 添加节点
    for i, layer in enumerate(layers):
        for j in range(layer):
            node = f"L{i}N{j}"
            nodes.append(node)
            if i == 0:  # 输入层
                labels[node] = "+1" if j == 0 else f"x{j}"
            elif i == len(layers) - 1:  # 输出层
                labels[node] = "h(x)"
            else:  # 隐层
                labels[node] = f"y{j+1}" if j < 2 else "+1"

    # 添加边
    for i in range(len(layers)-1):
        if i == 0 and len(layers) == 3:  # XOR/XNOR 输入层到隐层
            for j in range(layers[i]):  # 输入(+1, x1, x2)
                for k in range(layers[i+1]-1):  # 隐层神经元 y1, y2
                    edges.append((f"L{i}N{j}", f"L{i+1}N{k}"))
        else:  # 其他情况
            for j in range(layers[i]):
                for k in range(layers[i+1]):
                    edges.append((f"L{i}N{j}", f"L{i+1}N{k}"))

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # 绘制网络
    nx.draw(G, pos, ax=ax, node_color='lightblue', node_size=900,
            edge_color='gray', arrowsize=10)

    # 节点标签
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

    # 手动加权重标签
    weight_index = 0
    for i in range(len(layers)-1):
        if i == 0 and len(layers) == 3:  # 输入层到隐层
            for j in range(layers[i]):
                for k in range(layers[i+1]-1):
                    start_node = f"L{i}N{j}"
                    end_node = f"L{i+1}N{k}"
                    x_start, y_start = pos[start_node]
                    x_end, y_end = pos[end_node]
                    x_label = x_start + 0.2 * (x_end - x_start)
                    y_label = y_start + 0.2 * (y_end - y_start)
                    # 区分 θ1 / θ2
                    label_prefix = "θ1" if k == 0 else "θ2"
                    ax.text(x_label, y_label, f"{label_prefix}={weights[weight_index]:.1f}",
                            fontsize=10, verticalalignment='center', horizontalalignment='center')
                    weight_index += 1
        else:  # 隐层到输出层
            for j in range(layers[i]):
                for k in range(layers[i+1]):
                    start_node = f"L{i}N{j}"
                    end_node = f"L{i+1}N{k}"
                    x_start, y_start = pos[start_node]
                    x_end, y_end = pos[end_node]
                    x_label = x_start + 0.2 * (x_end - x_start)
                    y_label = y_start + 0.2 * (y_end - y_start)
                    ax.text(x_label, y_label, f"θ={weights[weight_index]:.1f}",
                            fontsize=10, verticalalignment='center', horizontalalignment='center')
                    weight_index += 1

    ax.set_title(title)
    ax.axis('off')
    plt.show()

# ========== 网络结构 ==========
# 单层：AND / OR
layers = [3, 1]  # 输入(+1, x1, x2)，输出(h(x))
pos = {}
for i, layer in enumerate(layers):
    for j in range(layer):
        pos[f"L{i}N{j}"] = (i, -j + (layer-1)/2)

# 两层：XOR / XNOR
layers_xor = [3, 3, 1]  # 输入(+1, x1, x2)，隐层(y1, y2, +1)，输出
pos_xor = {}
for i, layer in enumerate(layers_xor):
    for j in range(layer):
        pos_xor[f"L{i}N{j}"] = (i, -j + (layer-1)/2)

# ========== 权重参数 ==========
# AND: h(x) = step(-1.5 + 1*x1 + 1*x2)
weights_and = [-1.5, 1.0, 1.0]

# OR: h(x) = step(-0.5 + 1*x1 + 1*x2)
weights_or = [-0.5, 1.0, 1.0]

# XOR:
# y1 = OR(x1,x2), y2 = AND(x1,x2)
# h(x) = step(-0.5 + 1*y1 -2*y2)
weights_xor = [
    -0.5, 1.0, 1.0,   # 输入→y1
    -1.5, 1.0, 1.0,   # 输入→y2
    -0.5, 1.0, -2.0   # 隐层→输出
]

# XNOR = NOT(XOR)
# h(x) = step(0.5 -1*y1 + 2*y2)
weights_xnor = [
    -0.5, 1.0, 1.0,   # 输入→y1
    -1.5, 1.0, 1.0,   # 输入→y2
    0.5, -1.0, 2.0    # 隐层→输出
]

# ========== 绘图 ==========
draw_neural_network("AND 神经网络", layers, weights_and, pos)
draw_neural_network("OR 神经网络", layers, weights_or, pos)
draw_neural_network("XOR 神经网络", layers_xor, weights_xor, pos_xor)
draw_neural_network("XNOR 神经网络", layers_xor, weights_xnor, pos_xor)
