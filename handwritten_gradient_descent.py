import numpy as np


class Context:
    def __init__(self):
        self.save_tensors = []

    def save_for_backward(self, *data):
        self.save_tensors.extend(data)


class FunctionNode:
    def __init__(self, ctx, function_cls, inputs):
        self.ctx = ctx
        self.function_cls = function_cls
        self.variable_cls = None
        self.inputs = inputs
        self.generation = max([x.generation for x in self.inputs if isinstance(x, Variable)]+[-1]) +1

class Function:
    @classmethod
    def apply(cls, *inputs):
        ctx = Context()
        input_data = [x.data if isinstance(x, Variable) else x for x in inputs]
        output_data = cls.forward(ctx, *input_data)

        func_node = FunctionNode(ctx, cls, inputs)
        return Variable(output_data, func_node)

    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError

    @staticmethod
    def backward(grad_outputs, ctx):
        raise NotImplementedError

class Variable:
    def __init__(self, data, function_node=None):
        self.data = data
        self.function_node = function_node
        if function_node is not None:
            function_node.variable_cls = self
        self.store_history = False
        self.grad_history = []
        self.grad = None
        self.generation = 0 if function_node is None else function_node.generation

    def __mul__(self, other):
        # 当执行 x * y 时，自动调用 Mul.apply
        return Mul.apply(self, other)

    def __matmul__(self, other):
        # 当执行 x @ y 时，自动调用 MatMul.apply
        return MatMul.apply(self, other)

    def __add__(self, other):
        # 假设你定义了 Add 类
        return Add.apply(self, other)

    def __radd__(self, other):
        # 当执行 2.0 + self 时被触发
        # 因为加法满足交换律 (a + b == b + a)
        # 所以直接复用已经写好的 __add__ 逻辑即可
        return self.__add__(other)

    def __rmul__(self, other):
        # 当执行 2.0 * self 时被触发
        return self.__mul__(other)


    def enable_history(self):
        self.store_history = True

    def clear_history(self):
        self.grad_history = []

    def clear_grad(self):
        self.grad = None

    def accumulate_grad(self, grad, tag="default"):
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.store_history:
            self.grad_history.append({"tag": tag, "grad": self.grad})

    

    def backward(self, grad_output=None, tag="default"):
        # backward必然从根节点开始
        if grad_output is None:
            grad_output = np.ones_like(self.data)

        if self.function_node is None:
            return
        self.accumulate_grad(grad_output, tag)

        fuc_node_list = []
        seen_nodes_set = set()
        def add_node(function_node):
            if function_node not in seen_nodes_set:
                fuc_node_list.append(function_node)
                seen_nodes_set.add(function_node)
                fuc_node_list.sort(key=lambda x: x.generation)

        add_node(self.function_node)
        while fuc_node_list:
            current_fuc_node = fuc_node_list.pop()
            function_cls = current_fuc_node.function_cls
            ctx = current_fuc_node.ctx
            inputs = current_fuc_node.inputs
            input_gradients = function_cls.backward(current_fuc_node.variable_cls.grad, ctx)

            if not isinstance(input_gradients, tuple):
                input_gradients = (input_gradients,)

            for idx, input_variable in enumerate(inputs):
                if isinstance(input_variable, Variable):
                    input_variable.accumulate_grad(input_gradients[idx], tag)

                    if input_variable.function_node is not None:
                        add_node(input_variable.function_node)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b

    @staticmethod
    def backward(grad_output, ctx):
        a, b = ctx.save_tensors

        # 基础梯度计算
        grad_a = grad_output * b
        grad_b = grad_output * a


        if np.ndim(a) < np.ndim(grad_a):
            grad_a = np.sum(grad_a)
        if np.ndim(b) < np.ndim(grad_b):
            grad_b = np.sum(grad_b)

        return grad_a, grad_b

class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(grad_output, ctx):
        a, b = ctx.save_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a + b

    @staticmethod
    def backward(grad_output, ctx):
        a, b = ctx.save_tensors
        grad_a = grad_output
        grad_b = grad_output

        # 处理广播逻辑：
        # 如果 a 的维度比 grad_output 小，说明 a 被广播了，需要求和还原
        if np.ndim(a) < np.ndim(grad_output):
            grad_a = np.sum(grad_a)
        elif a.shape != grad_output.shape:
            # 处理形状相同但某些维度为 1 的情况 (例如 (3, 3) + (3, 1))
            # 找到那些长度为 1 的轴并求和，保持维度
            axes = tuple(i for i, (d_a, d_g) in enumerate(zip(a.shape, grad_output.shape)) if d_a < d_g)
            grad_a = np.sum(grad_a, axis=axes, keepdims=True)

        # 对 b 执行同样的逻辑
        if np.ndim(b) < np.ndim(grad_output):
            grad_b = np.sum(grad_b)
        elif b.shape != grad_output.shape:
            axes = tuple(i for i, (d_b, d_g) in enumerate(zip(b.shape, grad_output.shape)) if d_b < d_g)
            grad_b = np.sum(grad_b, axis=axes, keepdims=True)

        return grad_a, grad_b

# test
# x = Variable(2.0)
# y = Variable(3.0)
# # z = x * y
# z = x*y
# z = 2*x*z
# z.backward()
#
# print(f"x grad: {x.grad}") # 预期 3.0
# print(f"y grad: {y.grad}") # 预期 2.0
#
# A = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# B = Variable(np.array([[7, 8], [9, 10], [11, 12]]))
#
# # 前向传播
# Z = (x+A) @ (3*B)
#
#
# # 反向传播
# Z.backward()
#
# print("Gradient of A:\n", A.grad)
# print("Gradient of B:\n", B.grad)

class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return np.maximum(0, x)

    @staticmethod
    def backward(grad_output, ctx):
        x, = ctx.save_tensors
        grad_x = grad_output.copy()
        grad_x[x <= 0] = 0
        return grad_x

# 在 Variable 类中添加快捷方式
def relu(x):
    return ReLU.apply(x)

class Linear:
    def __init__(self, in_features, out_features):
        # 随机初始化权重 (Xavier/He 初始化思路)
        self.W = Variable(np.random.randn(in_features, out_features) * 0.1)
        self.b = Variable(np.zeros(out_features))

    def forward(self, x):
        # 使用你重载过的 @ 和 + 运算符
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

class NN:
    def setLayers(self, *layers):
        self.layers = layers

    def forward(self, x):
        data = x
        for layer in self.layers:
          data = layer.forward(data)
        return data

    def backward(self, loss, optimizer):
        loss.backward()
        optimizer.optimize(self.layers)


class Optimizer:
    pass

# 1. 准备数据
X_train = Variable(np.array([[1.0], [2.0], [3.0], [4.0]]))
Y_true = Variable(np.array([[3.0], [5.0], [7.0], [9.0]]))

# 2. 实例化模型
model = Linear(1, 1)
learning_rate = 0.01

# 3. 训练迭代
for epoch in range(100):
    # --- 前向传播 ---
    Y_pred = model.forward(X_train)

    # --- 计算 Loss (均方误差 MSE) ---
    diff = Y_pred + (Y_true * -1.0)  # 简易实现 y_pred - y_true
    loss = (diff * diff)  # 简易平方
    loss_val = np.mean(loss.data)

    # --- 反向传播 ---
    # 每次反向传播前，手动清空之前的梯度 (Zero Grad)
    for p in model.parameters():
        p.grad = None

    loss.backward()

    # --- 梯度下降更新权重 (Optimizer Step) ---
    for p in model.parameters():
        # p.data = p.data - lr * p.grad
        p.data -= learning_rate * p.grad

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss_val:.4f}")

print(f"\n训练完成！权重 W: {model.W.data[0][0]:.2f}, 偏置 b: {model.b.data[0]:.2f}")