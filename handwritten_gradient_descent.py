import numpy as np
from torch.fx.experimental.migrate_gradual_types.constraint import Prod


class Context:
    def __init__(self):
        self.save_tensors = []

    def save_for_backward(self, *data):
        self.save_tensors.extend(data)


class FunctionNode:
    def __init__(self, ctx, function_cls, inputs):
        self.ctx = ctx
        self.function_cls = function_cls
        self.output_variable = None
        self.inputs = inputs
        self.generation = max([x.generation for x in self.inputs if isinstance(x, Variable)]+[-1]) +1

class Function:
    @classmethod
    def apply(cls, *inputs):
        ctx = Context()
        input_data = [
            x.data if isinstance(x, Variable) else np.asarray(x)
            for x in inputs
        ]
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
        # 默认使用numpy格式的数据
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.function_node = function_node
        if function_node is not None:
            function_node.output_variable = self
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

    # 在 Variable 类中添加快捷方式
    def relu(self):
        return ReLU.apply(self)

    def leaky_relu(self):
        return LeakyReLU.apply(self)

    def sum(self):
        # 这里的 self 就是调用 sum 的那个 Variable 对象
        return Sum.apply(self)

    def mean(self):
        return Mean.apply(self)

    def log(self):
        return Log.apply(self)

    def roll(self, shift, axis):
        return Roll.apply(self, shift, axis)

    def prod(self):
        return Product.apply(self)

    def shape(self):
        return self.data.shape


    def enable_history(self):
        self.store_history = True

    def clear_history(self):
        self.grad_history = []

    def clear_grad(self):
        self.grad = None

    def accumulate_grad(self, grad, tag="default"):
        if self.grad is None:
            self.grad = np.array(grad, copy=True)
        else:
            self.grad += grad

        if self.store_history:
            self.grad_history.append({"tag": tag, "grad": self.grad})

    

    def backward(self, grad_output=None, tag="default"):
        # backward必然从根节点开始
        if grad_output is None:
            grad_output = np.ones_like(self.data)

        self.accumulate_grad(grad_output, tag)
        if self.function_node is None:
            return


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
            input_gradients = function_cls.backward(current_fuc_node.output_variable.grad, ctx)

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

        for i, (input_arr, grad_val) in enumerate([(a, grad_a), (b, grad_b)]):
            res_grad = grad_val
            if np.ndim(input_arr) < np.ndim(grad_output):
                res_grad = np.sum(res_grad, axis=tuple(range(np.ndim(grad_output) - np.ndim(input_arr))))

            # 处理形状相同但有维度为 1 的情况
            if res_grad.shape != input_arr.shape:
                axes = tuple(idx for idx, (d_in, d_g) in enumerate(zip(input_arr.shape, res_grad.shape)) if d_in < d_g)
                res_grad = np.sum(res_grad, axis=axes, keepdims=True)

            if i == 0:
                grad_a = res_grad
            else:
                grad_b = res_grad

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
        grad_a = grad_output.copy()
        grad_b = grad_output.copy()

        # 处理广播逻辑：
        # 如果 a 的维度比 grad_output 小，说明 a 被广播了，需要求和还原
        for i, (input_arr, grad_val) in enumerate([(a, grad_a), (b, grad_b)]):
            res_grad = grad_val
            if np.ndim(input_arr) < np.ndim(grad_output):
                res_grad = np.sum(res_grad, axis=tuple(range(np.ndim(grad_output) - np.ndim(input_arr))))

            # 处理形状相同但有维度为 1 的情况
            if res_grad.shape != input_arr.shape:
                axes = tuple(idx for idx, (d_in, d_g) in enumerate(zip(input_arr.shape, res_grad.shape)) if d_in < d_g)
                res_grad = np.sum(res_grad, axis=axes, keepdims=True)

            if i == 0:
                grad_a = res_grad
            else:
                grad_b = res_grad

        return grad_a, grad_b

class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.exp(a)

    @staticmethod
    def backward(grad_output, ctx):
        a, = ctx.save_tensors
        grad = grad_output * np.exp(a)
        return grad

class Product(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return np.prod(a)
    @staticmethod
    def backward(grad_output, ctx):
        a, = ctx.save_tensors

        grad = np.zeros_like(a)

        zero_mask = (a == 0)
        num_zero = np.sum(zero_mask)

        if num_zero == 0:
            total = np.prod(a)
            grad = total / a

        elif num_zero == 1:
            idx = np.where(zero_mask)[0][0]
            prod_non_zero = np.prod(a[~zero_mask])
            grad[idx] = prod_non_zero

        # num_zero >= 2 时，梯度全为 0，已经初始化好了

        return grad_output * grad


class Pow(Function):
    @staticmethod
    def forward(ctx, a, n):
        ctx.save_for_backward(a, n)
        return np.power(a, n)

    @staticmethod
    def backward(grad_output, ctx):
        a, n = ctx.save_tensors
        grad = grad_output * n * np.power(a, n-1)
        return grad, None

class Roll(Function):
    @staticmethod
    def forward(ctx, a, shift, axis):
        ctx.save_for_backward(a, shift, axis)
        return np.roll(a, shift, axis)
    @staticmethod
    def backward(grad_output, ctx):
        a, shift, axis = ctx.save_tensors
        grad = np.roll(grad_output, -shift, axis)
        return grad, None, None

class Log(Function):
    @staticmethod
    def forward(ctx, a):
        # 增加一个极小值 epsilon 防止 log(0)
        eps = 1e-12
        a_clipped = np.clip(a, eps, np.inf)
        ctx.save_for_backward(a_clipped)
        return np.log(a_clipped)
    @staticmethod
    def backward(grad_output, ctx):
        a, = ctx.save_tensors
        grad = grad_output * 1.0 / a
        return grad

class AddN(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.save_for_backward(*inputs)  # 展开存储！
        res = np.sum(inputs, axis=0)  # 更简洁的写法
        return res

    @staticmethod
    def backward(grad_output, ctx):
        inputs = ctx.save_tensors
        grads = []
        for input in inputs:
            grad = grad_output.copy()
            if np.ndim(input) < np.ndim(grad_output):
                # 比如 input 是 (3,), grad 是 (2, 3)，要把第 0 维缩减掉
                diff = np.ndim(grad) - np.ndim(input)
                grad = np.sum(grad, axis=tuple(range(diff)))
            # 处理维度数一致但某维度长度为 1 的情况 (如 (1, 3) vs (5, 3))
            if grad.shape != input.shape:
                axes = []
                for i, (d_in, d_g) in enumerate(zip(input.shape, grad.shape)):
                    if d_in < d_g:
                        axes.append(i)
                if axes:
                    grad = np.sum(grad, axis=tuple(axes), keepdims=True)

            grads.append(grad)
        return grads

class Sum(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.shape) # 存形状更省内存
        return np.sum(a)
    @staticmethod
    def backward(grad_output, ctx):
        a_shape, = ctx.save_tensors
        return np.broadcast_to(grad_output, a_shape)

class Mean(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a.shape)
        return np.mean(a)
    @staticmethod
    def backward(grad_output, ctx):
        a_shape, = ctx.save_tensors
        numel = np.prod(a_shape)
        return np.broadcast_to(grad_output/numel, a_shape)

class MaskSelect(Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(mask,x.shape)
        return x[mask]

    @staticmethod
    def backward(grad_output, ctx):
        mask, x_shape = ctx.save_tensors
        grad = np.zeros(x_shape, dtype=grad_output.dtype)
        grad[mask] = grad_output
        return grad, None

class Combine(Function):
    @staticmethod
    def forward(ctx, z_a, z_b, mask):
        ctx.save_for_backward(mask)
        # 创建空数组
        res = np.zeros(mask.shape, dtype=z_a.dtype)
        # 填入数据
        res[mask] = z_a
        res[~mask] = z_b
        return res

    @staticmethod
    def backward(grad_output, ctx):
        mask, = ctx.save_tensors
        # 将传入的完整梯度拆分回两部分
        grad_a = grad_output[mask]
        grad_b = grad_output[~mask]
        return grad_a, grad_b, None

# 不需要to grid，combine已经完成了
# class ToGrid(Function):
#     @staticmethod
#     def forward(ctx, z):
#         ctx.save_for_backward(z.shape)
#         L = int(np.sqrt(len(z)))
#         z2d = z[:L * L].reshape(L, L)
#         return z2d
#     @staticmethod
#     def backward(grad_output, ctx):
#         z_shape, = ctx.save_tensors
#         grad = grad_output.reshape(-1)
#         n = int(np.prod(z_shape))
#         if grad.size < n:
#             pad_width = n - grad.size
#             grad = np.pad(grad, (0, pad_width), mode='constant')
#
#         out = grad[:n].reshape(z_shape)
#         return out

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

class LeakyReLU(Function):
    @staticmethod
    def forward(ctx, x):
        alpha = 0.1
        ctx.save_for_backward(x,alpha)
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def backward(grad_output, ctx):
        x,alpha = ctx.save_tensors
        x_grad = np.ones_like(x)
        x_grad[x <= 0] = alpha
        return grad_output * x_grad

class Linear:
    def __init__(self, in_features, out_features):
        # 随机初始化权重 (Xavier/He 初始化思路)
        self.W = Variable(np.random.randn(in_features, out_features) * 0.1)
        self.b = Variable(np.zeros(out_features))

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

class NN:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = []
        self.set_all_parameters()

    def set_all_parameters(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                self.parameters += layer.parameters()

    def clear_gradient(self):
        for parameter in self.parameters:
            parameter.clear_grad()

    def forward(self, x):
        data = x
        for layer in self.layers:
          data = layer.forward(data)
        return data

    def backward(self, loss, optimizer):
        self.clear_gradient()
        loss.backward()
        optimizer.optimize(self.layers)


class Optimizer:
    def optimize(self):
        raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.parameters = parameters
        self.lr = lr
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_dict = {id(p): np.zeros_like(p.data) for p in self.parameters}
        self.v_dict = {id(p): np.zeros_like(p.data) for p in self.parameters}
        self.optimize_step = 0

    def optimize(self):
        self.optimize_step += 1
        t = self.optimize_step
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            p_id = id(parameter)
            self.m_dict[p_id] = self.beta1 * self.m_dict[p_id] + (1 - self.beta1) * parameter.grad
            self.v_dict[p_id] = self.beta2 * self.v_dict[p_id] + (1 - self.beta2) * np.square(parameter.grad)
            # deviation correction
            m_hat = self.m_dict[p_id] / (1 - self.beta1 ** t)
            v_hat = self.v_dict[p_id] / (1 - self.beta2 ** t)
            parameter.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

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