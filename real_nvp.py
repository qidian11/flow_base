import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1,36)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.layer2 = nn.Linear(36,3)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.layer3 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        return x

x = torch.unsqueeze(torch.linspace(-100, 100, 100), dim=1)
y = x.pow(2) # + 0.1 * torch.normal(torch.zeros(*x.size()))

net = SimpleNet()
optimizer = optim.Adam(net.parameters(),lr=0.01)

# 均方误差损失函数 (Mean Squared Error)
loss_func = nn.MSELoss()

print("start training")
for epoch in range(30000):
    prediction  = net.forward(x)
    loss = loss_func(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Step {epoch}, Loss: {loss.item():.4f}")


x_test = torch.unsqueeze(torch.linspace(-500, 500, 100), dim=1)
y_test = x_test.pow(2) # + 0.1 * torch.normal(torch.zeros(*x_test.size()))

prediction  = net.forward(x_test)
loss = loss_func(prediction, y_test)
print(f"Test Loss: {loss.item():.4f}")
plt.plot(x_test.numpy(), prediction.detach().numpy())
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Network Output Function")
plt.show()

