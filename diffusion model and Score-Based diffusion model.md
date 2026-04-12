# Derivation: From DDPM to Probability Flow ODE

This derivation bridges the discrete Markov chain of DDPM with continuous Stochastic Differential Equations (SDEs) and ultimately deterministic Ordinary Differential Equations (ODEs).

## 1. Discrete DDPM (Forward Process)
Given a variance schedule $\beta_t \in (0, 1)$ and $\alpha_t = 1 - \beta_t$:
The single-step transition is:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t \mathbf{I})$$

Using the reparameterization trick, the state update is:
$$x_t = \sqrt{1 - \beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \mathbf{I})$$

## 2. Continuous Limit (VP-SDE)
We map the discrete steps to a continuous time domain $t \in [0, 1]$. Let the step size $\Delta t \to 0$ and define $\beta_t \approx \beta(t)\Delta t$.

The change in state over $\Delta t$ is:
$$\Delta x = x_{t+\Delta t} - x_t = \sqrt{1 - \beta(t)\Delta t}x_t + \sqrt{\beta(t)\Delta t}\epsilon - x_t$$

Using the Taylor expansion $\sqrt{1 - z} \approx 1 - \frac{1}{2}z$ as $z \to 0$:
$$\Delta x \approx -\frac{1}{2}\beta(t)x_t \Delta t + \sqrt{\beta(t)}\sqrt{\Delta t}\epsilon$$

In the limit $\Delta t \to 0$, the noise term $\sqrt{\Delta t}\epsilon$ becomes the standard Brownian motion increment $dW_t$. This yields the Variance Preserving SDE (VP-SDE):
$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)}dW_t$$

## 3. Fokker-Planck Equation (FPE)
The marginal probability density $p_t(x)$ of the VP-SDE evolves according to the Fokker-Planck equation:
$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (f(x,t)p_t) + \frac{1}{2}g^2(t)\nabla_x^2 p_t$$

Substitute the drift $f(x,t) = -\frac{1}{2}\beta(t)x$ and diffusion $g(t) = \sqrt{\beta(t)}$:
$$\frac{\partial p_t}{\partial t} = -\nabla_x \cdot \left( -\frac{1}{2}\beta(t)x p_t \right) + \frac{1}{2}\beta(t)\nabla_x^2 p_t$$

## 4. Probability Flow ODE (PF-ODE)
We rewrite the Laplacian (diffusion term) using the identity $\nabla_x^2 p_t = \nabla_x \cdot (p_t \nabla_x \log p_t)$:
$$
\begin{aligned}
\frac{\partial p_t}{\partial t} &= -\nabla_x \cdot \left( -\frac{1}{2}\beta(t)x p_t \right) + \frac{1}{2}\beta(t)\nabla_x \cdot (p_t \nabla_x \log p_t) \\
&= -\nabla_x \cdot \left[ \left( -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t)\nabla_x \log p_t \right) p_t \right]
\end{aligned}
$$

This matches the Liouville equation for a deterministic system: $\frac{\partial p_t}{\partial t} = -\nabla_x \cdot (v p_t)$. 
By extracting the equivalent vector field $v(x,t)$, we obtain the deterministic PF-ODE that shares the exact same marginal probability densities as the SDE:
$$dx = \left[ -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t)\nabla_x \log p_t(x) \right] dt$$