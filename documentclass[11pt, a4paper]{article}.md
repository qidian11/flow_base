\documentclass[11pt, a4paper]{article}
\usepackage{tikz}
\usepackage{rotating}
\usetikzlibrary{positioning}
\usepackage{indentfirst}
\usepackage[utf8]{inputenc}
\usepackage{placeins}
\usepackage{float}
\usepackage{pdflscape}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\geometry{a4paper, margin=1in}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{cite}
\usepackage{abstract}
\usepackage{authblk}
\usepackage{xcolor}

\title{\textbf{An Application of Neural Network Techniques to Lattice Quantum Field Theory Simulations: Normalizing Flow Markov Chain Monte Carlo Algorithm for $\phi^4$ Theory}}

\author[1]{Zefeng Wang}
\affil[1]{\textit{}}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
Lattice field theory is currently one of the most important approaches for studying non-perturbative problems in quantum field theory. In lattice field theory, physical observables are typically obtained by evaluating high-dimensional path integrals over field configurations. This procedure generally requires efficient sampling from probability distributions governed by the Boltzmann weight.

Traditional Markov Chain Monte Carlo (MCMC) methods, such as the Metropolis-Hastings algorithm, Hybrid Monte Carlo (HMC), and Heat Bath algorithms, have been widely employed in lattice field theory simulations. However, as the system approaches the continuum limit or a critical point, or when field configurations are separated into different topological sectors, these conventional methods often suffer from increasing autocorrelation times, critical slowing down, and topological freezing. Such issues can significantly reduce sampling efficiency and limit the accuracy of numerical calculations.

With the rapid development of artificial intelligence in recent years, a number of AI-based sampling techniques have emerged as promising alternatives. Among them, Normalizing Flows provide a new framework for efficient sampling by constructing an invertible mapping from a simple prior distribution to a complex target distribution. This approach allows the exact probability density of generated samples to be computed, making it particularly attractive for applications in lattice field theory.

This work presents a study of the Flow-based Generative Model proposed in the paper \textit{Flow-based Generative Models for Markov Chain Monte Carlo in Lattice Field Theory}~\cite{Albergo2019}.. Building upon the original framework, we improve the model by incorporating symmetries of the field distribution and employing free-field prior distributions. Furthermore, we compare the proposed model with traditional sampling algorithms in terms of autocorrelation times for physical observables such as correlation functions and effective masses, as well as the computational cost required for sample generation.

\end{abstract}




\section{Introduction}

\section{Physical Background}
Before introducing the application of Normalizing Flows (NF) in Quantum Field Theory (QFT), it is essential to establish the underlying physical and mathematical framework. This section outlines the transition from the continuum path integral formulation of physical observables to their discrete representations. We will first discuss the evaluation of observables via the path integral in Euclidean spacetime, followed by the lattice discretization of fields. Finally, we will specify the action for the two-dimensional $\phi^4$ theory on the lattice, which serves as the foundational model for our subsequent algorithmic implementations.

\subsection{Physical Observables and Path Integrals}
In Quantum Field Theory (QFT), we aim to calculate the probability amplitude $\mathcal{A}$ for the evolution from the initial state $|\phi_i, t_i \rangle$ to the final state $|\phi_f, t_f \rangle$:
\begin{equation}
    \mathcal{A} = \langle \phi_f, t_f |e^{-i\hat{H}T} |\phi_i, t_i \rangle,
    \label{eq:evolution_amplitude}
\end{equation}
where $|\phi_i, t_i\rangle$ and $|\phi_f, t_f\rangle$ are the eigenstates of the field operator $\hat{\phi}$ at times $t_i$ and $t_f$. $\hat{H}$ is the Hamiltonian operator of the system (for the real scalar field theory considered here, it takes the form $\hat{H} = \int d^3x \left[ \hat{\pi}^2 + (\nabla\hat{\phi})^2 + m^2\hat{\phi}^2 + \lambda\hat{\phi}^4 \right]$), and $T = (t_f-t_i)$.

By dividing the total time $T$ into $N$ intervals of spacing $\epsilon$ and inserting a complete set of orthonormal field bases $I = \int \mathcal{D}\phi_k \, |\phi_k \rangle \langle \phi_k|$ at each intermediate time slice, we obtain:
\begin{equation}
    \mathcal{A} = \int \left( \prod_{k=1}^{N-1} \mathcal{D}\phi_k \right) \prod_{k=0}^{N-1} \langle \phi_{k+1} | e^{-i\hat{H}\epsilon} | \phi_k \rangle.
    \label{eq:time_slices}
\end{equation}

For a specific infinitesimal evolution matrix element $\langle \phi_{k+1} | e^{-i\hat{H}\epsilon} | \phi_k \rangle$, we further insert a complete set of orthonormal momentum bases $I = \int \mathcal{D}\pi_k \, |\pi_k \rangle \langle \pi_k|$:
\begin{equation}
    \langle \phi_{k+1} | e^{-i\hat{H}\epsilon} | \phi_k \rangle = \int \mathcal{D}\pi_k \, \langle \phi_{k+1} | \pi_k \rangle \langle \pi_k | e^{-i\hat{H}\epsilon} | \phi_k \rangle.
    \label{eq:matrix_element_momentum}
\end{equation}

Since $\epsilon$ is infinitesimal, we can apply the Hamiltonian operator $\hat{H} = \int d^3x \, \mathcal{H}(\hat{\phi}, \hat{\pi})$ directly onto the eigenstates on both sides, replacing the operator with the classical Hamiltonian density $\mathcal{H}(\phi_k, \pi_k)$. At this point, the inner product of the field state and the momentum state naturally remains in the matrix element. In QFT, the inner product of the field and momentum at every spatial point contributes a plane wave phase, and the inner product over all space can be written as:
\begin{equation}
    \langle \phi_{k+1} | \pi_k \rangle \langle \pi_k | \phi_k \rangle = \exp\left( i \int d^3x \, \pi_k(\mathbf{x}) \phi_{k+1}(\mathbf{x}) \right) \exp\left( -i \int d^3x \, \pi_k(\mathbf{x}) \phi_k(\mathbf{x}) \right).
    \label{eq:inner_product}
\end{equation}

By combining these two exponential terms, the momentum-field difference term related to the spatial integral is naturally extracted. Combining this with the Hamiltonian part, we obtain the integral expression for this matrix element:
\begin{equation}
    \langle \phi_{k+1} | e^{-i\hat{H}\epsilon} | \phi_k \rangle \approx \int \mathcal{D}\pi_k \, \exp\left\{ i \int d^3x \left[ \pi_k(\mathbf{x}) (\phi_{k+1}(\mathbf{x}) - \phi_k(\mathbf{x})) - \epsilon \mathcal{H}(\phi_k, \pi_k) \right] \right\}.
    \label{eq:matrix_element_integral}
\end{equation}

Combining all time slices and taking the continuum limit ($N \to \infty$, $\epsilon \to 0$), the functional measure $\int \mathcal{D}\phi$ corresponds to $\lim_{N \to \infty} \prod_{k=1}^{N-1} \mathcal{D}\phi_k$, and $\int \mathcal{D}\pi$ corresponds to $\lim_{N \to \infty} \prod_{k=0}^{N-1} \mathcal{D}\pi_k$. The difference term in the exponent transforms into the spacetime integral $dt \int d^3x \, \pi \dot{\phi}$ in the limit.

For $\phi^4$ theory, the Hamiltonian density is given by:
\begin{equation}
    \mathcal{H} = \pi^2 + (\nabla\phi)^2 + m^2\phi^2 + \lambda\phi^4.
    \label{eq:phi4_hamiltonian}
\end{equation}

Substituting Eq.~(\ref{eq:phi4_hamiltonian}) into the expression, we obtain the phase-space path integral formulation:
\begin{equation}
    \int \mathcal{D}\phi \mathcal{D}\pi \, \exp\left[ i \int_{t_i}^{t_f} d^4x \left( \pi \dot{\phi} - \pi^2 - (\nabla\phi)^2 - m^2\phi^2 - \lambda\phi^4 \right) \right].
    \label{eq:phase_space_path_integral}
\end{equation}

In this exponent, the integral over the momentum $\pi$ is a standard quadratic Gaussian integral. After completing the square and integrating out all momentum degrees of freedom over $\int \mathcal{D}\pi$, the results of the Gaussian integration are absorbed into the normalization constant, and the Hamiltonian form naturally transitions into the Lagrangian density $\mathcal{L}$. Finally, we obtain the path integral formulation based purely on field configurations:
\begin{equation}
    \mathcal{A} = \langle \phi_f, t_f | e^{-i\hat{H}T} | \phi_i, t_i \rangle = \int_{\phi(t_i)=\phi_i}^{\phi(t_f)=\phi_f} \mathcal{D}\phi \, \exp\left( i \int_{t_i}^{t_f} d^4x \, \mathcal{L} \right) = \int_{\phi_i, t_i}^{\phi_f, t_f} \mathcal{D}\phi \, e^{iS[\phi]}.
    \label{eq:final_path_integral}
\end{equation}

By introducing a small displacement of time toward the imaginary axis ($t \to t(1-i\epsilon)$) and expanding the field eigenstates in terms of energy eigenstates, we can project out the excited states by taking the limit as time approaches infinity, thereby isolating the ground state. Employing a similar methodology, the vacuum expectation value of a physical observable $\mathcal{O}$ is given by:
\begin{equation}
    \langle \Omega | T\{ \mathcal{O}[\phi] \} | \Omega \rangle = \lim_{T \to \infty(1-i\epsilon)} \frac{\int \mathcal{D}\phi \, \mathcal{O}[\phi] \exp\left( i \int_{-T}^{T} dt \int d^3x \, \mathcal{L} \right)}{\int \mathcal{D}\phi \, \exp\left( i \int_{-T}^{T} dt \int d^3x \, \mathcal{L} \right)},
    \label{eq:vacuum_exp_minkowski}
\end{equation}
where $|\Omega\rangle$ represents the energy ground state, which is the vacuum state, and $T$ is the time-ordering operator.

Because the path integral in Minkowski spacetime contains a complex phase $e^{iS}$, it manifests as a highly oscillatory integral, rendering it exceedingly difficult to evaluate via numerical techniques (such as Monte Carlo integration). To resolve this, we can analytically continue the integral across the complex plane and perform a Wick rotation, rotating the time coordinate $t$ to the imaginary axis such that $t = -i\tau$. 

Invoking Cauchy's integral theorem in the complex plane, we can successfully transform the path integral from Minkowski spacetime to Euclidean space. This yields the Euclidean partition function:
\begin{equation}
    Z = \int \mathcal{D}\phi \, \exp\left( - \int_{-\infty}^{\infty} d\tau \int d^3x \, \mathcal{L}_E \right) = \int \mathcal{D}\phi \, e^{-S_E[\phi]}.
    \label{eq:euclidean_partition}
\end{equation}

Consequently, we obtain the path integral expectation value for the observable $\mathcal{O}$ in Euclidean spacetime:
\begin{equation}
    \langle \mathcal{O} \rangle_E = \frac{\int \mathcal{D}\phi \, \mathcal{O}[\phi] \exp\left( - \int_{-\infty}^{\infty} d\tau \int d^3x \, \mathcal{L}_E \right)}{\int \mathcal{D}\phi \, \exp\left( - \int_{-\infty}^{\infty} d\tau \int d^3x \, \mathcal{L}_E \right)} = \frac{\int \mathcal{D}\phi \, \mathcal{O}[\phi] e^{-S_E[\phi]}}{\int \mathcal{D}\phi \, e^{-S_E[\phi]}}.
    \label{eq:euclidean_expectation}
\end{equation}
For the two-dimensional $\phi^4$ theory, the Euclidean action is given by $S_E[\phi] = \int d^2x \left[ \partial_\mu \phi \partial_\mu \phi + m^2 \phi^2 + \lambda \phi^4 \right]$. Consequently, the term $e^{-S_E[\phi]}$ takes the specific form $\exp\left( - \int d^2x \left[ \partial_\mu \phi \partial_\mu \phi + m^2 \phi^2 + \lambda \phi^4 \right] \right)$.






\subsection{Lattice Field Theory}

The fundamental formulation of lattice field theory is based on the discretization of continuous $d$-dimensional Euclidean spacetime into a finite lattice. Supposing the lattice spacing is $a$ and there are $L$ lattice sites along each direction, the total volume is given by $V = (aL)^d$. The continuous coordinates $x$ are replaced by discrete lattice positions:
\begin{equation}
x_\mu = a n_\mu, \quad n_\mu = 0,1,\ldots,L-1
\end{equation}

Accordingly, the continuous field variables $\phi(x)$ map to discrete variables $\phi_x$ defined on the lattice sites. In the following, we work in lattice units with the lattice spacing set to unity ($a=1$). To mitigate finite-volume boundary effects, periodic boundary conditions are conventionally employed in numerical simulations:

\begin{equation}
\phi_{x+L\hat{\mu}}=\phi_x.
\end{equation}
where $\hat{\mu}$ denotes the unit vector in the $\mu$-th direction.

Upon lattice discretization, for a real scalar field, the continuous path integral reduces to a finite-dimensional integral:

\begin{equation}
\int \mathcal{D}\phi
\quad \longrightarrow \quad
\prod_{x \in \Lambda}
\int_{-\infty}^{+\infty}
d\phi_x .
\end{equation}
where $\Lambda$ denotes the set of all lattice sites.



\subsection{Two-Dimensional scalar $\phi^4$ Theory on the Lattice}

In the continuum, we consider the two-dimensional Euclidean action for the $\phi^4$ theory, conventionally expressed without the fractional normalization factors:
\begin{equation}
S_E[\phi] = \int d^2x \left[ \partial_\mu \phi \partial_\mu \phi + m^2 \phi^2 + \lambda \phi^4 \right],
\end{equation}
where the parameters $m^2$ and $\lambda$ are the bare mass squared and bare coupling, 

By applying integration by parts to the kinetic term, it can be separated into a total derivative (divergence term) and a second-order derivative term:
\begin{equation}
\int d^2x \, (\partial_\mu \phi)(\partial_\mu \phi) = \int d^2x \left[ \partial_\mu(\phi \partial_\mu \phi) - \phi \partial_\mu \partial_\mu \phi \right],
\end{equation}
Assuming periodic boundary conditions, the surface integral associated with the total derivative vanishes. The kinetic term is thus reduced to $-\int d^2x \, \phi \partial^2 \phi$.

Discretizing the spacetime onto a lattice (and setting the lattice spacing $a=1$), the continuous second derivative $-\partial^2$ is replaced by the finite-difference representation. The lattice action takes the form:
\begin{equation}
S(\phi) = \sum_x \left( \sum_y \phi(x)\Box(x,y)\phi(y) + m^2\phi(x)^2 + \lambda\phi(x)^4 \right),
\end{equation}
where the lattice d'Alembert operator is defined by:
\begin{equation}
\sum_y \Box(x,y)\phi(y) = \sum_\mu (2\phi(x) - \phi(x-\hat{\mu}) - \phi(x+\hat{\mu})).
\end{equation}

Within this lattice field theory framework, the expectation value of a physical observable $\mathcal{O}$ can be expressed as

\begin{equation}
\langle \mathcal{O} \rangle
=
\frac{
\int \left( \prod_{x \in \Lambda} d\phi_x \right)
\mathcal{O}[\phi]\,
e^{-S_E[\phi]}
}{
\int \left( \prod_{x \in \Lambda} d\phi_x \right)
e^{-S_E[\phi]}
}
=
\frac{1}{Z}
\int
\left(
\prod_{x \in \Lambda}
d\phi_x
\right)
\mathcal{O}[\phi]\,
e^{-S_E[\phi]}
=
\int
\left(
\prod_{x \in \Lambda}
d\phi_x
\right)
\mathcal{O}[\phi]\,
p(\phi).
\end{equation}

\noindent Here, $p(\phi)=\frac{e^{-S_E[\phi]}}{Z}$ denotes the probability distribution of field configurations, where $Z$ is the partition function defined in Eq.~(\ref{eq:euclidean_partition}).


Evaluating this high-dimensional integral analytically is generally intractable due to the interaction term $\lambda\phi^4$. Instead, the expectation value is estimated numerically using Monte Carlo sampling from the probability distribution $p(\phi)$:

\begin{equation}
\langle \mathcal{O} \rangle
\approx
\frac{1}{N}
\sum_{n=1}^{N}
\mathcal{O}[\phi^{(n)}],
\end{equation}

\noindent where $\phi^{(n)}$ represents the $n$-th discrete field configuration sampled according to the Boltzmann probability distribution $p(\phi)$, and $N$ denotes the total number of sampled configurations.

Generating field configurations according to the target distribution $p(\phi)$ is the central task of lattice field theory simulations. Traditionally, this is achieved using Markov Chain Monte Carlo (MCMC) methods, such as the Hybrid Monte Carlo (HMC) algorithm, which will be briefly reviewed in the next section.




\section{Traditional sampling methods}

\subsection{Markov Chain Monte Carlo (MCMC)}
Markov Chain Monte Carlo (MCMC) methods constitute a class of numerical algorithms designed for random sampling from complicated probability distributions. When attempting to sample from the aforementioned probability distribution of field configurations, direct sampling is typically unfeasible because the partition function (as defined in Eq.~\ref{eq:euclidean_partition}) is computationally intractable to evaluate. Direct sampling refers to generating independent field configurations distributed according to the target probability distribution $p(\phi)$ in a single step. For the interacting lattice field theory considered here, such a procedure is generally unavailable because the target distribution is high-dimensional and only known up to the normalization factor $Z$. Consequently, we construct a sampling method called Markov chain such that its stationary distribution---the long-term equilibrium of the chain---converges exactly to our desired target distribution.


Specifically, Markov chain constructs a state sequence $\phi_1 \to \phi_2 \to \dots \to \phi_n$ as illustrated in Fig.~\ref{fig:markov_chain}. 
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{markov_chain.png}
    \caption{Markov chain.}
    \label{fig:markov_chain}
\end{figure}
In this sequence, the probability of the next state $\phi_{t+1}$ occurring depends only on the current state $\phi_{t}$, which we call the transition probability $T(\phi' | \phi)$. Assuming the probability distribution of current states is $P_t(\phi)$, after one evolution of the transition probability $T(\phi' | \phi)$, the probability $P_{t+1}(\phi')$ of being in state $\phi'$ at step $t+1$ is equal to the sum of the probabilities of transitioning from all possible previous states $\phi$:
\begin{equation}
P_{t+1}(\phi') = \int P_t(\phi) T(\phi' | \phi) d\phi
\end{equation}
When the system is in a stationary state, that is, the probability distribution of the system at each step is equal, $P_t(\phi)$=$P_{t+1}(\phi)$=$p(\phi)$, then:
\begin{equation}
p(\phi') = \int p(\phi) T(\phi' | \phi) d\phi,
\end{equation}
which is stationary dist condition.

To ensure that this sequence chain eventually converges to the desired state distribution $p(\phi)$, the transition probability $T(\phi' | \phi)$ should satisfy the relation called the detailed balance condition:
\begin{equation}
p(\phi) T(\phi' | \phi) = p(\phi') T(\phi | \phi')
\label{eq:balance}
\end{equation}
This is the sufficient condition of stationary distribution.

Integrating both sides of the equation over $\phi$
yields the stationary distribution.



\subsection{Metropolis-Hastings(MH)}
This part we will introduce Metropolis-Hastings (MH) algorithm. The Metropolis-Hastings algorithm is a widely used MCMC method that generates samples from a target distribution through a proposal-and-acceptance procedure satisfying detailed balance, ensuring convergence to the desired equilibrium distribution.

In the Metropolis-Hastings algorithm, the transition probability $T(\phi'|\phi)$ is factored into two components:
\begin{equation}
T(\phi' | \phi)
=
g(\phi' |\phi)
\times
A(\phi' |\phi)
\label{eq:transition}
\end{equation}

where the proposal distribution $g(\phi' | \phi)$ and the acceptance probability $A(\phi' | \phi)$.

The proposal distribution
$g(\phi' \mid \phi)$
is chosen by ourself.
For example, we may use a symmetric random walk,
in which case:$g(\phi' | \phi) = g(\phi | \phi')$

\noindent For the acceptance probability $A(\phi' | \phi)$, the MH algorithm cleverly designs:
\begin{equation}
A(\phi' | \phi) = \min\left(1, \frac{p(\phi') g(\phi | \phi')}{p(\phi) g(\phi' | \phi)}\right)
\label{eq:acc}
\end{equation}

\noindent By substituting Eq.~(\ref{eq:acc}) into Eq.~(\ref{eq:transition}), we find this perfectly satisfying the detailed balance condition Eq.~(\ref{eq:balance}).

Combining the proposal distribution with the acceptance probability in
Eq.~\eqref{eq:acc}, the MH algorithm constructs a
transition kernel that satisfies the detailed balance condition
Eq.~\eqref{eq:balance}. Consequently, the target distribution
$p(\phi)$ is preserved as the stationary distribution of the Markov chain.
After sufficient equilibration, the generated field configurations may
therefore be regarded as samples drawn from the desired distribution.


\subsection{Hamiltonian Monte Carlo (HMC) Algorithm}

Although the Metropolis--Hastings algorithm guarantees convergence to the
desired target distribution, its efficiency can become poor in
high-dimensional systems due to low acceptance rate when using a non-sophisticated proposal distribution $g(\phi'|\phi)$. The Hamiltonian Monte Carlo (HMC) algorithm addresses this
problem by introducing auxiliary momentum variables and generating
proposals through deterministic Hamiltonian dynamics.

The starting point of HMC is to modify
\begin{equation}
p(\phi)
=
\frac{1}{Z}
e^{-S(\phi)}.
\end{equation}

Then we introduce an auxiliary momentum variable
$\pi_x$ for every field degree of freedom $\phi_x$.
Each momentum variable is drawn independently from a Gaussian
distribution,

\begin{equation}
p(\pi)
\propto
\exp\!\left(
-\frac12
\sum_x \pi_x^2
\right) = e^{-K(\pi)},
\end{equation}
where $K(\pi) = \frac{1}{2}\sum_x \pi_x^2$.

Because the momentum variables are introduced independently of the
field variables, multiplying the target distribution by the momentum
distribution does not change the physics of the original system. The
joint distribution becomes

\begin{equation}
p_{\mathrm{HMC}}(\phi,\pi)
=
\frac{1}{Z_{\mathrm{HMC}}}
e^{-S(\phi)}
e^{-K(\pi)}
= 
\frac{e^{-H(\phi, \pi)}}{Z_{\mathrm{HMC}}},
\end{equation}

where

\begin{equation}
H(\phi,\pi)
=
K(\pi)
+
S(\phi)
\end{equation}

is interpreted as a Hamiltonian.
Integrating out the momentum variables recovers the original target
distribution,

\begin{equation}
\int \mathcal D\pi\,
p_{\mathrm{HMC}}(\phi,\pi)
=
p(\phi).
\end{equation}

Therefore, sampling the joint distribution \(p_{\mathrm{HMC}}(\phi,\pi)\) is
equivalent to sampling the original distribution \(p(\phi)\). We now define the new state $\Phi \equiv (\phi, \pi)$ with Hamiltonian $H(\Phi) = S(\phi) + K(\pi)$. The expectation value becomes
\begin{equation}
\langle \mathcal{O} \rangle = \int \mathcal{D}\Phi \, \mathcal{O}(\Phi) \frac{e^{-H(\Phi)}}{Z_{\mathrm{HMC}}} = \int \mathcal{D}\phi \mathcal{D}\pi \, \mathcal{O}(\phi) \frac{e^{-H(\phi, \pi)}}{Z_{\mathrm{HMC}}}.
\end{equation}
Then Eq.~(\ref{eq:transition}) and Eq.~(\ref{eq:acc}) can be replaced by $g(\Phi'|\Phi)A(\Phi'|\Phi)$, and we apply the MH algorithm on this.

To generate new configurations, HMC evolves the system in the
augmented phase space $(\phi,\pi)$ according to Hamilton's equations
of motion,

\begin{align}
\frac{d\phi_x}{d\tau}
&=
\frac{\partial H}{\partial \pi_x}
=
\pi_x,
\\
\frac{d\pi_x}{d\tau}
&=
-\frac{\partial H}{\partial \phi_x}
=
-\frac{\partial S}{\partial \phi_x},
\end{align}

where $\tau$ is a fictitious simulation time. Starting from an
initial configuration $(\phi,\pi)$, these equations generate a
trajectory in phase space and produce a candidate configuration
$(\phi',\pi')$.

A key property of Hamiltonian dynamics is that it preserves the phase
space volume and conserves the Hamiltonian,

\begin{equation}
H(\phi',\pi')
=
H(\phi,\pi),
\end{equation}
for the exact continuous-time evolution. Consequently, configurations
with large probability density can be connected by long deterministic
trajectories, allowing the algorithm to move efficiently through the
configuration space without relying on a sequence of small random
steps.
In practice, the equations of motion cannot be solved exactly and must
be integrated numerically. HMC therefore employs a reversible and
volume-preserving integrator, most commonly the Leapfrog algorithm.

For a step size $\epsilon$, one Leapfrog update consists of

\begin{align}
\pi_x\!\left(\tau+\frac{\epsilon}{2}\right)
&=
\pi_x(\tau)
-
\frac{\epsilon}{2}
\frac{\partial S}{\partial \phi_x},
\\
\phi_x(\tau+\epsilon)
&=
\phi_x(\tau)
+
\epsilon\,
\pi_x\!\left(\tau+\frac{\epsilon}{2}\right),
\\
\pi_x(\tau+\epsilon)
&=
\pi_x\!\left(\tau+\frac{\epsilon}{2}\right)
-
\frac{\epsilon}{2}
\frac{\partial S}{\partial \phi_x(\tau+\epsilon)}.
\end{align}

\noindent Repeating these updates $L$ times generates a trajectory of length

\begin{equation}
\tau = L\epsilon.
\end{equation}
Because the Leapfrog integrator uses a finite step size, the
Hamiltonian is not conserved exactly. After a trajectory has been
generated, the candidate configuration is therefore subjected to a
Metropolis accept--reject step.

The proposal $(\phi',\pi')$ is accepted with probability

\begin{equation}
P_{\rm acc}
=
\min
\left(
1,
e^{-\Delta H}
\right),
\end{equation}

where

\begin{equation}
\Delta H
=
H(\phi',\pi')
-
H(\phi,\pi).
\end{equation}

If the proposal is rejected, the original configuration is retained.
This acceptance step removes the discretization error introduced by
the numerical integration and ensures that the Markov chain samples
the exact target distribution.
A complete HMC update therefore consists of the following steps:

\begin{enumerate}
\item Draw a fresh momentum field $\pi$ from the Gaussian distribution, where $p(\pi) \propto e^{-K(\pi)}$.

\item Construct the Hamiltonian $H(\phi,\pi)=K(\pi)+S(\phi)$.



\item Evolve $(\phi,\pi)$ using Leapfrog integration for $L$ steps.

\item Obtain a candidate state $(\phi',\pi')$.

\item Accept the proposal with probability
$P_{\rm acc}=\min(1,e^{-\Delta H})$.

\item Discard the momentum variables and retain the accepted field
configuration $\phi'$ as the next Markov-chain state.
\end{enumerate}

By combining Hamiltonian dynamics with a Metropolis correction step,
HMC is able to generate distant proposals with high acceptance rates,
significantly reducing autocorrelation compared with ordinary
random-walk-based MCMC methods.

Although HMC has been highly successful and remains one of the most widely used algorithms in lattice field theory, difficulties can arise as the lattice spacing becomes small. In this regime, the Markov chain may remain in the same topological sector for a long time, making transitions between different sectors increasingly rare. This phenomenon, known as \emph{topological freezing}, leads to long autocorrelation times and reduces the efficiency of sampling.

To address these challenges, researchers have recently explored machine-learning-based approaches for Monte Carlo sampling. Among them, Normalizing Flows have attracted considerable attention and have been applied to the construction of more efficient sampling algorithms. We will introduce the basic idea of Normalizing Flows and discuss their application to lattice field theory.






\subsection{Normalizing Flows}
The idea of Normalizing Flows is as follows. As usual in elementary integration, one of the common methods is integration by substitution. Using the following change of integration variables:
\begin{equation}
    \phi = f(\mathbf{z}), \quad \mathbf{z} = f^{-1}(\phi),
\end{equation}
where $f$ is an invertible function, the partition function $Z = \int \mathcal{D}\phi \, P(\phi)$ can be transformed as:
\begin{equation}
    Z = \int \mathcal{D}\mathbf{z} \left| \det \frac{\partial f(\mathbf{z})}{\partial \mathbf{z}} \right| P(f(\mathbf{z})) \equiv \int \mathcal{D}\mathbf{z} \, V(\mathbf{z}).
    \label{eq:substitution_Z}
\end{equation}
When $V(\mathbf{z})$ becomes a simple function, such as a Gaussian $V(\mathbf{z}) \propto e^{-|\mathbf{z}|^2/2}$, the generation of the field ensemble from the distribution of $V(\mathbf{z})$ becomes easy using the transformation $\phi = f(\mathbf{z})$.

In QFT cases, unfortunately, the exact transformation $f$ is not easily available. However, an approximation with a tolerable computational cost can be available equipped with recent artificial intelligence (AI) and neural network technologies. We will use $f_\theta$ as an approximator for $f$, defined by $\phi = f_\theta(\mathbf{z})$, where $\theta$ is a parameter set of the function. The deviation of $f_\theta$ from the ideal $f$ can be compensated using the MH algorithm. The efficiency gain highly depends on the approximator $f_\theta$.

The construction of this parameterized approximator $f_\theta$ is exactly what Normalizing Flows are designed to achieve. In the context of machine learning, a Normalizing Flow is based on invertible transformations. Its fundamental mechanism perfectly aligns with our mathematical motivation: it utilizes a simple, easy-to-sample prior distribution $r(\mathbf{z})$—which plays the role of our desired simple function $V(\mathbf{z})$—and systematically transforms it into a complex target distribution through a sequence of invertible and differentiable mappings. 

Formally, let
\begin{equation}
    \phi = f_\theta(\mathbf{z}), \quad \mathbf{z} \sim r(\mathbf{z}),
\end{equation}
where $\mathbf{z}$ is the prior variable, $\phi$ represents the generated field configuration, and $f_\theta$ is the invertible transformation. Typically, $f_\theta$ is composed of a sequence of simple invertible transformations (layers):
\begin{equation}
    f_\theta = g_K \circ g_{K-1} \circ \cdots \circ g_1.
\end{equation}
Each individual layer $g_k$ (which will be explicitly implemented as affine coupling layers in the subsequent section) is deliberately designed to be a simple function. This constraint ensures that its Jacobian determinant can be easily and efficiently calculated. Crucially, while each layer is simple, composing a massive number of these simple transformations guarantees that the overall network $f_\theta$ achieves highly expressive power.

Defining the intermediate variables $\mathbf{z}_k$ with $\mathbf{z}_0 = \mathbf{z}$,
\begin{equation}
    \mathbf{z}_k = g_k(\mathbf{z}_{k-1}), \quad k=1,2,\ldots,K,
\end{equation}
we ultimately obtain $\phi = \mathbf{z}_K$. 

Inserting the backward transformation $\mathbf{z} = f_\theta^{-1}(\phi)$ into Eq.~\eqref{eq:substitution_Z}, we can expand the integral into a complete chain:
\begin{equation}
    Z = \int \mathcal{D}\mathbf{z} \, r(\mathbf{z}) = \int \mathcal{D}\phi \left| \det \frac{\partial f_\theta^{-1}(\phi)}{\partial \phi} \right| r(f_\theta^{-1}(\phi)) = \int \mathcal{D}\phi \, q_{\theta}(\phi).
\end{equation}

Ideally, we desire an exact transformation $f$ such that the prior distribution $r(\mathbf{z})$ maps perfectly to the target distribution $p(\phi)$. However, since finding such an exact analytical mapping is generally intractable, we denote the actual model-generated distribution as $q_\theta(\phi)$. Consequently, the relationship between these distributions is given by:
\begin{equation}
    q_\theta(\phi) = r(\mathbf{z}) \left| \det \frac{\partial f_\theta^{-1}(\phi)} {\partial \phi} \right|.
\end{equation}
Equivalently, for the forward transformation $\phi=f_\theta(\mathbf{z})$, using the relation $\left| \det \frac{\partial \mathbf{z}}{\partial \phi} \right| = \left| \det \frac{\partial \phi}{\partial \mathbf{z}} \right|^{-1}$, we have:
\begin{equation}
    \log q_\theta(\phi) = \log r(\mathbf{z}) - \log \left| \det \frac{\partial f_\theta(\mathbf{z})}{\partial \mathbf{z}} \right|.
\end{equation}
Since $f_\theta$ is a composition of multiple transformations, its Jacobian determinant can be decomposed into the product of the Jacobian determinants of each layer:
\begin{equation}
    \log q_\theta(\phi) = \log r(\mathbf{z}) - \sum_{k=1}^{K} \log \left| \det \frac{\partial g_k(\mathbf{z}_{k-1})}{\partial \mathbf{z}_{k-1}} \right|.
\end{equation}
Therefore, as long as the inverse mapping and the Jacobian determinant of each individual layer $g_k$ can be computed efficiently, Normalizing Flows can simultaneously achieve rapid sample generation and exact probability density calculation. Then we can more effectively generate the field configurations.





\subsection{Limitations of Traditional Sampling Methods}

While traditional Markov Chain Monte Carlo (MCMC) methods are fundamental to lattice field theory, they inherently suffer from a critical limitation known as sample autocorrelation. Adjacent samples generated by the Markov chain are not statistically independent. Consequently, the effective sample size is significantly smaller than the total number of generated configurations. Most of the low efficacy comes from this sample autocorrelation:

\begin{enumerate}
    \item \textbf{Thermalization Time:} A substantially long burn-in (thermalization) period is required prior to the actual measurement phase to eradicate any biases introduced by the artificial initial configuration.
    
    \item \textbf{Critical Slowing Down:} As the system approaches a continuous phase transition (critical point), the physical correlation length diverges. This causes the algorithmic autocorrelation time of local update methods to increase dramatically, severely impeding efficiency.
    
    \item \textbf{Topological Freezing:} In certain gauge field theories, MCMC updates struggle to overcome high action barriers to tunnel between different topological sectors, causing the simulation to become "frozen" in a single topological charge sector.
    
    \item \textbf{Diminished Efficiency in Large Dimensions:} As the physical lattice volume scales up, the dimensionality of the configuration space expands exponentially, making efficient phase-space exploration exceedingly difficult due to the critical slowing down and topological freezing.
\end{enumerate}

These formidable challenges have compelled researchers to explore modern machine learning generative models to either augment or completely replace traditional sampling paradigms. Normalizing Flow is one of the candidates for resolving these difficulties equipped with neural network technology.






\section{Neural Networks and Normalizing Flow Algorithm}
In this section, we briefly introduce the basic structure and working principles of neural networks, including the processes of forward propagation and backpropagation. Building upon the foundational concepts of Normalizing Flows introduced previously, we then detail the neural network implementation of specific invertible transformations—namely, affine coupling layers—and define the network training objective based on the Kullback-Leibler divergence.







\subsection{Neural Networks as Parameterized Functions}

From a mathematical perspective, a neural network can be viewed as a highly expressive parameterized function $F_\theta(\mathbf{x})$, where $\mathbf{x}$ denotes the input variables. For our purpose, the neural network is defined as the mapping $F_\theta(\mathbf{x}) = \mathbf{h}^{(L)}$, where $\theta = \{W^{(l)}, \mathbf{b}^{(l)}\}_{l=1}^{L}$ represents the set of all trainable parameters (weights and biases) across all layers.

A neural network is constructed by stacking $L$ sequential layers. Let $\mathbf{h}^{(0)} = \mathbf{x}$ denote the input vector. The activation output $\mathbf{h}^{(l)}$ of the $l$-th layer is recursively defined as:
\begin{equation}
    \mathbf{h}^{(l)} = \sigma^{(l)} \left( W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \right), \quad l=1, 2, \dots, L,
\end{equation}
where $W^{(l)}$ and $\mathbf{b}^{(l)}$ are the weight matrix and bias vector of the $l$-th layer, respectively, $\sigma^{(l)}$ is the nonlinear activation function applied to each element. A ``layer'' refers to a collection of neurons that takes the output of the previous layer $\mathbf{h}^{(l-1)}$ as input and produces a new vector representation $\mathbf{h}^{(l)}$. By stacking these layers, deep neural networks acquire the capacity to approximate highly complex non-linear mappings from the input space to the target space, with the final network output given by:
\begin{equation}
    F_{\theta}(\mathbf{x}) = \mathbf{h}^{(L)}.
\end{equation}

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{full_connection.png}
    \caption{A fully connected neural network.}
    \label{fig:full_connection}
\end{figure}







\FloatBarrier
\subsection{Forward and Backward Propagation}

A neural network represents a parameterized function $F_\theta(\mathbf{x})$, where $\mathbf{x}$ is the input and $\theta$ denotes all trainable parameters in the network. 
This function is introduced for a specific purpose. 
In a general machine learning problem, $F_\theta(\mathbf{x})$ may be used to predict a label, approximate an unknown function, or transform data into a desired output form. 
In the present work, $F_\theta$ is used as part of a normalizing flow: it transforms samples drawn from a simple prior distribution into field configurations that should approximate the target lattice field distribution. 
Therefore, the purpose of training is to tune the parameters $\theta$ so that the output of $F_\theta$ becomes closer to the desired target distribution.

The training process of a neural network typically consists of two phases: forward propagation and backward propagation. 
During forward propagation, the input data sequentially passes through the transformations in each layer to yield the final model output $F_\theta(\mathbf{x})$. 
Based on this output and the desired training objective, the loss function $\mathcal{L}(\theta)$ is computed. 
In our case, this loss measures how far the generated distribution is from the target Boltzmann distribution.

In backward propagation, the chain rule is applied to calculate the gradient of the loss function with respect to the parameters of each layer, $\nabla_\theta \mathcal{L}(\theta)$. 
These gradients indicate how the parameters should be changed in order to reduce the loss. 
The parameters are then updated via gradient descent or its variants:
\begin{equation}
    \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta),
\end{equation}
where $\eta$ is the learning rate. 
By repeating forward propagation, loss evaluation, backward propagation, and parameter updates, the network gradually learns a better function $F_\theta$ for the given purpose.





\FloatBarrier
\subsection{Forward and Backward Propagation}

The training process of a neural network typically consists of two phases: forward propagation and backward propagation. We can intuitively understand this process by drawing an analogy to a physical system: imagine a ball moving on a high-dimensional potential energy surface defined by the parameter space $\theta$, as illustrated in Fig.~\ref{fig:forward_backward}.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{forward_and_backward.png}
    \caption{A physical analogy of the optimization process. Forward propagation determines the ball's current position and potential energy (loss). Backward propagation calculates the negative gradient, driving the parameters along the direction of steepest descent towards the target position.}
    \label{fig:forward_backward}
\end{figure}

During forward propagation, the input data sequentially passes through the transformations in each layer to yield the final model output. In our physical analogy, this step computes the current position of the ball on the potential energy surface. Based on this output, the loss function $\mathcal{L}(\theta)$ is computed. This loss measures how far the generated distribution is from the target Boltzmann distribution, effectively representing the magnitude of the potential energy at the current position. Our ultimate goal is to navigate the surface and find the minimum of this potential energy (the target position).

In backward propagation, the chain rule is applied to calculate the gradient of the loss function with respect to the parameters of each layer, $\nabla_\theta \mathcal{L}(\theta)$. Mathematically, the gradient vector always points in the direction of steepest ascent. Therefore, to minimize the loss, we must move the parameters in the exact opposite direction—the negative gradient ($-\nabla_\theta \mathcal{L}(\theta)$)—which physically corresponds to the direction of steepest descent (the fastest downward path). These negative gradients indicate how the parameters should be changed in order to reduce the loss most efficiently. 

In machine learning, this optimization process is formally called training. The parameter space is then acted upon, moving the parameters along this steepest descent direction via gradient descent or its variants:
\begin{equation}
    \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta),
\end{equation}
where $\eta$ is the learning rate. Using these updated parameters, the network recalculates the ball's new position on the surface. By repeating forward propagation, loss evaluation, backward propagation, and parameter updates, the network gradually learns a better mapping. Because the parameter space $\theta$ is exceptionally huge, many advanced optimization methods have been developed to efficiently navigate this high-dimensional surface. In this paper, we skip the intricate mathematical details of backpropagation; comprehensive discussions and derivations can be found in standard machine learning textbooks.






\subsection{Detail of Normalizing Flow with Neural Networks}

To construct the invertible transformations $g$ introduced previously, we utilize the affine coupling layer. The affine coupling layer is a widely used invertible transformation structure in Normalizing Flows. Its core benefit is that it makes the calculation of the Jacobian determinant extremely simple and efficient.

Its basic idea is to partition the input variables into two parts, where one part remains unchanged, while the other undergoes an affine transformation parameterized by scale and translation functions derived from the first part. Let the input variable be $\bm{\phi} = (\bm{\phi}_a, \bm{\phi}_b)$. The forward transformation of the affine coupling layer is defined as:
\begin{align}
    \mathbf{z}_a &= \bm{\phi}_a, \\
    \mathbf{z}_b &= \bm{\phi}_b \odot \exp \left[ s_\theta(\bm{\phi}_a) \right] + t_\theta(\bm{\phi}_a),
\end{align}
where $\odot$ denotes element-wise multiplication, and $s_\theta(\cdot)$ and $t_\theta(\cdot)$ represent the scale and translation functions, respectively, which are typically parameterized by neural networks. In practice, the functional roles of partitions $a$ and $b$ can be switched in alternating layers to ensure that all field variables are evenly updated.

The inverse transformation $g^{-1}$ can be explicitly written as:
\begin{align}
    \bm{\phi}_a &= \mathbf{z}_a, \\
    \bm{\phi}_b &= \left[ \mathbf{z}_b - t_\theta(\mathbf{z}_a) \right] \odot \exp \left[ - s_\theta(\mathbf{z}_a) \right].
\end{align}
Thus, the affine coupling layer naturally satisfies the invertibility requirement.

Because of this specific partition design, the Jacobian matrix $J = \frac{\partial \mathbf{z}}{\partial \bm{\phi}}$ for the affine coupling layer takes a lower block-triangular form:
\begin{equation}
    J = 
    \begin{pmatrix}
    I & 0 \\
    A & D
    \end{pmatrix},
\end{equation}
where
\begin{equation}
    A = \frac{\partial \mathbf{z}_b}{\partial \bm{\phi}_a}, \quad D = \operatorname{diag} \left( e^{s_{\theta,1}}, e^{s_{\theta,2}}, \ldots \right).
\end{equation}
Therefore, the determinant is simply the product of its diagonal elements:
\begin{equation}
    \det J = \det(D) = \prod_i e^{s_{\theta,i}(\bm{\phi}_a)},
\end{equation}
which yields a highly tractable log-determinant:
\begin{equation}
    \log |\det J| = \sum_i s_{\theta,i}(\bm{\phi}_a).
\end{equation}

Correspondingly, the Jacobian matrix of the inverse transformation exhibits a similar structure:
\begin{equation}
    J^{-1} = \frac{\partial \bm{\phi}}{\partial \mathbf{z}} = 
    \begin{pmatrix}
    I & 0 \\
    B & D^{-1}
    \end{pmatrix},
\end{equation}
where $D^{-1} = \operatorname{diag} \left( e^{-s_{\theta,1}}, e^{-s_{\theta,2}}, \ldots \right)$. Hence,
\begin{equation}
    \log |\det J^{-1}| = -\sum_i s_{\theta,i}(\bm{\phi}_a).
\end{equation}









\subsection{Kullback-Leibler Divergence and Training Objective}
Next, we briefly introduce the Kullback-Leibler (KL) divergence, outline its derivation, and design the loss function by combining the KL divergence with our target distribution.

In statistical analysis, it is often necessary to quantify the similarity between two distributions, and the KL divergence serves exactly this purpose. For two probability distributions $q(\phi)$ and $p(\phi)$, the KL divergence is defined as:
\begin{equation}
    D_{\mathrm{KL}} \left( q || p \right) = \int d\phi\, q(\phi) \log \frac{ q(\phi) }{ p(\phi) }.
\end{equation}
From Jensen's inequality, $\log(\mathbb{E}[x]) \ge \mathbb{E}[\log(x)]$, we obtain:
\begin{equation}
    \log \left( \int d\phi\, q(\phi) \frac{p(\phi)}{q(\phi)} \right) \ge \int d\phi\, q(\phi) \log \frac{p(\phi)}{q(\phi)}.
\end{equation}
Noting that the probability distribution satisfies the normalization condition $\int d\phi\, p(\phi)=1$, we have:
\begin{equation}
    \log \left( \int d\phi\, q(\phi) \frac{p(\phi)}{q(\phi)} \right) = \log (1) = 0.
\end{equation}
Thus,
\begin{equation}
    0 \ge \int d\phi\, q(\phi) \log \frac{p(\phi)}{q(\phi)}.
\end{equation}
Multiplying both sides by $-1$ yields:
\begin{equation}
    D_{\mathrm{KL}}(q||p) = -\int d\phi\, q(\phi) \log \frac{p(\phi)}{q(\phi)} \ge 0.
\end{equation}
Therefore, the KL divergence is always non-negative, and the equality holds if and only if $q(\phi)=p(\phi)$. This demonstrates that the KL divergence effectively measures the discrepancy between two probability distributions. When the KL divergence is 0, the model distribution is equal to the target distribution.

In lattice field theory, the target distribution is typically $p(\phi) = \frac{1}{Z} e^{-S[\phi]}$, and the distribution generated by the Normalizing Flow is denoted as $q_\theta(\phi)$. To make $q_\theta(\phi)$ approximate $p(\phi)$, we minimize the KL divergence as our training objective. Substituting the target and generated distributions into the equation yields:
\begin{equation}
    D_{\mathrm{KL}} \left( q_\theta || p \right) = \int d\phi\, q_\theta(\phi) \left[ \log q_\theta(\phi) + S[\phi] + \log Z \right].
\end{equation}
This can be expressed as an expectation value:
\begin{equation}
    D_{\mathrm{KL}} \left( q_\theta || p \right) = \mathbb{E}_{\phi \sim q_\theta} \left[ \log q_\theta(\phi) + S[\phi] \right] + \log Z.
\end{equation}
Since $\log Z$ is independent of the model parameters $\theta$, this constant term can be ignored during training. The training loss is then defined as:
\begin{equation}
    \mathcal{L}(\theta) = \mathbb{E}_{\phi \sim q_\theta} \left[ \log q_\theta(\phi) + S[\phi] \right].
\end{equation}
Using $\phi=f_\theta(\mathbf{z})$ with $\mathbf{z}\sim r(\mathbf{z})$, this can be further rewritten as:
\begin{equation}
    \mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z} \sim r(\mathbf{z})} \left[ S(f_\theta(\mathbf{z})) + \log q_\theta(f_\theta(\mathbf{z})) \right].
\end{equation}
According to the change of variables formula, 
\begin{equation}
    \log q_\theta(f_\theta(\mathbf{z})) = \log r(\mathbf{z}) - \log \left| \det \frac{\partial f_\theta(\mathbf{z})} {\partial \mathbf{z}} \right|.
\end{equation}
Therefore, the final training objective is:
\begin{align}
    \mathcal{L}(\theta) &= \mathbb{E}_{\mathbf{z} \sim r(\mathbf{z})} \Big[ S(f_\theta(\mathbf{z})) + \log r(\mathbf{z}) \nonumber \\
    &\quad - \log \left| \det \frac{\partial f_\theta(\mathbf{z})} {\partial \mathbf{z}} \right| \Big].
\end{align}

In practice, the expectation value over the prior distribution is evaluated numerically by discretizing the integral using Monte Carlo sampling. For a given batch of $B$ independent samples $\{ \mathbf{z}^{(i)} \}_{i=1}^B$ drawn from the prior distribution $r(\mathbf{z})$, the discretized empirical loss function becomes:
\begin{equation}
    \mathcal{L}(\theta) \approx \frac{1}{B} \sum_{i=1}^B \left[ S(f_\theta(\mathbf{z}^{(i)})) + \log r(\mathbf{z}^{(i)}) - \log \left| \det \frac{\partial f_\theta(\mathbf{z}^{(i)})}{\partial \mathbf{z}^{(i)}} \right| \right].
\end{equation}






\subsection{Kullback-Leibler Divergence and Training Objective}
As we mentioned in the backward propagation section, we need to define a loss function to optimize the parameters. Next, we briefly introduce the Kullback-Leibler (KL) divergence, outline its derivation, and design the loss function by combining the KL divergence with our target distribution.

In statistical analysis, it is often necessary to quantify the similarity between two distributions, and the KL divergence serves exactly this purpose. For two probability distributions $q(\phi)$ and $p(\phi)$, the KL divergence is defined as:
\begin{equation}
    D_{\mathrm{KL}} \left( q || p \right) = \int d\phi\, q(\phi) \log \frac{ q(\phi) }{ p(\phi) }.
\end{equation}
From Jensen's inequality, $\log(\mathbb{E}[x]) \ge \mathbb{E}[\log(x)]$, we obtain:
\begin{equation}
    \log \left( \int d\phi\, q(\phi) \frac{p(\phi)}{q(\phi)} \right) \ge \int d\phi\, q(\phi) \log \frac{p(\phi)}{q(\phi)}.
\end{equation}
Noting that the probability distribution satisfies the normalization condition $\int d\phi\, p(\phi)=1$, we have:
\begin{equation}
    \log \left( \int d\phi\, q(\phi) \frac{p(\phi)}{q(\phi)} \right) = \log (1) = 0.
\end{equation}
Thus,
\begin{equation}
    0 \ge \int d\phi\, q(\phi) \log \frac{p(\phi)}{q(\phi)}.
\end{equation}
Multiplying both sides by $-1$ yields:
\begin{equation}
    D_{\mathrm{KL}}(q||p) = -\int d\phi\, q(\phi) \log \frac{p(\phi)}{q(\phi)} \ge 0.
\end{equation}
Therefore, the KL divergence is always non-negative, and the equality holds if and only if $q(\phi)=p(\phi)$. This demonstrates that the KL divergence effectively measures the discrepancy between two probability distributions. When the KL divergence is 0, the model distribution is equal to the target distribution.

In lattice field theory, the target distribution is typically $p(\phi) = \frac{1}{Z} e^{-S[\phi]}$, and the distribution generated by the Normalizing Flow is denoted as $q_\theta(\phi)$. To make $q_\theta(\phi)$ approximate $p(\phi)$, we minimize the KL divergence as our training objective. Substituting the target and generated distributions into the equation yields:
\begin{equation}
    D_{\mathrm{KL}} \left( q_\theta || p \right) = \int d\phi\, q_\theta(\phi) \left[ \log q_\theta(\phi) + S[\phi] + \log Z \right].
\end{equation}
This can be expressed as an expectation value:
\begin{equation}
    D_{\mathrm{KL}} \left( q_\theta || p \right) = \mathbb{E}_{\phi \sim q_\theta} \left[ \log q_\theta(\phi) + S[\phi] \right] + \log Z.
\end{equation}
Since $\log Z$ is independent of the model parameters $\theta$, this constant term can be ignored during training. The training loss is then defined as:
\begin{equation}
    \mathcal{L}(\theta) = \mathbb{E}_{\phi \sim q_\theta} \left[ \log q_\theta(\phi) + S[\phi] \right] = \int d\phi\, q_\theta(\phi) \left[ \log q_\theta(\phi) + S[\phi] \right].
\end{equation}
Using $\phi=f_\theta(\mathbf{z})$ with $\mathbf{z}\sim r(\mathbf{z})$, this can be further rewritten as:
\begin{equation}
    \mathcal{L}(\theta) = \mathbb{E}_{\mathbf{z} \sim r(\mathbf{z})} \left[ S(f_\theta(\mathbf{z})) + \log q_\theta(f_\theta(\mathbf{z})) \right].
\end{equation}
According to the change of variables formula, 
\begin{equation}
    \log q_\theta(f_\theta(\mathbf{z})) = \log r(\mathbf{z}) - \log \left| \det \frac{\partial f_\theta(\mathbf{z})} {\partial \mathbf{z}} \right|.
\end{equation}
Therefore, the final training objective is:
\begin{align}
    \mathcal{L}(\theta) &= \mathbb{E}_{\mathbf{z} \sim r(\mathbf{z})} \Big[ S(f_\theta(\mathbf{z})) + \log r(\mathbf{z}) \nonumber \\
    &\quad - \log \left| \det \frac{\partial f_\theta(\mathbf{z})} {\partial \mathbf{z}} \right| \Big].
\end{align}

In practice, we can approximate this expectation value over the prior distribution numerically by discretizing the integral using Monte Carlo (M-C) sampling. For a given batch of $B$ independent samples $\{ \mathbf{z}^{(i)} \}_{i=1}^B$ drawn from the prior distribution $r(\mathbf{z})$, the discretized empirical loss function becomes:
\begin{equation}
    \mathcal{L}(\theta) \approx \frac{1}{B} \sum_{i=1}^B \left[ S(f_\theta(\mathbf{z}^{(i)})) + \log r(\mathbf{z}^{(i)}) - \log \left| \det \frac{\partial f_\theta(\mathbf{z}^{(i)})}{\partial \mathbf{z}^{(i)}} \right| \right].
\end{equation}
Then we minimize this loss, and many optimization methods are used to achieve this, such as Stochastic Gradient Descent (SGD) and its adaptive variants like Adam or RMSprop. In subsequent training and optimization, we useed the Adam method.






\subsection{Metropolis-Hastings with Normalizing Flow}
If the flow-generated distribution $q_\theta(\phi)$ perfectly matches the target distribution $p(\phi)$, the samples generated by the flow can be directly used to compute physical observables. However, in practical training, $q_\theta(\phi)$ is usually only an approximation of $p(\phi)$. To ensure exact sampling, the flow model can be employed as an independent proposal distribution in conjunction with a Metropolis-Hastings correction.

Suppose the current configuration is $\phi$, and a candidate configuration $\phi'$ is independently generated from the flow model such that $\phi'\sim q_\theta(\phi')$. Therefore, our proposal distribution is $g(\phi' | \phi) = q_\theta(\phi')$, and the acceptance probability is:
\begin{equation}
    A(\phi \to \phi') = \min \left[ 1, \frac{ p(\phi')q_\theta(\phi) }{ p(\phi)q_\theta(\phi') } \right].
\end{equation}
Since $p(\phi) \propto e^{-S[\phi]}$, the acceptance rate can be written as:
\begin{equation}
    A(\phi \to \phi') = \min \left[ 1, e^{-S[\phi'] + S[\phi] + \log q_\theta(\phi) - \log q_\theta(\phi')} \right].
\end{equation}
This approach guarantees the correctness of the target distribution while leveraging the flow model to generate global proposals, thereby effectively circumventing critical slowing down and reducing sample autocorrelation.








\section{Model Architecture}
In Section 4, we introduced neural networks and the normalizing flow algorithm. To implement the normalizing flow algorithm, we need to rely on neural networks. Next, we will detail the specific network architecture used in this paper.







\subsection{Overall Neural Network Architecture}

As shown in Figure~\ref{fig:network_structure}, during formal training, our neural network is configured with 12 coupling layers. Each coupling layer contains 3 residual blocks and two branch networks that each contain 1 residual block. The outputs of these 3 residual blocks are treated as inputs entering the branch networks, which then output $s^i$ and $t^i_{\theta}$ respectively, where $i$ represents the $i$-th coupling layer. Each residual block contains two convolutional layers connected via a residual connection. (A residual connection simply adds the input of a block directly to its output, allowing information to bypass layers.) This mechanism effectively prevents the vanishing gradient problem and makes it much easier to train deep networks.

In addition, during initialization, we set all parameters of the last convolutional layer in each coupling layer to 0. In this way, the $s^i$ and $t^i_{\theta}$ outputs are correspondingly 0, which guarantees that our network is close to an identity mapping at the initial stage of training under the affine transformation. Other hidden layers are initialized using a Gaussian distribution with a mean of 0 and a variance of 0.01.

Next, we will introduce convolutional neural networks (CNNs) and explain why we choose to use them over fully connected neural networks.

\clearpage
\begin{sidewaysfigure}
    \centering
    \includegraphics[width=\textheight]{network_structure.png}
    \caption{Overall architecture of the convolutional neural network for the normalizing flow.}
    \label{fig:network_structure}
\end{sidewaysfigure}
\clearpage










\subsection{Convolutional Neural Networks and Translational Equivariance}

In Section 4, we introduced Fully Connected Networks (FCNs). Applying FCNs directly to lattice field theory poses a practical challenge: the number of parameters explodes proportionally to the square of the lattice volume $V=L^2$. For an input with $V$ degrees of freedom, a dense weight matrix typically scales as $O(V^2)$. This brings huge memory and computational overhead, making it extremely hard to generalize the model to larger lattice sizes.

More importantly, physical actions in lattice field theory have spatial translational symmetry. In other words, the physical laws governing the system do not depend on the absolute position of a lattice site, but only on the relative relationships between neighboring sites. If we use an ordinary FCN, the network is fundamentally unaware of this structure and has to relearn this translational invariance from scratch during training, which is incredibly inefficient.

To fix this, we replace the context networks in the flow model with Convolutional Neural Networks (CNNs). The core computational components of a CNN are convolution kernels and channels. Suppose we have a 2D input field $\mathbf{h}^{(l-1)}$ with a shape of $C_{\mathrm{in}} \times L \times L$, where $C_{\mathrm{in}}$ is the number of input channels and $L$ is the lattice dimension. The convolution operation slides a small weight matrix (like a $3 \times 3$ kernel) across the input data, calculating a weighted sum within each local neighborhood.

For the $c_{\mathrm{out}}$-th output channel of the $l$-th layer, the output at position $(x,y)$ is given by:
\begin{equation}
h^{(l)}_{c_{\mathrm{out}}}(x,y) = \sigma \left( \sum_{c_{\mathrm{in}}=1}^{C_{\mathrm{in}}} \sum_{i=-k}^{k} \sum_{j=-k}^{k} W^{(l)}_{c_{\mathrm{out}},c_{\mathrm{in}}}(i,j) h^{(l-1)}_{c_{\mathrm{in}}}(x+i,y+j) + b^{(l)}_{c_{\mathrm{out}}} \right),
\end{equation}
where:
\begin{itemize}
    \item $W^{(l)}$ is the convolution kernel of the $l$-th layer;
    \item The kernel size is $(2k+1)\times(2k+1)$;
    \item $b^{(l)}_{c_{\mathrm{out}}}$ is the bias term;
    \item $\sigma$ is a non-linear activation function;
    \item $(x+i,y+j)$ represents the local neighborhood centered at $(x,y)$.
\end{itemize}

\begin{figure}[htbp]
    \centering
    % 如果需要调整图片大小，可以修改 width 的比例，例如 0.6\textwidth 或 8cm
    \includegraphics[width=0.5\textwidth]{convolution.png}
    \caption{Diagram of the convolution operation (3x3 convolution kernel)}
    \label{fig:convolution}
\end{figure}

As shown in this formula, all positions across the entire lattice share the same set of convolution weights $W$. This weight-sharing mechanism massively compresses the parameter space. More precisely, the number of trainable parameters in a CNN is mainly determined by the kernel size, channel count, and network depth, and does not scale directly with the lattice volume $V=L^2$. Naturally, the computational cost for forward and backward passes still grows with the number of sites $V$, but compared to the $O(V^2)$ scaling of FCNs, CNNs are fundamentally a better fit for lattice systems.

For physical systems with periodic boundary conditions, we must handle boundary effects correctly in the convolution calculations. Lattice field theory usually uses periodic boundary conditions, defined as:
\begin{equation}
\phi(x+L\hat{\mu})=\phi(x).
\end{equation}
Therefore, we use circular padding in the CNN. When the convolution kernel slides to the edge of the lattice, it automatically wraps around and reads values from the opposite side, perfectly respecting the physical topology of the periodic lattice.

With this setup, the CNN naturally shows a sufficient condition for translational symmetry, which is known as translational equivariance. Let $T_{\hat{\mu}}$ be the operator that translates the field configuration by $\hat{\mu}$ lattice sites. An ideal convolutional network satisfies:
\begin{equation}
f_\theta(T_{\hat{\mu}}\phi)=T_{\hat{\mu}} f_\theta(\phi).
\end{equation}
This means shifting the input field by a certain distance and passing it through the network gives the exact same result as passing the original field through the network and then shifting the output by the same distance. The network no longer has the burden of learning translational symmetry from the data because this symmetry is structurally guaranteed by the convolutional architecture and circular padding.

However, as we introduced in Section 4, our Normalizing Flow model uses a checkerboard masking strategy during updates. The checkerboard mask splits the lattice sites into two different sub-lattices, usually called ``red'' and ``black'' sites. Since a single coupling layer only updates half the sites while freezing the other half, the fixed checkerboard mask explicitly separates these two sub-lattices. A single-site translation ($1\hat{\mu}$) swaps the black and red sub-lattices. Because the order in which red and black sites enter the network is inconsistent, the information relied upon for subsequent site updates is also inconsistent. Therefore, even if we keep the network parameters for the red and black grids identical, the strict $1\hat{\mu}$ translational equivariance is still broken.

As shown in Figure~\ref{fig:1a_broken}, $r$ represents the input at the red mask positions, and $b$ represents the input at the black mask positions. The terms $r_u$ and $b_u$ denote the updates at the corresponding positions after passing through the affine coupling layer. It can be observed that, depending on the order of entering the coupling layer, the same values before and after translation result in different updated values.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.6\textwidth]{1a_broken.png}
    \caption{Breaking of 1$\hat{\mu}$ equivariance}
    \label{fig:1a_broken}
\end{figure}

Accutually, we can prove that our convolutional neural network possesses $2\hat{\mu}$
translational equivariance.(A detailed proof is provided in Appendix~\ref{app:proof translational equivariance}.)






\subsection{Convolutional Details in the Coupling Layer}

As mentioned in the previous subsection, we chose convolutional neural networks over fully connected ones to optimize the number of parameters and computational load, and to respect translation invariance. In this subsection, we will describe the specific details of these convolutional networks.

As shown in Figure~\ref{fig:coupling_layer}, each coupling layer is actually made of multiple convolutional layers connected together. The top part of the figure shows the first two convolutional layers at the beginning of each coupling layer, while the bottom part shows the last two convolutional layers at the end of the coupling layer. Every two convolutional layers are connected via a residual connection, forming what we call a residual block, as shown in the previous figure (Figure~\ref{fig:network_structure}).
For lattice sizes of $L=12$ and $14$, the output channels of our convolutional layers are set to 32, corresponding to $32\times32$ kernels of size $3\times3$. For $L=6, 8$, and $10$, the output channels are 16, corresponding to $16\times16$ kernels of size $3\times3$.


This concludes the specific architecture of our convolutional neural network. In the next section, we will introduce our optimization of the input distribution to improve training efficiency, our attempts at introducing $Z_2$ symmetry, and a comparison of the computational cost between the normalizing flow algorithm and traditional algorithms.

\clearpage
\begin{sidewaysfigure}
    \centering
    \includegraphics[width=\textheight]{coupling_layer.png}
    \caption{Detailed architecture of a single coupling layer.}
    \label{fig:coupling_layer}
\end{sidewaysfigure}
\clearpage








\section{Prior Sample, $Z_2$ Symmetry, and Computational Cost}

In physics, symmetry always plays a crucial role. Our $2D$ $\phi^4$ lattice scalar theory naturally possesses translational symmetry and $Z_2$ symmetry. Translational symmetry corresponds to the invariance of the system under the shift $\phi'(n) = \phi(n + \hat{\mu})$, where $\hat{\mu}$ is the unit vector in the $\mu$-th direction. Under this transformation, the partition function remains unchanged:
$$ Z = \int D\phi \, e^{-S(\phi)} = \int D\phi' \, e^{-S(\phi')} = Z' $$
Similarly, the $Z_2$ symmetry means the system is invariant under the field flip $\phi'(n) = -\phi(n)$. 

When using normalizing flows to approximate the target Boltzmann distribution, it is highly preferable, or even required, that our approximated distribution $q_\theta(\phi)$ explicitly maintains these physical symmetries. This means we want $q_\theta(\phi') = q_\theta(\phi)$. To realize this, both the neural network transformation $f_\theta(z)$ and the prior distribution $r(z)$ must be constrained to preserve these symmetries. Specifically, the transformation should satisfy $f_\theta(Tz) = T f_\theta(z)$, and the prior should satisfy $r(Tz) = r(z)$, where $T$ represents the corresponding symmetry operation.

A normalizing flow algorithm mainly consists of three parts: an initial sample $z$ from a simple prior distribution $r(z)$, an invertible mapping function $f_\theta(z)$ (which is our neural network), and a loss function associated with the target distribution. Therefore, the optimization of the algorithm should naturally focus on these three targets. For the prior sampling, we can design the prior distribution to contain as much information about the target distribution as possible. For the neural network, we can explicitly encode the symmetries of the target distribution, such as translational symmetry and $Z_2$ symmetry, into the network architecture. If a symmetry is inherently difficult for the network to learn, encoding it explicitly maybe helpful. Finally, for the loss function, we can also try adding symmetry constraints as an auxiliary loss term to improve learning efficiency.

As discussed in the previous section, we have already respected the translational symmetry by choosing Convolutional Neural Networks (CNNs). Therefore, in this section, we will focus on the remaining aspects. First, we will test the feasibility of optimizing the initial prior distribution. Next, we will introduce and test the $Z_2$ equivalence in our model. Finally, we will compare the computational cost between our normalizing flow algorithm and traditional algorithms, with a specific focus on the autocorrelation time.







\subsection{Optimizing the Input Distribution: The Free-Field Prior}

As outlined at the beginning of this section, optimizing the initial prior distribution is our first step to improve the algorithm's learning efficiency. In standard normalizing flows, the initial latent variable usually starts from uncorrelated Gaussian white noise. However, because of the specific structure of our coupling layers, forcing the network to learn physical spatial correlations from scratch is very difficult. In this subsection, we will explain the connectivity bottleneck caused by this network architecture, introduce a "free-field prior" to solve it, and analytically find the optimal initial distribution.

\subsubsection{The Checkerboard Masking Bottleneck}

In Section 4, we introduced that in the affine coupling layers, we usually use a checkerboard mask to split the lattice sites into two disjoint sets: $\phi = \phi_a + \phi_b$, where $\phi_a$ is the frozen part, and $\phi_b$ is the updated part. A typical affine coupling transformation looks like this:
\begin{equation}
\phi_b' = \left(\phi_b - t_\theta(\phi_a)\right) \odot \exp[-s_\theta(\phi_a)],
\end{equation}
while the frozen part stays the same: $\phi_a'=\phi_a$.

However, this checkerboard coupling inevitably creates a connectivity bottleneck. When the network updates a specific site in $\phi_b$, the site's own value is not input into the network for training, but is only multiplied element-wise with the network's output value during the final affine transformation. 

Take the kinetic term of a scalar field as an example. On a discrete lattice, it is equivalent to:
\begin{equation}
\sum_y \Box(x,y)\phi(y) = \sum_\mu (2\phi(x) - \phi(x-\hat{\mu}) - \phi(x+\hat{\mu})).
\end{equation}
For the network to capture the complete kinetic term, it must know all the information of the site itself and its four surrounding sites. The mask prevents it from knowing the site's own original value, so it has to rely entirely on the surrounding frozen sites in $\phi_a$ to make a prediction. Even though this alternating update scheme keeps invertibility and a tractable Jacobian, it fundamentally causes a delay in information propagation. For field configurations with strong long-range correlations, the network has to stack a huge number of coupling layers to iteratively propagate this information.

To demonstrate this, we compared the training acceptance rates on a small model with 12 coupling layers and 16 hidden layer output channels, on a $14 \times 14$ lattice. The target distributions are respectively the pure kinetic action with an effective mass $S_{\mathrm{kin}+m_{\mathrm{eff}}}(\phi)$ and the pure potential action $S_{\mathrm{pot}}(\phi)$. $m_{\mathrm{eff}}=0.3$ is added to avoid divergence.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{kinetic_acc.png}
    \caption{Comparison of MCMC acceptance rates over training iterations. Exp A (blue line) represents the model trained on the pure kinetic action with an effective mass, while Exp B (red line) represents the model trained on the pure potential action. The noticeably lower acceptance rate for Exp A highlights the difficulty of capturing spatial correlations under the checkerboard masking bottleneck.}
    \label{fig:kinetic_acc}
\end{figure}

As Figure~\ref{fig:kinetic_acc} shows, the kinetic term in the $\phi^4$ action is indeed much harder to learn compared to the mass term.

\subsubsection{Momentum-Space Pre-coding}

To ease this problem, we introduce a free-field prior (momentum-space pre-coding). The core idea is simple: instead of starting from uncorrelated Gaussian white noise, we build an initial field in momentum space upfront, giving it the correct two-point correlation function structure governed by the free-field action. 

By doing this, the latent variable $z$ fed into the flow model inherently contains free-field spatial correlations. The neural network then does not need to learn the finite difference structure of the kinetic term from scratch, and can focus all its expressive power on learning the non-Gaussian deformations caused by the non-linear interaction term $\lambda\phi^4$.

For a 2D periodic lattice, the free-field distribution in momentum space can be exactly decoupled. The target discrete distribution we want to sample from is:
\begin{equation}
r(\phi)=\frac{1}{Z_{\mathrm{free}}}\exp\left[-\sum_k \left(\tilde{K}(k)+m_{\mathrm{free}}^2\right)|\tilde{\phi}(k)|^2\right],
\end{equation}
where $\tilde{K}(k)$ are the eigenvalues of the discrete Laplacian:
\begin{equation}
\tilde{K}(k)=4\sin^2\frac{k_1}{2}+4\sin^2\frac{k_2}{2}, \qquad k_\mu=\frac{2\pi n_\mu}{L}.
\end{equation}
Here, $m_{\mathrm{free}}^2$ is the mass parameter of the free field, which controls the variance of the Gaussian modes. To generate these samples, we simply draw independent Gaussian variables for the real and imaginary parts of each momentum mode according to this variance, and then use the Fast Fourier Transform (FFT) to map them back to real space. The detailed mathematical derivation of this momentum-space decoupling is provided in Appendix~\ref{app:free_field_prior}.

\subsubsection{Finding the Optimal Parameters}

Free-field pre-sampling is not just an empirical engineering trick; it can also be explained from an optimization perspective. Early in training, because the flow network uses small parameter initialization, the model transformation is roughly an identity mapping ($f_\theta(z)\approx z$). Therefore, the model's initially generated distribution $q_0(\phi)$ is approximately equal to the prior distribution we selected ($r(\phi)$). If we pick a good prior, training becomes much easier.

We want our initial distribution to meet these three principles:
\begin{enumerate}
    \item The initial distribution should match the hardest-to-learn quadratic kinetic structure of the target distribution as closely as possible.
    \item The initial distribution should maintain $Z_2$ symmetry, preventing initial asymmetry from disturbing the training.
    \item Among the candidate priors that satisfy the first two conditions, the Kullback-Leibler (KL) divergence between the initial distribution and the target distribution should be minimized.
\end{enumerate}

Based on the first principle, to match the kinetic structure, we construct the initial prior as a general Gaussian distribution in momentum space:
\begin{equation}
r(\phi; m_{\mathrm{free}}^2, c) = \frac{1}{Z_{\mathrm{free}}} \exp\left[-\sum_k \left(\tilde{K}(k) + m_{\mathrm{free}}^2\right) |\tilde{\phi}(k) - c|^2\right].
\end{equation}
Because the normalization constant $Z_{\mathrm{free}}$ is simply the result of a Gaussian integral completely determined by the variance parameter $m_{\mathrm{free}}^2$, this distribution intrinsically has exactly two free parameters: the mass parameter $m_{\mathrm{free}}^2$ and the mean shift $c$.

Then, the second and third principles respectively determine the optimal values for these two parameters. According to the second principle, to preserve the $Z_2$ symmetry ($\phi \to -\phi$), the optimal mean shift $c$ must be exactly 0. According to the third principle, the optimal mass parameter $m_{\mathrm{free}}^2$ is the one that minimizes the initial KL divergence $D_{\mathrm{KL}}(q_{m_{\mathrm{free}}^2}||p)$. 

By calculating the expected value of the target action and the entropy of the prior, we can analytically find this minimum. The complete derivation is provided in Appendix~\ref{app:kl_divergence}. By setting the derivative of the KL divergence to zero, we obtain the following self-consistent equation for the optimal mass parameter:
\begin{equation}
m_{\mathrm{free},\star}^2=m^2+6\lambda G(0),
\end{equation}
where $G(0)$ is the zero-distance two-point function defined as:
\begin{equation}
G(0)=\frac{1}{V}\sum_k\frac{1}{2[\tilde{K}(k)+m_{\mathrm{free},\star}^2]}.
\end{equation}

For our specific testing parameters ($L=14$, $m^2=-4$, $\lambda=5.113$), solving this equation numerically gives us the optimal free-field prior mass parameter:
\begin{equation}
m_{\mathrm{free},\star}^2\approx 0.6005.
\end{equation}

\subsubsection{Performance Comparison}

To verify our optimization strategy, we compared the loss descent history and Metropolis-Hastings (MH) acceptance rates under different initial distributions.

We compared the standard Gaussian distribution against our free-field distribution. For the standard Gaussian prior, we evaluated a smaller model (32 channels, 12 layers) and a larger model (64 channels, 18 layers). For the free-field prior, we uniformly applied the smaller architecture while varying the mass parameter $m_{\mathrm{free}}^2$ and a mean shift $c$.

As shown in Figure~\ref{fig:30k_comparison}, the free-field distribution with our derived optimal parameter ($m_{\mathrm{free},\star}^2 \approx 0.6005, c=0$) exhibits outstanding performance. It not only achieves the lowest initial loss but also converges to the highest acceptance rate. Furthermore, intentionally breaking the $Z_2$ symmetry by setting a non-zero mean shift ($c=1.0$) severely negatively impacts the acceptance rate and introduces significant interference early in training. This confirms the validity of our three initialization principles.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{paper_figure_30k_comparison_final.png}
    \caption{Comparison of training loss history and MCMC acceptance rates under different initial distributions. The standard Gaussian prior is tested with two different network sizes, while the free-field prior is evaluated under varying mass parameters $m_{\mathrm{free}}^2$ and mean shifts $c$. All free-field experiments utilize the 32-channel, 12-layer architecture.}
    \label{fig:30k_comparison}
\end{figure}






\subsection{Free-Field Prior}
In Section 4, we introduced that in the affine coupling layers of Normalizing Flows, we usually use a checkerboard mask to split the lattice sites into two disjoint sets:
\begin{equation}
\phi = \phi_a + \phi_b,
\end{equation}
where $\phi_a$ is the frozen part, and $\phi_b$ is the updated part. 

Under a checkerboard mask, half of the sites are frozen, and the other half are updated. A typical affine coupling transformation looks like this:
\begin{equation}
\phi_b' = \left(\phi_b - t_\theta(\phi_a)\right) \odot \exp[-s_\theta(\phi_a)],
\end{equation}
while the frozen part stays the same:
\begin{equation}
\phi_a'=\phi_a.
\end{equation}

However, this checkerboard coupling inevitably creates a connectivity bottleneck. When the network updates a specific site in $\phi_b$, the site's own value is not input into the network for training, but is only multiplied element-wise with the network's output value during the final affine transformation. This brings serious problems for the actions in lattice field theory. 

Take the kinetic term of a scalar field as an example. The continuous kinetic term is:
\begin{equation}
(\partial_\mu \phi)^2.
\end{equation}
On a discrete lattice, this is equivalent to:
\begin{equation}
\sum_y \Box(x,y)\phi(y) = \sum_\mu (2\phi(x) - \phi(x-\hat{\mu}) - \phi(x+\hat{\mu})).
\end{equation}
For the network to capture the complete kinetic term, it must know all the information of the site itself and its four surrounding sites. The mask prevents it from knowing the site's own original value, so it has to rely entirely on the surrounding frozen sites in $\phi_a$ to make a prediction.

More specifically, when updating the red sub-lattice, the network can only inject information from the black sub-lattice into the red sites. In the subsequent layers, when updating the black sub-lattice, the network can finally use the red sites (which were modified in the previous step) to update the black sites. Even though this alternating update scheme keeps invertibility and a tractable Jacobian, it fundamentally causes a delay in information propagation. For field configurations with strong long-range correlations, the network has to stack a huge number of coupling layers to iteratively propagate this information.

To demonstrate this, we compared the training acceptance rates on a small model with 12 coupling layers and 16 hidden layer output channels, on a $14 \times 14$ lattice. The initial distribution is a standard Gaussian distribution, and the target distributions are respectively $S_{\mathrm{kin}+m_{\mathrm{eff}}}(\phi) = \sum_x \left[ \sum_y \phi(x)\Box(x,y)\phi(y) + m_{\mathrm{eff}}^2 \phi(x)^2 \right]$ and $S_{\mathrm{pot}}(\phi) = \sum_x \left[ m^2\phi(x)^2 + \lambda \phi(x)^4 \right]$. To avoid the divergence of the kinetic term, we added an effective mass, where $m_{\mathrm{eff}}=0.3$ is around the effective mass of the $\phi^4$ field on the $14 \times 14$ lattice. $m$ and $\lambda$ are the bare parameters of the $\phi^4$ field.


\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{kinetic_acc.png}
    \caption{Comparison of MCMC acceptance rates over training iterations. Exp A (blue line) represents the model trained on the pure kinetic action with an effective mass, while Exp B (red line) represents the model trained on the pure potential action. The noticeably lower acceptance rate for Exp A highlights the difficulty of capturing spatial correlations under the checkerboard masking bottleneck.}
    \label{fig:kinetic_acc}
\end{figure}

It can be seen that the kinetic term in the $\phi^4$ action is indeed much harder to learn compared to the mass term.

To ease this problem, we introduce a free-field prior (momentum-space pre-coding). The core idea is simple: instead of starting from uncorrelated Gaussian white noise, we build an initial field in momentum space upfront, giving it the correct two-point correlation function structure governed by the free-field action. In other words, we want the latent variable $z$ fed into the flow model to inherently contain free-field spatial correlations. Because of this, the neural network, built upon the Affine coupling layers, does not need to learn the finite difference structure of the kinetic term from scratch. Instead, it can focus all its expressive power on learning the non-Gaussian deformations caused by the non-linear interaction term $\lambda\phi^4$.

The free scalar field action in continuous space is given by:
\begin{equation}
S_{\mathrm{free}}=\int d^2x\,\phi(x)\left(-\partial^2 + m_{\mathrm{free}}^2\right)\phi(x).
\end{equation}
On a periodic lattice, we use the discrete Fourier transform to map the real space field $\phi(x)$ to momentum space:
\begin{equation}
\phi(x)=\frac{1}{\sqrt{V}}\sum_k\tilde{\phi}(k)e^{ik\cdot x},
\end{equation}
where $V=L^2$ is the lattice volume, and the allowed momenta are:
\begin{equation}
k_\mu=\frac{2\pi n_\mu}{L},\qquad n_\mu=0,1,\dots,L-1.
\end{equation}
Since $\phi(x)$ is a real scalar field, meaning $\phi^*(x)=\phi(x)$, the Fourier modes in momentum space have to satisfy the Hermitian condition:
\begin{equation}
\tilde{\phi}^*(-k)=\tilde{\phi}(k).
\end{equation}
This means $\tilde{\phi}(k)$ and $\tilde{\phi}(-k)$ are not independent degrees of freedom, but are complex conjugates of each other.

On a discrete lattice, the Laplacian operator is diagonalized in momentum space. The eigenvalues of the 2D periodic lattice Laplacian are:
\begin{equation}
\tilde{K}(k)=\sum_{\mu=1}^{2}\left(2-2\cos k_\mu\right)=4\sin^2\frac{k_1}{2}+4\sin^2\frac{k_2}{2}.
\end{equation}
Therefore, the quadratic form of the free field can be represented in momentum space as:
\begin{equation}
S_{\mathrm{free}}=\sum_k\left[\tilde{K}(k)+m_{\mathrm{free}}^2\right]|\tilde{\phi}(k)|^2.
\end{equation}
To avoid divergence, $m_{\mathrm{free}}^2$ must take a positive value. Here, we simply set $m_{\mathrm{free}}^2 = |m^2|$. The specific value of $m_{\mathrm{free}}^2$ will be further discussed in the next subsection.

Define $\tilde{M}(k)=\tilde{K}(k)+m_{\mathrm{free}}^2$, the free-field distribution becomes:
\begin{equation}
r(\phi)=\frac{1}{Z_{\mathrm{free}}}\exp\left[-\sum_k\tilde{M}(k)|\tilde{\phi}(k)|^2\right].
\end{equation}

From this expression, it is obvious that different momentum modes of the free field are completely decoupled in momentum space. The free field, which shows long-range correlations in real space, breaks down into a set of independent Gaussian modes in momentum space. So, sampling the free field can be done by independently sampling a Gaussian variable for each momentum mode. If we temporarily ignore the conjugate constraints imposed by the real number condition, we can formally write the complex Fourier modes as:
\begin{equation}
\tilde{\phi}(k)=\phi^R(k)+i\phi^I(k).
\end{equation}
Then,
\begin{equation}
|\tilde{\phi}(k)|^2=\left(\phi^R(k)\right)^2+\left(\phi^I(k)\right)^2.
\end{equation}
Thus, both the real and imaginary parts of each mode correspond to independent Gaussian variables, with their variance controlled by $\tilde{M}(k)$. Given standard normal variables $\gamma \sim \mathcal{N}(0,1)$, we can formally generate the corresponding free-field modes like this:
\begin{equation}
\phi^R(k)=\frac{\gamma^R(k)}{\sqrt{2\tilde{M}(k)}},\qquad\phi^I(k)=\frac{\gamma^I(k)}{\sqrt{2\tilde{M}(k)}}.
\end{equation}
Using the Fast Fourier Transform (FFT), we can efficiently generate free-field samples.







\subsubsection{How to Find a "Good" Initial Distribution}

Free-field pre-sampling isn't just an empirical engineering trick; it can also be explained from an optimization perspective. Our goal isn't just to pick a random prior distribution, but to find an initial distribution that is as close to the target distribution as possible, while still satisfying physical structure and training startup stability.

Early in training, because the flow network uses small parameter initialization, the model transformation is roughly an identity mapping:
\begin{equation}
f_\theta(z)\approx z.
\end{equation}
So the initially generated field satisfies:
\begin{equation}
\phi_{\mathrm{init}}=f_\theta(z)\approx z.
\end{equation}
In other words, the model's generated distribution $q_0(\phi)$ early in training is approximately equal to the prior distribution we selected:
\begin{equation}
q_0(\phi_{\mathrm{init}})\approx r(\phi_{\mathrm{init}}).
\end{equation}
Therefore, how we choose the prior distribution actually decides where the model starts at the beginning of training. If we pick a good prior, the flow network only needs to learn the remaining non-Gaussian interaction corrections; if we pick a bad prior, the network has to learn kinetic correlations and long-range structures from zero, making training much harder.

\paragraph{Three Initialization Principles}

We want our initial distribution to meet these three principles:
\begin{enumerate}
    \item The initial distribution should match the hardest-to-learn quadratic kinetic structure of the target distribution as closely as possible;
    \item The initial distribution should maintain $Z_2$ symmetry, so that the network maintains $Z_2$ symmetry at initialization, and subsequent training will not be disturbed by initial asymmetry;
    \item Among the candidate priors that satisfy the first two conditions, the KL divergence between the initial distribution and the target distribution should be as small as possible.
\end{enumerate}

The first principle requires the prior distribution to contain the structure of the lattice kinetic term. For a 2D periodic lattice, the eigenvalues of the discrete Laplacian in momentum space are:
\begin{equation}
\tilde{K}(k)=\sum_{\mu=1}^{2}(2-2\cos k_\mu)=4\sin^2\frac{k_1}{2}+4\sin^2\frac{k_2}{2}.
\end{equation}
Where the allowed momenta are:
\begin{equation}
k_\mu=\frac{2\pi n_\mu}{L},\qquad n_\mu=0,1,\dots,L-1.
\end{equation}
So, $\tilde{K}(k)$ isn't some extra bare parameter; it is a quantity completely determined by the lattice size $L$ and the periodic boundary conditions.

Therefore, we choose the initial distribution to be a Gaussian distribution in momentum space with two free parameters (mass parameter $m_{\mathrm{free}}^2$ and mean shift $c$):
\begin{equation}
\exp\left[-(\tilde{K}(k)+m_{\mathrm{free}}^2)|\tilde{\phi}(k)-c|^2\right].
\end{equation}
The second principle makes us choose $c = 0$.

The third principle is about picking the value among these allowed $m_{\mathrm{free}}^2$ that minimizes the KL divergence:
\begin{equation}
m_{\mathrm{free},\star}^2=\arg\min_{m_{\mathrm{free}}^2\in\mathcal{A}}D_{\mathrm{KL}}(q_{m_{\mathrm{free}}^2}||p).
\end{equation}

To find the $m_{\mathrm{free}}^2$ that minimizes the KL divergence under the momentum space Gaussian distribution, we perform the following derivation:

The target distribution is the Boltzmann distribution:
\begin{equation}
p(\phi)=\frac{1}{Z}e^{-S_{\mathrm{target}}[\phi]}.
\end{equation}
According to the formula (placeholder for previous formula reference), the target action is:
\begin{equation}
S_{\mathrm{target}}[\phi]=\sum_x\left[\phi(x)(-\Box\phi)(x)+m^2\phi(x)^2+\lambda\phi(x)^4\right].
\end{equation}
Its quadratic part in momentum space is:
\begin{equation}
S_{\mathrm{quad}}[\phi]=\sum_k[\tilde{K}(k)+m^2]|\tilde{\phi}(k)|^2.
\end{equation}
The KL divergence is defined as:
\begin{equation}
D_{\mathrm{KL}}(q_{m_{\mathrm{free}}^2}||p)=\int \mathcal{D}\phi\,q_{m_{\mathrm{free}}^2}(\phi)\log\frac{q_{m_{\mathrm{free}}^2}(\phi)}{p(\phi)}.
\end{equation}
Plugging in $p(\phi)=\frac{1}{Z}e^{-S_{\mathrm{target}}[\phi]}$, we get:
\begin{equation}
D_{\mathrm{KL}}(q_{m_{\mathrm{free}}^2}||p)=\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{target}}(\phi)]-H(q_{m_{\mathrm{free}}^2})+\log Z,
\end{equation}
where
\begin{equation}
H(q_{m_{\mathrm{free}}^2})=-\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[\log q_{m_{\mathrm{free}}^2}(\phi)]
\end{equation}
is the entropy of the free-field prior.

\paragraph{Expected Value of the Target Action}
The target action splits into a quadratic term and a quartic interaction term:
\begin{equation}
S_{\mathrm{target}}=S_{\mathrm{quad}}+S_{\mathrm{int}},
\end{equation}
where
\begin{equation}
S_{\mathrm{int}}=\lambda\sum_x\phi(x)^4.
\end{equation}
Under the identity mapping initialization, for the quadratic term, we have:
\begin{equation}
\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{quad}}]=\sum_k[\tilde{K}(k)+m^2]\left\langle|\tilde{\phi}(k)|^2\right\rangle_{q_{m_{\mathrm{free}}^2}}.
\end{equation}
Substituting $\left\langle|\tilde{\phi}(k)|^2\right\rangle_{q_{m_{\mathrm{free}}^2}}=\frac{1}{2[\tilde{K}(k)+m_{\mathrm{free}}^2]}$, we get:
\begin{equation}
\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{quad}}]=\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{\tilde{K}(k)+m_{\mathrm{free}}^2}.
\end{equation}
For the quartic term, since $q_{m_{\mathrm{free}}^2}$ is a Gaussian distribution, we can use Wick's theorem:
\begin{equation}
\left\langle\phi(x)^4\right\rangle_{q_{m_{\mathrm{free}}^2}}=3\left\langle\phi(x)^2\right\rangle_{q_{m_{\mathrm{free}}^2}}^2=3G(0)^2.
\end{equation}
So
\begin{equation}
\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{int}}]=\lambda\sum_x\left\langle\phi(x)^4\right\rangle_{q_{m_{\mathrm{free}}^2}}=3\lambda V G(0)^2.
\end{equation}
Putting it together:
\begin{equation}
\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{target}}]=\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{\tilde{K}(k)+m_{\mathrm{free}}^2}+3\lambda V G(0)^2.
\end{equation}

\paragraph{Entropy Term of the Free-Field Prior}
The free-field prior is a Gaussian distribution, and its covariance in momentum space is proportional to:
\begin{equation}
C_{m_{\mathrm{free}}^2}(k)\propto\frac{1}{\tilde{K}(k)+m_{\mathrm{free}}^2}.
\end{equation}
The entropy of a Gaussian distribution satisfies:
\begin{equation}
H(q_{m_{\mathrm{free}}^2})=\frac{1}{2}\log\det C_{m_{\mathrm{free}}^2}+\text{const}.
\end{equation}
Therefore
\begin{equation}
H(q_{m_{\mathrm{free}}^2})=-\frac{1}{2}\sum_k\log[\tilde{K}(k)+m_{\mathrm{free}}^2]+\text{const}.
\end{equation}
Thus, ignoring constants independent of $m_{\mathrm{free}}^2$, the KL divergence can be written as:
\begin{equation}
D_{\mathrm{KL}}(m_{\mathrm{free}}^2)=\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{\tilde{K}(k)+m_{\mathrm{free}}^2}+3\lambda V G(0)^2+\frac{1}{2}\sum_k\log[\tilde{K}(k)+m_{\mathrm{free}}^2].
\end{equation}
Where
\begin{equation}
G(0)=\frac{1}{V}\sum_k\frac{1}{2[\tilde{K}(k)+m_{\mathrm{free}}^2]}.
\end{equation}

To find the free-field mass parameter that minimizes the KL divergence, we take the derivative with respect to $m_{\mathrm{free}}^2$. Define
\begin{equation}
M(k)=\tilde{K}(k)+m_{\mathrm{free}}^2.
\end{equation}
Then
\begin{equation}
G(0)=\frac{1}{V}\sum_k\frac{1}{2M(k)}.
\end{equation}
So
\begin{equation}
\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}=-\frac{1}{2V}\sum_k\frac{1}{M(k)^2}.
\end{equation}
Taking the derivative of the KL divergence:
\begin{equation}
\frac{\partial D_{\mathrm{KL}}}{\partial m_{\mathrm{free}}^2}=-\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{M(k)^2}+6\lambda V G(0)\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}+\frac{1}{2}\sum_k\frac{1}{M(k)}.
\end{equation}
Merge the first and third terms. Because
\begin{equation}
\frac{1}{M(k)}=\frac{M(k)}{M(k)^2},
\end{equation}
we have
\begin{equation}
-\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{M(k)^2}+\frac{1}{2}\sum_k\frac{1}{M(k)}=\frac{1}{2}\sum_k\frac{M(k)-[\tilde{K}(k)+m^2]}{M(k)^2}.
\end{equation}
And since
\begin{equation}
M(k)-[\tilde{K}(k)+m^2]=[\tilde{K}(k)+m_{\mathrm{free}}^2]-[\tilde{K}(k)+m^2]=m_{\mathrm{free}}^2-m^2,
\end{equation}
we get
\begin{equation}
-\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{M(k)^2}+\frac{1}{2}\sum_k\frac{1}{M(k)}=\frac{1}{2}\sum_k\frac{m_{\mathrm{free}}^2-m^2}{M(k)^2}.
\end{equation}
At the same time, since
\begin{equation}
\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}=-\frac{1}{2V}\sum_k\frac{1}{M(k)^2},
\end{equation}
so
\begin{equation}
\sum_k\frac{1}{M(k)^2}=-2V\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}.
\end{equation}
Therefore
\begin{equation}
\frac{\partial D_{\mathrm{KL}}}{\partial m_{\mathrm{free}}^2}=V\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}\left[m^2-m_{\mathrm{free}}^2+6\lambda G(0)\right].
\end{equation}
If the KL minimum is located inside the allowed set $\mathcal A$, it must satisfy:
\begin{equation}
\frac{\partial D_{\mathrm{KL}}}{\partial m_{\mathrm{free}}^2}=0.
\end{equation}
Since $\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}\neq 0$, we get the equation:
\begin{equation}
m^2-m_{\mathrm{free}}^2+6\lambda G(0)=0.
\end{equation}
Which means
\begin{equation}
m_{\mathrm{free}}^2=m^2+6\lambda G(0).
\end{equation}
Where
\begin{equation}
G(0)=\frac{1}{V}\sum_k\frac{1}{2[\tilde{K}(k)+m_{\mathrm{free}}^2]}.
\end{equation}
Therefore, the first and second principles restrict the initial prior to a family of Gaussian distributions in momentum space; the third principle finds the optimal mass parameter from this family by minimizing the initial KL divergence.

\subsubsection{Numerical Solution for $L=14$}

For:
\begin{equation}
L=14,\qquad m^2=-4,\qquad \lambda=5.113,
\end{equation}
the lattice volume is
\begin{equation}
V=L^2=196.
\end{equation}
Now we need to solve
\begin{equation}
m_{\mathrm{free}}^2=-4+6\times 5.113\times\frac{1}{196}\sum_{n_1=0}^{13}\sum_{n_2=0}^{13}\frac{1}{2[\tilde{K}(n_1,n_2)+m_{\mathrm{free}}^2]}.
\end{equation}
Where
\begin{equation}
\tilde{K}(n_1,n_2)=4\sin^2\left(\frac{\pi n_1}{14}\right)+4\sin^2\left(\frac{\pi n_2}{14}\right).
\end{equation}
Solving this equation numerically gives us:
\begin{equation}
m_{\mathrm{free},\star}^2\approx 0.6005269985.
\end{equation}
The corresponding zero-distance two-point function is:
\begin{equation}
G(0)\approx 0.1499617641.
\end{equation}
Verifying the right side of the self-consistent equation:
\begin{equation}
-4+6\times 5.113\times 0.1499617641\approx 0.6005269985.
\end{equation}
So, for the parameter set $L=14$, $m^2=-4$, $\lambda=5.113$, under our current action normalization convention, the optimal free-field prior mass parameter in the sense of KL is:
\begin{equation}
m_{\mathrm{free},\star}^2\approx 0.6005.
\end{equation}

\subsubsection{Gradient Descent and Acceptance Rate Comparison under Different Initial Distributions}

Based on our calculations, when $m_{\mathrm{free},\star}^2\approx 0.6005$, the KL divergence in the initial stage should be smaller than picking any other $m_{\mathrm{free}}$. We compared the loss descent history and MH acceptance rates under different initial distributions.
Based on our calculations, when $m_{\mathrm{free},\star}^2\approx 0.6005$, the KL divergence in the initial stage should be smaller than picking any other $m_{\mathrm{free}}^2$. We compared the loss descent history and Metropolis-Hastings (MH) acceptance rates under different initial distributions.

Specifically, we compared the standard Gaussian distribution against the free-field distribution. For the standard Gaussian prior, we evaluated two different network architectures: a smaller model with 32 channels and 12 coupling layers, and a larger model with 64 channels and 18 coupling layers. For the free-field prior, we uniformly applied the smaller architecture (32 channels, 12 coupling layers) while varying the mass parameter $m_{\mathrm{free}}^2$ and the mean shift $c$ .

As shown in Figure~\ref{fig:30k_comparison}, the free-field distribution with our derived optimal parameter ($m_{\mathrm{free},\star}^2 \approx 0.6005, c=0$) exhibits outstanding performance. It not only achieves the lowest initial loss but also converges to the highest acceptance rate. Moreover, it is evident that setting the mean shift to a non-zero value ($c=1.0$) negatively impacts the acceptance rate and introduces significant interference during the initial training phase.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{paper_figure_30k_comparison_final.png}
    \caption{Comparison of training loss history and MCMC acceptance rates under different initial distributions. The standard Gaussian prior is tested with two different network sizes, while the free-field prior is evaluated under varying mass parameters $m_{\mathrm{free}}^2$ and mean shifts $c$. All free-field experiments utilize the 32-channel, 12-layer architecture.}
    \label{fig:30k_comparison}
\end{figure}





\subsection{Introduction of $\mathbb{Z}_2$ Symmetry}

The 2D $\phi^4$ theory has an obvious global $\mathbb{Z}_2$ symmetry, defined by this transformation:
\begin{equation}
\phi(x)\rightarrow -\phi(x).
\end{equation}
Because the target action only has even powers of the field, meaning $S[\phi]=S[-\phi]$, the target probability distribution naturally satisfies:
\begin{equation}
p(\phi)=p(-\phi).
\end{equation}
For an ideal generative model, we'd expect the generated distribution to also satisfy:
\begin{equation}
q_\theta(\phi)=q_\theta(-\phi).
\end{equation}
If the model fails to fully learn this symmetry, the trained distribution might lean toward a specific magnetization direction. When the double-well structure is prominent or training isn't sufficient, this happens a lot, causing the model to get stuck around a single mode. While this might not be topological freezing in the strictest sense, we can think of it as mode freezing or mode imbalance between the two $\mathbb{Z}_2$ sectors.

The most direct fix is to hardcode this symmetry into the network architecture, forcing the generative mapping to be an odd function:
\begin{equation}
f_\theta(-z)=-f_\theta(z).
\end{equation}
If the prior distribution itself is symmetric, meaning $r(z)=r(-z)$, and the mapping satisfies this odd-function constraint, then the generated distribution will naturally satisfy $q_\theta(\phi)=q_\theta(-\phi)$.

However, in Real NVP-style affine coupling layers, the transformation is usually written as:
\begin{equation}
\phi_b'=\left(\phi_b-t_\theta(\phi_a)\right)\exp[-s_\theta(\phi_a)].
\end{equation}
To make the whole mapping satisfy the odd-function property, we have to enforce strict parity constraints on the functions $s_\theta$ and $t_\theta$. If the input is globally negated ($\phi_a \rightarrow -\phi_a, \phi_b \rightarrow -\phi_b$), we demand the output to also be globally negated. This means the scaling function $s_\theta$ must be an even function:
\begin{equation}
s_\theta(-\phi_a)=s_\theta(\phi_a),
\end{equation}
and the translation function $t_\theta$ must be an odd function:
\begin{equation}
t_\theta(-\phi_a)=-t_\theta(\phi_a).
\end{equation}
Only then can the affine coupling transformation globally satisfy $f_\theta(-\phi)=-f_\theta(\phi)$.

Forcing these parity properties onto a real neural network comes with huge penalties. To make $s_\theta$ even, the usual trick is to make it depend on $\phi_a^2$ or other even-function features. To make $t_\theta$ odd, you need to constrain the network architecture or use a specific mix of activation functions (like $\tanh$). We tried to hardcode the $\mathbb{Z}_2$ symmetry using aggressive channel splitting and manually designing odd/even functions through specific activations, but the training results were pretty terrible.

The main problems we ran into were:
\begin{enumerate}
    \item A sharp drop in the network's expressive power;
    \item Severe restrictions on the degrees of freedom available for the scaling and translation functions;
    \item Worsened gradient propagation in the early training stages;
    \item The model's inability to flexibly correct the difference between the free-field prior and the true target distribution.
\end{enumerate}

So, even though hardcoding $\mathbb{Z}_2$ symmetry looks elegant on paper, with our current model size and training setup, the loss of expressive power way offsets any benefits from the symmetry constraints.

A milder alternative is using a Soft Penalty. We can leave the network architecture alone and instead add an extra penalty term related to global magnetization into the loss function. Defining the mean field (or magnetization) of a single configuration as:
\begin{equation}
\bar{\phi}=\frac{1}{V}\sum_x\phi(x),
\end{equation}
we can add the following symmetry penalty term:
\begin{equation}
\mathcal{L}_{\mathrm{sym}}=V\lambda_{\mathrm{sym}}\left(\frac{1}{V}\sum_x\phi(x)\right)^2=V\lambda_{\mathrm{sym}}\bar{\phi}^2.
\end{equation}
This penalty term pushes the mean field of every individual generated configuration close to zero, trying to suppress the model's tendency to lean toward a specific magnetization direction. The total loss then becomes:
\begin{equation}
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{KL}}+\mathcal{L}_{\mathrm{sym}}.
\end{equation}

However, our experimental results clearly show that this soft penalty method is also very suboptimal and directly hurts training stability. To quantitatively evaluate the impact of the soft $\mathbb{Z}_2$ penalty, we tracked the Markov Chain Monte Carlo (MCMC) acceptance rate and the expectation value of the mean field $\langle\phi\rangle$ during training.

\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\textwidth]{mcmc_comparison_full.png}
    \caption{MCMC acceptance rate and mean field expectation (Global View).}
    \label{fig:mcmc_comparison}
\end{figure}

\FloatBarrier
Thinking about it from an information perspective, our free-field prior significantly boosts training efficiency by injecting partial information from the target distribution. This indicates that simply adding a soft $\mathbb{Z}_2$ penalty does not provide the model with enough effective, usable structural information. Compared to the free-field prior, which directly changes the input distribution and pre-codes the kinetic structure, the soft penalty merely alters the optimization objective and easily competes with the primary KL task. Therefore, under our current experimental setup, it did not bring the expected benefits. Moving forward, we might need to explore other methods to inject $\mathbb{Z}_2$ symmetry information, perhaps by studying additional equations satisfied by the target distribution to find symmetry constraints that are easier for the network to learn.









\section{Verification of Physical Observables and Autocorrelation Time, and Computational Overhead Comparison between Flow-based and Traditional Algorithms}

\subsection{Physical Observables}
\FloatBarrier
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{physics_observables_comparison.png}
    \caption{Comparison of physical observables. Left: The connected two-point correlation function $\tilde{G}_c(0,t)$ as a function of Euclidean time $t$. Right: The effective mass $m_p^{\mathrm{eff}}$ extracted from the correlation function. The measurements from the ML model are compared with traditional HMC and Local update methods.}
    \label{fig:physics_observables}
\end{figure}
\FloatBarrier
\subsection{Autocorrelation Time}

\subsection{Computational Overhead Comparison}

\section{Summary}

\appendix
\section{Proof of $2\hat{\mu}$ Translational Equivariance}
\label{app:proof translational equivariance}
This part, we prove that the convolutional neural network for the normalizing flow possesses $2\hat{\mu}$ translational equivariance:

Define the mask matrix $M$, where $M=1$ represents black sites (Frozen), and $M=0$ represents red sites (Updated). Define the operator for translating by $2\hat{\mu}$ sites as $T_{2\hat{\mu}}$. Based on the geometric properties of the checkerboard lattice, the mask remains unchanged under a $2\hat{\mu}$ translation:
\begin{equation}
T_{2\hat{\mu}}M = M, \quad T_{2\hat{\mu}}(1-M) = 1-M
\end{equation}
At the same time, because the networks $s$ and $t$ use CNNs with circular padding, they inherently satisfy strict translational equivariance. For any input field $\psi$:
\begin{equation}
T_{2\hat{\mu}}t(\psi) = t(T_{2\hat{\mu}}\psi), \quad T_{2\hat{\mu}}s(\psi) = s(T_{2\hat{\mu}}\psi)
\end{equation}
Also, the translation operator commutes with element-wise multiplication $\odot$: $T_{2\hat{\mu}}(A \odot B) = (T_{2\hat{\mu}}A) \odot (T_{2\hat{\mu}}B)$.

\begin{enumerate}
    \item \textbf{Red site update layer $f_r$}: First step, freeze black sites ($M$), update red sites ($1-M$). The network mapping is $f_r$:
    \begin{equation}
    f_r(\phi) = M \odot \phi + (1-M) \odot \left[ (\phi - t_r(M \odot \phi)) \odot e^{-s_r(M \odot \phi)} \right]
    \end{equation}
    We first prove that the single layer $f_r$ has $2\hat{\mu}$ translational equivariance. Apply $T_{2\hat{\mu}}$ to both sides and distribute it inside:
    \begin{equation}
    T_{2\hat{\mu}} f_r(\phi) = (T_{2\hat{\mu}}M) \odot (T_{2\hat{\mu}}\phi) + T_{2\hat{\mu}}(1-M) \odot \left[ (T_{2\hat{\mu}}\phi - T_{2\hat{\mu}}t_r(M \odot \phi)) \odot e^{-T_{2\hat{\mu}}s_r(M \odot \phi)} \right]
    \end{equation}
    Substitute the translation invariance of the mask $T_{2\hat{\mu}}M = M$, and the CNN equivariance $T_{2\hat{\mu}}t_r(M \odot \phi) = t_r(M \odot T_{2\hat{\mu}}\phi)$:
    \begin{equation}
    T_{2\hat{\mu}} f_r(\phi) = M \odot (T_{2\hat{\mu}}\phi) + (1-M) \odot \left[ (T_{2\hat{\mu}}\phi - t_r(M \odot T_{2\hat{\mu}}\phi)) \odot e^{-s_r(M \odot T_{2\hat{\mu}}\phi)} \right]
    \end{equation}
    Looking at the right side of the equation, this is exactly the form of substituting $T_{2\hat{\mu}}\phi$ as the initial input into $f_r$, so:
    \begin{equation}
    T_{2\hat{\mu}} f_r(\phi) = f_r(T_{2\hat{\mu}}\phi)
    \end{equation}

    \item \textbf{Black site update layer $f_b$}: Second step, freeze red sites ($1-M$), update black sites ($M$). The network mapping is $f_b$:
    \begin{equation}
    f_b(\psi) = (1-M) \odot \psi + M \odot \left[ (\psi - t_b((1-M) \odot \psi)) \odot e^{-s_b((1-M) \odot \psi)} \right]
    \end{equation}
    Similarly, apply $T_{2\hat{\mu}}$ to both sides and use the equivariance and invariance properties:
    \begin{equation}
    T_{2\hat{\mu}} f_b(\psi) = (1-M) \odot (T_{2\hat{\mu}}\psi) + M \odot \left[ (T_{2\hat{\mu}}\psi - t_b((1-M) \odot T_{2\hat{\mu}}\psi)) \odot e^{-s_b((1-M) \odot T_{2\hat{\mu}}\psi)} \right]
    \end{equation}
    This gives us:
    \begin{equation}
    T_{2\hat{\mu}} f_b(\psi) = f_b(T_{2\hat{\mu}}\psi)
    \end{equation}
    
    \item \textbf{$2\hat{\mu}$ equivariance of the full flow model $F$}: A complete update step is the composite mapping of the red and black layers:
    \begin{equation}
    F(\phi) = (f_b \circ f_r)(\phi) = f_b(f_r(\phi))
    \end{equation}
    We apply $T_{2\hat{\mu}}$ to the entire composite process:
    \begin{equation}
    T_{2\hat{\mu}} F(\phi) = T_{2\hat{\mu}} [f_b(f_r(\phi))]
    \end{equation}
    Because $f_b$ has $2\hat{\mu}$ equivariance, the translation operator $T_{2\hat{\mu}}$ can pass through the outer $f_b$:
    \begin{equation}
    T_{2\hat{\mu}} [f_b(f_r(\phi))] = f_b(T_{2\hat{\mu}} f_r(\phi))
    \end{equation}
    Then, because $f_r$ also has $2\hat{\mu}$ equivariance, $T_{2\hat{\mu}}$ continues to pass through the inner $f_r$:
    \begin{equation}
    f_b(T_{2\hat{\mu}} f_r(\phi)) = f_b(f_r(T_{2\hat{\mu}}\phi))
    \end{equation}
    Writing the result back into composite form completes the proof:
    \begin{equation}
    T_{2\hat{\mu}} F(\phi) = F(T_{2\hat{\mu}}\phi)
    \end{equation}
\end{enumerate}

Therefore, even though the CNN itself has translational equivariance, the overall flow architecture with checkerboard coupling usually only preserves $2\hat{\mu}$ translational equivariance, not full $1\hat{\mu}$ equivariance, even if we use different network parameters for the red and black grids. This is an unavoidable consequence of the mask structure.





\section{Derivation of the Free-Field Prior in Momentum Space}
\label{app:free_field_prior}

The free scalar field action in continuous space is given by:
\begin{equation}
S_{\mathrm{free}}=\int d^2x\,\phi(x)\left(-\partial^2 + m_{\mathrm{free}}^2\right)\phi(x).
\end{equation}
On a periodic lattice, we use the discrete Fourier transform to map the real space field $\phi(x)$ to momentum space:
\begin{equation}
\phi(x)=\frac{1}{\sqrt{V}}\sum_k\tilde{\phi}(k)e^{ik\cdot x},
\end{equation}
where $V=L^2$ is the lattice volume, and the allowed momenta are $k_\mu=\frac{2\pi n_\mu}{L}$. Since $\phi(x)$ is a real scalar field, meaning $\phi^*(x)=\phi(x)$, the Fourier modes in momentum space have to satisfy the Hermitian condition:
\begin{equation}
\tilde{\phi}^*(-k)=\tilde{\phi}(k).
\end{equation}

On a discrete lattice, the Laplacian operator is diagonalized in momentum space. The eigenvalues of the 2D periodic lattice Laplacian are:
\begin{equation}
\tilde{K}(k)=\sum_{\mu=1}^{2}\left(2-2\cos k_\mu\right)=4\sin^2\frac{k_1}{2}+4\sin^2\frac{k_2}{2}.
\end{equation}
Therefore, the quadratic form of the free field can be represented in momentum space as:
\begin{equation}
S_{\mathrm{free}}=\sum_k\left[\tilde{K}(k)+m_{\mathrm{free}}^2\right]|\tilde{\phi}(k)|^2.
\end{equation}

Defining $\tilde{M}(k)=\tilde{K}(k)+m_{\mathrm{free}}^2$, the free-field distribution becomes:
\begin{equation}
r(\phi)=\frac{1}{Z_{\mathrm{free}}}\exp\left[-\sum_k\tilde{M}(k)|\tilde{\phi}(k)|^2\right].
\end{equation}
From this expression, it is obvious that different momentum modes of the free field are completely decoupled in momentum space. If we formally write the complex Fourier modes as $\tilde{\phi}(k)=\phi^R(k)+i\phi^I(k)$, both the real and imaginary parts of each mode correspond to independent Gaussian variables. Given standard normal variables $\gamma \sim \mathcal{N}(0,1)$, we can generate the corresponding free-field modes like this:
\begin{equation}
\phi^R(k)=\frac{\gamma^R(k)}{\sqrt{2\tilde{M}(k)}},\qquad\phi^I(k)=\frac{\gamma^I(k)}{\sqrt{2\tilde{M}(k)}}.
\end{equation}
Using the Fast Fourier Transform (FFT) to transform these momentum modes back to real space, we can efficiently generate the free-field samples required for the initial prior distribution.


\section{Minimizing the Initial KL Divergence}
\label{app:kl_divergence}

The target distribution is the Boltzmann distribution $p(\phi)=\frac{1}{Z}e^{-S_{\mathrm{target}}[\phi]}$, where the target action is:
\begin{equation}
S_{\mathrm{target}}[\phi]=\sum_x\left[\phi(x)(-\Box\phi)(x)+m^2\phi(x)^2+\lambda\phi(x)^4\right].
\end{equation}
Its quadratic part in momentum space is $S_{\mathrm{quad}}[\phi]=\sum_k[\tilde{K}(k)+m^2]|\tilde{\phi}(k)|^2$.

The KL divergence between the free-field prior $q_{m_{\mathrm{free}}^2}$ and the target $p$ is defined as:
\begin{equation}
D_{\mathrm{KL}}(q_{m_{\mathrm{free}}^2}||p)=\int \mathcal{D}\phi\,q_{m_{\mathrm{free}}^2}(\phi)\log\frac{q_{m_{\mathrm{free}}^2}(\phi)}{p(\phi)}.
\end{equation}
Plugging in $p(\phi)$, we get:
\begin{equation}
D_{\mathrm{KL}}(q_{m_{\mathrm{free}}^2}||p)=\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{target}}(\phi)]-H(q_{m_{\mathrm{free}}^2})+\log Z,
\end{equation}
where $H(q_{m_{\mathrm{free}}^2})=-\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[\log q_{m_{\mathrm{free}}^2}(\phi)]$ is the entropy of the prior.

For the quadratic term expected value, substituting $\left\langle|\tilde{\phi}(k)|^2\right\rangle=\frac{1}{2[\tilde{K}(k)+m_{\mathrm{free}}^2]}$, we get:
\begin{equation}
\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{quad}}]=\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{\tilde{K}(k)+m_{\mathrm{free}}^2}.
\end{equation}
For the quartic interaction term, using Wick's theorem for a Gaussian distribution, we have $\left\langle\phi(x)^4\right\rangle=3\left\langle\phi(x)^2\right\rangle^2=3G(0)^2$, yielding:
\begin{equation}
\mathbb{E}_{q_{m_{\mathrm{free}}^2}}[S_{\mathrm{int}}]=3\lambda V G(0)^2.
\end{equation}

The entropy of the Gaussian free-field prior, ignoring constants independent of $m_{\mathrm{free}}^2$, is:
\begin{equation}
H(q_{m_{\mathrm{free}}^2})=-\frac{1}{2}\sum_k\log[\tilde{K}(k)+m_{\mathrm{free}}^2].
\end{equation}

Thus, the KL divergence can be written as:
\begin{equation}
D_{\mathrm{KL}}(m_{\mathrm{free}}^2)=\frac{1}{2}\sum_k\frac{\tilde{K}(k)+m^2}{\tilde{K}(k)+m_{\mathrm{free}}^2}+3\lambda V G(0)^2+\frac{1}{2}\sum_k\log[\tilde{K}(k)+m_{\mathrm{free}}^2].
\end{equation}
Where $G(0)=\frac{1}{V}\sum_k\frac{1}{2[\tilde{K}(k)+m_{\mathrm{free}}^2]}$.

To find the minimum, we take the derivative with respect to $m_{\mathrm{free}}^2$. Let $M(k)=\tilde{K}(k)+m_{\mathrm{free}}^2$. Since $\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}=-\frac{1}{2V}\sum_k\frac{1}{M(k)^2}$, the derivative of the KL divergence simplifies to:
\begin{equation}
\frac{\partial D_{\mathrm{KL}}}{\partial m_{\mathrm{free}}^2}=V\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}\left[m^2-m_{\mathrm{free}}^2+6\lambda G(0)\right].
\end{equation}
Setting this derivative to zero (and noting $\frac{\partial G(0)}{\partial m_{\mathrm{free}}^2}\neq 0$), we arrive at the self-consistent equation:
\begin{equation}
m_{\mathrm{free}}^2=m^2+6\lambda G(0).
\end{equation}



\begin{thebibliography}{99}

\bibitem{Albergo2019}
M.~S.~Albergo,
G.~Kanwar,
and P.~E.~Shanahan,
``Flow-based Generative Models for Markov Chain Monte Carlo in Lattice Field Theory,''
Phys.\ Rev.\ D {\bf 100}, 034515 (2019).

\end{thebibliography}


\end{document}