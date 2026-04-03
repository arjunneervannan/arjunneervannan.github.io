---
layout: default
title: "An Intuitive Explanation of Diffusion LLMs"
date: 2026-02-03
---

# An Intuitive Explanation of Diffusion LLMs

Diffusion LLMs have gained recent attention for their fast generation time, especially when compared to traditional autoregressive models. In this post, I aim to build an intuitive explanation for Diffusion LLMs, starting from the first principles for generative models, moving through a derivation of VAE, and its connection to Diffusion models. I'll keep this as math-light as possible and will assume no prior knowledge on diffusion for the reader.

This post collates material from these readings. I found these quite helpful while creating this blog and will reference them throughout.
- [Understanding Diffusion Models: A Unified Perspective (Calvin Luo)](https://arxiv.org/pdf/2208.11970)
- [Variational Autoencoders (Yoni Gottesman)](https://yonigottesman.github.io/2023/03/11/vae.html)
- [CS231n Lecture 15: Generative Models (2023)](https://cs231n.stanford.edu/slides/2023/lecture_15.pdf)
- [A Short Note on Variational Autoencoders](https://faculty.washington.edu/yenchic/short_note/note_vae.pdf)
- [Diffusion Models as a Kind of VAE (Angus Turner)](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html)
- [What are Diffusion Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

## Background

Feel free to skip this section if you are familiar with probability distributions, expectations of random variables, and KL divergence.

At a high level, a probability distribution assigns likelihoods to outcomes. For continuous data we talk about a density $p(x)$; for discrete data, a mass function. A joint distribution $p(x, z)$ describes two variables together, and the conditional $p(x \mid z)$ captures how $x$ behaves given $z$.

Marginalization means summing or integrating out a variable:

$$
p(x) = \int p(x, z)\,dz \quad \text{or} \quad p(x) = \sum_z p(x, z).
$$

The **expectation** (or expected value) of a random variable is just its average value if you repeated the process many times. For a continuous variable $X$:

  $$
  \mathbb{E}[X] = \int x\,p(x)\,dx
  $$

**Bayes' rule** flips conditionals:

$$
p(x \mid z) = \frac{p(z \mid x)\,p(x)}{p(z)}.
$$

The Gaussian (normal) distribution is the workhorse in diffusion models. In 1D:

$$
\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right),
$$

where $\mu$ is the mean and $\sigma^2$ is the variance. In higher dimensions,
we replace $\sigma^2$ with a covariance matrix $\Sigma$ and the same idea holds.

The **Kullback-Leibler (KL) divergence** is a measure of how one probability distribution diverges from a second, reference probability distribution. For two distributions $P$ and $Q$ over the same random variable $x$, the KL divergence from $Q$ to $P$ is defined as:

$$
\mathrm{KL}(Q \parallel P) = \int q(x) \log \frac{q(x)}{p(x)}\,dx
$$

where $q(x)$ and $p(x)$ are the densities (or probability mass functions in the discrete case) of $Q$ and $P$ respectively.

**Interpretation:**  
- KL divergence is always non-negative: $\mathrm{KL}(Q \parallel P) \geq 0$.
- It is zero only if $Q$ and $P$ are the same almost everywhere.
- It measures the "extra" amount of information required to represent samples from $Q$ using a code optimized for $P$, rather than a code optimized for $Q$ itself.

**Important:**  
KL divergence is **not symmetric**; in general,
$$
\mathrm{KL}(Q \parallel P) \ne \mathrm{KL}(P \parallel Q).
$$
This means that the "distance" from $Q$ to $P$ is not the same as from $P$ to $Q$, so KL divergence is not a true distance metric.

In variational methods (like VAEs and diffusion models), we often use KL divergence to quantify how different our variational approximation $q_\phi(z|x)$ is from the true posterior $p(z|x)$. Minimizing KL divergence helps us make our approximation as close as possible to the true distribution.


## Introduction to Generative Models

Given a true data distribution $D$ found in nature, the goal of generative models is to model the distribution $p(x)$ for observed samples $x \sim D$. In real life, this could be anything -- generating images, text, videos, code, and more. 

Now, let's think about the concept of a latent variable $z$. This variable contains some information that is not immediately observable but directly may represent information present in $x$. A nice explanation that [1] gives is Plato's [Allegory of the Cave](https://en.wikipedia.org/wiki/Allegory_of_the_cave): we only observe the "shadows" (data $x$), while the latent causes $z$ are hidden behind the scenes. 

Similarly, we can think about "nature's distribution" being the prior $p(z)$, and by approximating the true posterior distribution $p(z|x)$ we can generate new samples of $x$. We do not know $p(z|x)$ and $p(z)$ but if we did, we'd be able to generate as many examples of $x$ as possible.


**INSERT IMAGE HERE**


## Variational Autoencoders and ELBO Derivation

Mathematically, let's express the above goal. We can decompose the probability distribution $p(x)$ as follows:

$$p(x) = \int p(x, z) dz \tag{1}$$
Here, we marginalize the variable $z$, and integrating over $dz$ allows us to pull that out.
$$p(x) = \frac{p(x, z)}{p(z | x)} \tag{2}$$
And this one is just chain rule of probability.

**INSERT IMAGE HERE THAT SHOWS THE GRAPH OF VAE**

Now, note that actually calculating (1) is intractable because we cannot integrate over all variables $z$, and $p(z|x)$ is a ground truth about the latent variables that we do not have access to. 

Instead, we can provide a lower bound on this that we can instead optimize on to predict this distribution $p(x)$. Let's pick a family of functions $Q$, and optimize over some function $q^* \in Q$ such that $$q^*(z|x) \approx p(z|x)$$

We expect $Q$ to be a simple family of functions that can model the more complex posterior distribution found in nature. In practice, $Q$ is parametrized by a deep neural network.

The idea here is that by learning to encode information about $x$ into $z$ via the posterior, and regularizing this posterior toward a simple prior $p(z)$, we can later sample from the prior to generate new samples of $x$. 

Let's consider the log likelihood of (1):

$$
\begin{align}
\log p(x) &= \log \int p(x, z) dz \tag{3}\\
&= \log \int \frac{p(x, z) q_\phi(z|x)}{q_\phi(z|x)} dz \tag{4} \\
\end{align}
$$
Note that the integrand in (4) is just the expectation of $\frac{p(x, z)}{q_\phi(z | x)}$, which leaves us with:
$$\log p(x) = \log \mathbb{E}_{q_\phi (z|x)} \bigg[ \frac{p(x, z)}{q_\phi(z | x)} \bigg] \tag{5}$$

Next, we can apply Jensen's inequality here because $\log$ is a concave function. This gives us:
$$\log \mathbb{E}_{q_\phi (z|x)} \bigg[ \frac{p(x, z)}{q_\phi(z | x)} \bigg] \geq \mathbb{E}_{q_\phi (z|x)} \bigg[\log \frac{p(x, z)}{q_\phi(z | x)} \bigg] \tag{6}$$

and thus we get our lower bound -- or the Evidence Lower Bound (ELBO), which is defined as follows:
$$\log p(x) \geq \mathbb{E}_{q_\phi (z|x)} \bigg[\log \frac{p(x, z)}{q_\phi(z | x)} \bigg] \tag{7} $$


**INSERT IMAGE HERE THAT SHOWS THE DIFFERENCE BETWEEN LOSS AND ELBO**

If we maximize this log-likelihood, we'll get the optimal model here that can encode some information $z$ -- what's called an encoder. 

Another derivation of ELBO uses KL Divergence. We seek to minimize the KL Divergence between our parameterized model $q_\phi$ and $p$, 
$$
\begin{align}
 D_{\mathrm{KL}}\!\left(q_\phi(z \mid x) \,\|\, p(z \mid x)\right)
 &= \mathbb{E}_{q_\phi}\!\left[\log q_\phi(z \mid x) - \log p(z \mid x)\right] \tag{8} \\
 &= \mathbb{E}_{q_\phi}\!\left[\log q_\phi(z \mid x) - \log p(x, z) + \log p(x)\right] \tag{9} \\
 &= \log p(x) - \mathbb{E}_{q_\phi}\!\left[\log p(x, z) - \log q_\phi(z \mid x)\right] \tag{10}
\end{align}
$$
Rearranging gives the ELBO:
$$
\log p(x) = \mathrm{ELBO} + D_{\mathrm{KL}}\!\left(q_\phi(z \mid x) \,\|\, p(z \mid x)\right) \tag{11},
$$
so
$$
\mathrm{ELBO} = \mathbb{E}_{q_\phi}\!\left[\log p(x, z) - \log q_\phi(z \mid x)\right] \le \log p(x) \tag{12}
$$

This is another interesting result that's explained in [1] and [2]; the difference between ELBO and our log likelihood of the distribution $p(x)$ is the KL Divergence. Since $p(x)$ is a constant, minimizing KL Divergence is the same as maximizing ELBO. 

We can expand the ELBO term further. Using the chain rule:

$$\log p_\theta(x, z) = \log p_\theta(x \mid z) + \log p(z) \tag{13}$$

So:

$$\mathrm{ELBO} = \mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z) + \log p(z) - \log q_\phi(z \mid x)\right] \tag{14}$$

Grouping terms:

$$\mathrm{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z \mid x)}\left[\log p_\theta(x \mid z)\right]}_{\text{reconstruction}} - \underbrace{D_{\mathrm{KL}}\left(q_\phi(z \mid x) \,\|\, p(z)\right)}_{\text{regularization}} \tag{15}$$

The first term encourages the decoder to reconstruct $x$ from $z$, while the KL term regularizes the learned posterior toward the prior. In simpler terms, the reconstruction loss is generally used to train the decoder (parameterized by $\theta$) and the regularization term is used to train the encoder.

In order to actually differentiate this objective, one can use the [reparameterization trick](https://gregorygundersen.com/blog/2018/04/29/reparameterization/). 

This essentially encompasses the high-level details of the Variational Autoencoder. While I'm barely scratching the surface here, there's plenty of great readings on VAEs. In the next section, I'll connect these principles with Diffusion Models.

#### Drawbacks of VAE

**Simple priors.** Standard VAEs assume a simple prior $p(z) = \mathcal{N}(0, I)$. This works well when the true latent structure is roughly Gaussian, but for complex, multimodal data the prior may be too restrictive. The model is forced to squeeze all variation into a single distribution.

**Posterior collapse.** During training the KL regularization term can dominate, pushing the encoder $q_\phi(z \mid x)$ to match the prior so closely that $z$ carries almost no information about $x$. When this happens the decoder ignores $z$ entirely and just models an unconditional $p(x)$—the latent variable has "collapsed."

Hierarchical VAEs address both problems by introducing multiple layers of latent variables $z_1, z_2, \ldots, z_T$. Each layer conditions on the one above, creating a richer, more expressive prior.

## Diffusion Models

 Diffusion models take the idea of Hierarchical VAEs to an extreme: the "hierarchy" becomes a long Markov chain of latent variances, and the forward process is just Gaussian noise additions that's fixed. We assume a few critical properties about Diffusion models:

- **Latent Dimensions are rich.** We assume the latent dimensions are the same as input dimensions. Instead of latent variables $z$, we use latent variables $x_{1:T}$ which represent the sequence of perturbed samples given the input sample $x_0$.
- **Encoders are fixed.** The encoder or "forward" process in a diffusion model is fixed with a schedule of parameters that's used to apply Gaussian noise.
- **At timestep T, the sample is pure noise.** At the end of the forward process (which is T here), the input sample is pure noise and our decoder must learn to generate samples from pure noise.

Crucially, this differs from VAEs in three ways: (1) we don't learn an encoder, (2) the latent space is the same dimensionality as the data rather than a compressed bottleneck, and (3) generation happens iteratively through many small denoising steps rather than a single decoder pass.

**INSERT IMAGE HERE OF PICTURE GOING THROUGH GAUSSIAN NOISE**

Note that we abuse the notation a bit here -- in our prior derivation, $p$ was our encoder and $q$ was our decoder. Here, we'll refer to our forward Gaussian process with $q$, and our reverse process that we're trying to learn with $p_\theta$ where $\theta$ represents the set of parameters we're trying to learn.

#### Why Diffusion?

There are a few quesions that I had at this point when I was learning about diffusion models. First, why does diffusion work better if we don't train an encoder at all? Are we not losing information if we pass inputs through Gaussian noise filters? And second, why specifically Gaussian noise? Why not a different perturbation?

To answer the first question, learning adversarial encoders and decoders, as VAEs and GANs do, are incentivized to compress the input data to a smaller latent dimension $z$. This is the posterior collapse problem that is described above. In contrast, diffusion not only keeps the latent variables the same dimensions as the inputs by simply perturbing them, but also learns to solve a simpler problem: local inference. 

In the example of diffusion models for image generation, a singular step of Gaussian Noise only perturbs a small part of the image. As a result, our encoder has to only learn the inverse transformation one at a time. Crucially, certain properties of Diffusion Models (i.e., Markov) allow this objective to still be tractable.

Fundamentally, this is a slightly different problem than the objective that VAEs are optimizing. VAEs and HVAEs frame the problem as learning latent variables and decoding those latents. Diffusion tries to learn to reverse a stochastic dynamic process. In many ways, this is an easier problem because we're just learning to decode small noise reductions at each step. When compared with VAEs which try to decode images one-shot and from an arbitrary latent space that compresses information, this is a slightly easier problem -- and as we show below, the stochastic process that we are utilizing here has nice properties that make it tractable.

#### Why Gaussian Noise?

To answer the second question, Gaussian distributions have very nice properties. Let's consider our forward process (the "noising" step) as follows:

$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t I) \tag{16}$$

At each timestep $t$, we take the previous sample $x_{t-1}$, scale it down slightly by $\sqrt{1 - \beta_t}$, and add Gaussian noise with variance $\beta_t$.

The full forward trajectory is a Markov chain:

$$q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}) \tag{17}$$

What happens if we expand out this term? While this blog will not go into the details, [6], [1], and [7] (D3PM paper) all have great full-length derivations for this statement that leverage the properties of Gaussians, notably that merging two gaussians with variances $\sigma_1^2$ and $\sigma_2^2$ results in a new Gaussian $\mathcal{N}(0, (\sigma_1^2 + \sigma_2^2)\bold{I})$. Essentially, we see that we get a closed form solution for the Gaussian at some timestep $t$. Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, and assume $\epsilon_i \sim \mathcal{N}(0,1) \forall i$. Then:
$$
\begin{align}
x_t &= \sqrt{\alpha_{t}} x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1} \tag{18} \\
&= \sqrt{\prod_{s=1}^{t} \alpha_s} x_0 + \sqrt{1 - \prod_{s=1}^{t} \alpha_s} \epsilon_0 \tag{19}  \\
q(x_t \mid x_0)&= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1 - \bar{\alpha}_t) I) \tag{20}
\end{align}
$$

We've skipped much of the derivation here, but the key property that makes Gaussian Noise easy to work with here is that we can sample $x_t$ at any arbitrary time step $t$. Additionally, when learning the reverse process, because we can jump from any point $s$ to another point $t$ in time with a closed form solution, it becomes a lot easier to learn the reverse process here. Diffusion models are themselves inspired by thermodynamic physics. [6] has a good overview on the connection to Langevin Dynamics in Physics.

As $t \to T$, $\bar{\alpha}_t \to 0$, so $x_T \approx \mathcal{N}(0, I)$, which is pure noise. This means that, at the end of our forward process, we're left with pure Gaussian noise, and our objective for the reverse process that we want to learn will be to reverse this process.

#### A Brief Re-Derivation of ELBO, for Diffusion

How do we optimize this objective? Like we did for VAEs, we can optimize the ELBO. Please note that we'll again skip a few steps here -- there are many great resources that go super in-depth in this derivation, including [1], [6], [7], and [8]. 

To recap, we want to model the below process to generate a new sample from the pure noise $x_T$:

$$p_\theta(x_{t-1} \mid x_t) = p_\theta(x_t | x_{t-1}) \tag{21}$$

We also know that:

$$
\begin{align}
p_\theta(x_{0:T}) &= p_\theta(x_0, x_1, ... x_T) \tag{22}\\
&= p_\theta(x_0) \prod_{t=1}^T p(x_t \mid x_{t-1}) \tag{23}\\
\end{align}
$$

Note that going from (22) to (23) makes use of the same marginalization idea we had looked at before. Instead of explicitly marginalizing a latent variable $z$, we instead marginalize the perturbed examples of $x_0$, and we know that because of the Markov property here, namely that we have a closed form solution for the Gaussian to go between any two timesteps, we can explicitly marginalize out these as our latent variables.

We can then optimize the log-likelihood via ELBO:

$$
\begin{align}
\mathbb{E}_{q(x_0)}[\log p_\theta(x_0)] &= \mathbb{E}_{q(x_0)}\bigg[ \log \int p_\theta(x_{0:T}) dx_{1:T} \bigg] \tag{24} \\
&= \mathbb{E}_{q(x_0)}\bigg[ \log \int p_\theta(x_{0:T}) \frac{q_(x_{1:T} \mid x_0)}{q_(x_{1:T} \mid x_0)} dx_{1:T} \bigg] \tag{24} \\
&= \mathbb{E}_{q(x_0)} \bigg[ \log \mathbb{E}_{q(x_{1:T} \mid x_0)} \frac{p_\theta(x_{0:T})}{q_(x_{1:T} \mid x_0)} dx_{1:T} \bigg] \tag{25} \\
&= \mathbb{E}_{q(x_0:T)} \log \bigg[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \bigg] \tag{26} \\
&\geq \log \mathbb{E}_{q(x_0:T)} \bigg[ \frac{p_\theta(x_{0:T})}{q(x_{1:T} \mid x_0)} \bigg] \tag{27}
\end{align}
$$

Note that we follow a very similar derivation as what we saw for the ELBO for VAEs. We go from step (24) to (25) by the definition of expectation, (25) to (26) because $q(x_0:T) = q(x_0)q(x_{1:T})$ which allows us to combine the expectations, and from (26) to (27) by Jensen's inequality. This derivation is very similar to what [6] follows, starting from minimizing the Cross-entropy loss.

**WRITE ABOUT THE DECOMPOSITION OF THE LOSS HERE**

## Diffusion LLMs

(write about discrete diffusion here)

## Bibliography

[1] [Understanding Diffusion Models: A Unified Perspective (arXiv:2208.11970)](https://arxiv.org/pdf/2208.11970)

[2] [Variational Autoencoders (Yoni Gottesman)](https://yonigottesman.github.io/2023/03/11/vae.html)

[3] [CS231n Lecture 15: Generative Models (2023)](https://cs231n.stanford.edu/slides/2023/lecture_15.pdf)

[4] [A Short Note on Variational Autoencoders](https://faculty.washington.edu/yenchic/short_note/note_vae.pdf)

[5] [Diffusion Probabilistic Models, Part I](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html)

[6] [What are Diffusion Models? (Lilian Weng)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

[7] [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239)