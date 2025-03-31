# Document

**Proposition 0.1.** Global Truncation-Aware truncation significantly enhances overall model accuracy.

**Proof.** The global whitening matrix $S_{k}=X_{k}^{\prime} X_{k}^{\prime T}$ exhibits controlled spectral properties, the effective condition number $\kappa\left(S_{k}\right)$ of the global whitening matrix is bounded by:

$$\kappa\left(S_{k}\right)\triangleq\frac{\sigma_{\max}\left(X_{k}^{\prime} X_{k}^{\prime T}\right)}{\sigma_{\min}\left(X_{k}^{\prime} X_{k}^{\prime T}\right)}\leq\frac{1+\sum_{i=0}^{k-1}\alpha_{i}\left\|X_{i} X_{i}^{T}\right\|_{2}}{\sigma_{\min}\left(X_{k} X_{k}^{T}\right)}\triangleq\kappa_{0}\qquad(1)$$

where the decay coefficients $\alpha_{i}$ follow:

$$\alpha_{i}=0.02\log\left(1+e^{(i+1)/ N}\right)\qquad(2)$$

The decay coefficients $\alpha_{i}$ ensures logarithmic decay of historical contributions,which implies:

$$\kappa\left(S_{k}\right)^{\text{global}}\ll\kappa\left(S_{k}\right)^{\text{local}}\quad\text{(Stabler gradients)}\qquad(3)}$$

Additionally, global features strictly dominate local features in mutual infor-mation, according to the condition number bound and cramer-rao inequality, we can get:

$$\frac{I\left(Y; W_{k}^{\prime} X_{k}^{\prime}\right)}{I\left(Y; W_{k} X_{k}\right)}\geq 1+\underbrace{\frac{\sigma_{\min}^{2}\left(W_{k}^{\prime} X_{k}^{\prime}\right)}{\sigma_{\max}^{2}\left(W_{k} X_{k}\right)}}_{\text{Feature quality}}\cdot\underbrace{\frac{\kappa_{0}^{-1}}{1-\kappa_{0}^{-1}}}_{\text{Stability benefit}}\qquad(4)$$

Combining Feature Stability Condition and Information Preservation, we can get:

$$H\left(Y\mid W_{k}^{\prime} X_{k}^{\prime}\right)\leq H\left(Y\mid W_{k} X_{k}\right)-\log\left(1+\frac{\sigma_{\min}^{2}\left(W_{k}^{\prime} X_{k}^{\prime}\right)}{\sigma_{\max}^{2}\left(W_{k} X_{k}\right)}\cdot\frac{1}{\kappa_{0}-1}\right)\qquad(5)$$

The prediction error $P_{e}=1-A(M)$ then satisfies:

$$P_{e}^{\text{global}}\leq P_{e}^{\text{local}}-\underbrace{\frac{\log\left(1+\frac{\sigma_{\min}^{2}}{\sigma_{\max}^{2}\left(\kappa_{0}-1\right)}\right)}{\log(|\mathcal{Y}|-1)}}_{\Delta(\text{Explicit Improvement Term})}\qquad(6)$$

we derive the global accuracy superiority:

$$A(M)_{\text{global}}\geq A(M)_{\text{local}}+\underbrace{\log\left(1+\frac{\sigma_{\min}^{2}}{\sigma_{\max}^{2}\left(\kappa_{0}-1\right)}\right)}_{\text{Explicit improvement term}}\frac{1}{\log(|\mathcal{Y}|-1)}\qquad(7)$$

This complete derivation demonstrates the direct role of global truncation-aware truncation in enhancing model accuracy through improved information preservation.

**Proposition 0.2.** Momentum-Enhanced Alternating Least Squares achieve a lower loss

**Proof.** We derive the momentum-enhanced alternating least squares method for optimizing the low-rank decomposition problem in MAS-SVD. The objective is to minimize the Frobenius norm between the original weight matrix W and its compressed counterpart $W^{\prime}$:

$$\mathcal{L}(U, V)=\left\|W X-U V^{T}\right\|_{F}^{2}$$

where WX is the observed matrix, U and V are the low-rank factor matrices to be optimized. When only update U, the objective function is[?]:

$$\mathcal{L}\left(U_{t+1}, V_{t}\right)=\mathcal{L}\left(U_{t}, V_{t}\right)-\Delta_{U}\mathcal{L}$$

In contrary, we alternatively fix one variable and optimize the other, and the objective function is:

$$\mathcal{L}\left(U_{t+1}, V_{t+1}\right)=\mathcal{L}\left(U_{t}, V_{t}\right)-\left(\Delta_{U}\mathcal{L}+\Delta_{V}\mathcal{L}\right)$$

Since $\Delta_{V}\mathcal{L}>0$, we have:

$$\mathcal{L}\left(U_{t+1}, V_{t+1}\right)<\mathcal{L}\left(U_{t+1}, V_{t}\right)\quad(11)$$

Similarly, when fix matrix U while update V, we have:

$$\mathcal{L}\left(U_{t+1}, V_{t+1}\right)<\mathcal{L}\left(U_{t}, V_{t+1}\right)\qquad(12)$$

This shows alternating updates of U and V lead to a greater reduction in objective function. Additionally, we introduce momentum terms for U and V to enhance the performance of Alternating Least Squares. Among them, the momentum for U is updated as:

$$m_{u}=\beta m_{u}+(1-\beta)\Delta U$$

where $m_{u}$ is the momentum term for U, initialized as a zero matrix, $\beta$ is the momentum coefficient(typically $\beta=0.9$ or 0.95), and $\Delta U$ is the update for matrix $U_{\text{new}}$ computed using the Alternating Least Squares update:

$$\Delta U=W X V\left(V^{T} V\right)^{-1}-U_{\text{old}}$$

where $U_{\text{old}}$ is the matrix before update, The updated matrix U is then computed as:

$$U_{\text{new}}=U_{\text{old}}+\eta m_{u}\qquad(15)$$

where $\eta$ is the learning rate. Similarly, the momentum term for matrix V is updated as:

$$m_{v}=\beta m_{v}+(1-\beta)\Delta V\quad(16)$$

where $m_{v}$ is the momentum term for V, initialized as a zero matrix, and $\Delta V$ is the update for matrix V computed using the Alternating Least Squares update:

$$\Delta V=\left(U^T U\right)^{-1} U^T W X-V_{\text{old}}\quad(17)$$

Similarly, $V_{\text{old}}$ is the matrix before update, The updated of matrix $V_{\text{new}}$ is then computed as:

$$V_{\text{new}}=V_{\text{old}}+\eta m_{v}$$

The objective function at iteration t is $\mathcal{L}_{t}$. After applying the momentum-enhanced updates, the change in the objective function can be approximated using a Taylor expansion:

$$\mathcal{L}_{t+1}\approx\mathcal{L}_{t}-\eta\left\langle m_{t},\nabla\mathcal{L}_{t}\right\rangle+O\left(\eta^{2}\right)$$

where $\langle\cdot,\cdot\rangle$ denotes the matrix inner product, $m_{t}$ is the momentum term(either $m_{u}$ or $\left.m_{v}\right)$, and $\nabla\mathcal{L}_{t}$ is the change of the objective function at iteration t.Under the Kurdyka-Lojasiewicz(KL) inequality, the momentum-enhanced ALS optimization satisfies:

$$\left\|\nabla\mathcal{L}\left(U_{t}, V_{t}\right)\right\|_{2}\leq\frac{C}{(1-\beta) t}$$

where C depends on initial conditions and learning rate $\eta,\beta\in(0,1)$ is the momentum coefficient. While the vanilla ALS optimization satisfies:

$$\left\|\nabla\mathcal{L}\left(U_{t}, V_{t}\right)\right\|_{2}\leq\frac{C}{t}$$

Hence, momentum-enhanced ALS achieves accelerated convergence compared to standard ALS. Additionally, The momentum update rule ensures the optimization avoids stationary points where:

$$\|\mathcal{L}\|\geq\frac{\beta}{1-\beta}\left\|m_{t-1}\right\|$$

This effectively filters out suboptimal critical points that would trap vanilla ALS.That is to say, When the update directions are consistent, the momentum term accumulates the update information, increasing the step size and accelerating the decrease of the loss. When the update directions are inconsistent, the momentum term smooths out the update direction, reducing oscillations in the loss function.Therefore, Momentum-enhanced Alternating Least Squares facilitates it easier to achieve a lower loss, avoiding over-adapting to local patterns and ensuring an optimized accuracy. $\square$
