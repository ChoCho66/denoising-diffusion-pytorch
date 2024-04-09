
# Code 跟 math 對應表

## U-Net

- `forward(x,t,classes)`: Training 時的 $\varepsilon_{\theta}(x_t,c)$ (包含忘記 $c$)
  - `cond_drop_prob`$=0\iff$ keep conditioning information
  - `cond_drop_prob`$=$ the probability of forgetting conditioning information

- `forward_with_cond_scale()`: Sampling 時的 $\widetilde{\varepsilon_{\theta}}(x_t,c)$
  
  - return a linear interpvolation of `scaled_logits` and `rescaled_logits` 
  
  - `cond_scale`$= s = w + 1$
  
  - `logits` = $\varepsilon_{\theta}(x_t,c)$
  - `null_logits` = $\varepsilon_{\theta}(x_t,\emptyset)$
  - `scaled_logits` = $\widetilde{\varepsilon_{\theta}}(x_t,c)$
  - `rescaled_logits` = $\widetilde{\varepsilon_{\theta}}(x_t,c)\cdot \dfrac{\mathtt{std}(\varepsilon_{\theta}(x_t,c))}{\mathtt{std}(\widetilde{\varepsilon_{\theta}}(x_t,c))}$
  

## GaussianDiffusion

- `posterior_mean_coef1`, `posterior_mean_coef2`:
  $$
  \begin{aligned}
    \mu_t(x_t,x_0) = \mathtt{posterior_mean_coef1}\quad x_0 + \mathtt{posterior_mean_coef2} \quad x_t.
  \end{aligned}
  $$


- `q_posterior(x_start, x_t, t)`:
  $$
  \begin{aligned}
    (x_0,x_t,t) \longmapsto \bigl( \mu_t(x_0,x_t), \Sigma_t, \log \Sigma_t \bigr)
  \end{aligned}
  $$

- `model_predictions(x, t, classes)`:

  - Return`['pred_noise', 'pred_x_start']`
    - For example, if `self.objective == 'pred_noise'`, then `pred_noise`$=$`forward_with_cond_scale`

    - 有用到 `forward_with_cond_scale`, 所以這裡是有關 sampling 的部分.
  
- `p_mean_variance(x, t, classes)`:
  $$
  \begin{aligned}
    (x_t,t,c) \longmapsto \bigl( \mu_{\theta}(x_t,c), \Sigma_t, \log \Sigma_t, \widehat{x}_0(x_t,c) \bigr),
  \end{aligned}
  $$
  where $p_{\theta}(x_t,c)\sim \mathcal{N}(\mu_\theta(x_t,c),\Sigma_t).$

- `p_sample(x, t: int, classes)`:
  
  - Given $(x_t,t,c).$ 
  - Sample $x_{t-1}\sim p_{\theta}(x_{t-1}\vert x_t).$
  - Return $\bigl( x_{t-1}(x_t,c),\widehat{x}_0(x_t,c)  \bigr).$
  
- `p_sample_loop(self, classes, shape)`: 
  - Sample $x_T,x_{T-1},\cdots, x_0$ inductively by `p_sample(x, t: int, classes)`. 
  - Return $x_0.$

- `ddim_sample`: DDIM 版的 `p_sample_loop`.

- `q_sample(x_start, t, noise)`: 
  - Input $(x_0,t,\overline{\varepsilon}_t).$
  - Return $x_t:= \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\overline{\varepsilon}_t.$

- `p_losses(x_start, t, classes, noise)`: 
  - Input: $(x_0,t,c,\varepsilon).$
  - `model_out = model(x, t, classes)`
  - Sample $x_t\sim q(x_t\vert x_0,c).$
  - Return $\left\lVert \underbrace{\varepsilon_{\theta}(x_t,c)}_{\text{U-net}(x_t,c)}-\underbrace{\varepsilon}_{\text{target}} \right\rVert^2.$

- `forward(img, classes)`: 
  - Input $(x_0,c).$
  - Random Sample $t.$
    - !!! Note 為 Training 時的 t
  - Return $\left\lVert \varepsilon_{\theta}(x_t,c)-\varepsilon \right\rVert^2$ by `p_losses(img, t, classes)`.
    - !!! Note 為 Training 時的 loss
