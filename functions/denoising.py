import torch
import numpy as np

# in the paper Song et al. 2021, the authors use the following formula for alpha
# In Ho et al. 2020 \bar{alpha}_T = \prod_{t=1}^T (1 - \beta_t) which is alpha_t in Song et al. 2021 
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# sampling steps for ddim
def generalized_steps(x, 
                      seq,  # tau subsequence to sample from (times)
                      model, 
                      b, # the self.betas
                      **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            
            t = (torch.ones(n) * i).to(x.device)
            
            next_t = (torch.ones(n) * j).to(x.device)
            
            at = compute_alpha(b, t.long())
            
            at_next = compute_alpha(b, next_t.long())
            
            xt = xs[-1].to('cuda')
            
            # epsilon(x_t) 
            et = model(xt, t)
            
            # prediction of x0 at time t
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # sequence of x0 predictions
            x0_preds.append(x0_t.to('cpu'))
            
            # sigma t in equation 12 in Song et al. 2021
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            
            # coefficient before model output
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            # prediction * sqrt(alpha_t) + c1 * N(0, 1) + c2 * epsilon(x_t
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds


# sampling steps for ddim with RMSProp (following from Wang et al. 2023)
def reversed_generalized_steps(x, 
                      seq,  # tau subsequence to sample from (times)
                      model, 
                      b, # the self.betas
                      **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        # reverse sampling to get latents
        # as in 
        for i, j in zip(seq, seq_next):
            
            t = (torch.ones(n) * i).to(x.device)
            
            next_t = (torch.ones(n) * j).to(x.device)
            
            at = compute_alpha(b, t.long())
            
            at_next = compute_alpha(b, next_t.long())
            
            xt = xs[-1].to('cuda')
            
            # epsilon(x_t) 
            et = model(xt, t)
            
            # prediction of x0 at time t
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # sequence of x0 predictions
            x0_preds.append(x0_t.to('cpu'))
            
            # sigma t in equation 12 in Song et al. 2021
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            
            # coefficient before model output
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            
            # prediction * sqrt(alpha_t) + c1 * N(0, 1) + c2 * epsilon(x_t)
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

# Abduction of exogenous variables might require storing the noise predictions




# Boosting Diffusion Models with an Adaptive Momentum Sampler
def generalized_steps_rms1(x, 
                          seq,  # tau subsequence to sample from (times)
                          model, 
                          b, # the self.betas
                          beta_rms, # the beta param for RMSProp
                          use_scalar_V="norm", # whether to use scalar V
                          eps=1e-8, # epsilon for RMSProp,
                          debug = False,
                          **kwargs):
    
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        V = torch.ones_like(x)
        
        for i, j in zip(reversed(seq), reversed(seq_next)):
            
            t = (torch.ones(n) * i).to(x.device)
            
            next_t = (torch.ones(n) * j).to(x.device)
            
            at = compute_alpha(b, t.long())
            
            at_next = compute_alpha(b, next_t.long())
            
            xt = xs[-1].to('cuda')
            
            # define xt bar in paper
            xt_bar = xt / at.sqrt()
            
            # epsilon(x_t) 
            et = model(xt, t)
            
            # prediction of x0 at time t
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # coefficient before xt
            # coeff_xt = torch.sqrt(at_next) / torch.sqrt(at)
            
            # sigma t in equation 12 in DDIM Song et al. 2021
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            
            # mu in the paper
            mu = torch.sqrt((1 - at_next - c1**2) / at_next) - torch.sqrt((1 - at) / at)            
            
            # dxt bar in the paper
            dxt_bar = mu * et + (c1 / torch.sqrt(at_next)) * torch.randn_like(x)
            
            # update moving average
            if use_scalar_V=="norm":
                V = beta_rms * V + (1 - beta_rms) * (torch.linalg.norm(dxt_bar)**2)
                
            elif use_scalar_V=="mean":
                V = beta_rms * V + (1 - beta_rms) * ((dxt_bar**2).mean())
                
            elif use_scalar_V=="vector":
                V = beta_rms * V + (1 - beta_rms) * (dxt_bar**2)
            
            # sequence of x0 predictions
            x0_preds.append(x0_t.to('cpu'))
            
            # coefficient before model output
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
        
            if not debug:
                xt_bar_next = xt_bar + (dxt_bar / (torch.sqrt(V)+eps))
            else:
                xt_bar_next = xt_bar + dxt_bar
                
            xt_next = at_next.sqrt() * xt_bar_next
                    
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

def generalized_steps_adam(x, 
                          seq,  # tau subsequence to sample from (times)
                          model, 
                          b, # the self.betas
                          beta_rms, # the beta param for RMSProp
                          a_adam, # the beta param for RMSProp
                          eps=1e-8, # epsilon for RMSProp
                          debug = False,
                          use_V = True, # whether to use V
                          **kwargs):
    print(f"Parameters for generalized_steps_adam: beta_rms={beta_rms}, a_adam={a_adam}, eps={eps}, use_V={use_V}")
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        V = torch.ones_like(x)
        M = torch.zeros_like(x)
        if debug:
            print("debugging")
            V_list = []
            M_list = []
                    
        for i, j in zip(reversed(seq), reversed(seq_next)):
            
            t = (torch.ones(n) * i).to(x.device)
            
            next_t = (torch.ones(n) * j).to(x.device)
            
            at = compute_alpha(b, t.long())
            
            at_next = compute_alpha(b, next_t.long())
            
            xt = xs[-1].to('cuda')
            
            # define xt bar in paper
            xt_bar = xt / at.sqrt()
            
            # epsilon(x_t) 
            et = model(xt, t)
            
            # prediction of x0 at time t
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            # coefficient before xt
            # coeff_xt = torch.sqrt(at_next) / torch.sqrt(at)
            
            # sigma t in equation 12 in DDIM Song et al. 2021
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            
            # mu in the paper
            mu = torch.sqrt((1 - at_next - c1**2) / at_next) - torch.sqrt((1 - at) / at)            
            
            # dxt bar in the paper
            dxt_bar = mu * et + (c1 / torch.sqrt(at_next)) * torch.randn_like(x)
            
            # moving average of momentum
            M = a_adam * M + np.sqrt(1 - a_adam**2) * dxt_bar
            
            # update moving average
            if use_V:
                # print(dxt_bar.cpu().numpy().shape)
                V = beta_rms * V + (1 - beta_rms) * (torch.linalg.norm(dxt_bar.view(n, -1), 
                                                                       dim=-1)**2).view(n, 1, 1, 1) *\
                                                                           torch.ones_like(x)
                                                                        
                # print(((torch.linalg.norm(dxt_bar.view(n, -1), dim=-1)**2).view(n, 1, 1, 1) *\
                                                                        #    torch.ones_like(x)).shape)
            # sequence of x0 predictions
            x0_preds.append(x0_t.to('cpu'))
            
            # coefficient before model output
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        
            if use_V:
                xt_bar_next = xt_bar +  (M / (torch.sqrt(V)+eps))
            else:
                xt_bar_next = xt_bar +  M
                
            if debug:
                V_list.append(V)
                M_list.append(M)
            
            xt_next = at_next.sqrt() * xt_bar_next
            
            xs.append(xt_next.to('cpu'))
    if debug:
        return xs, x0_preds, V_list, M_list
    else:
        return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
            
    return xs, x0_preds
