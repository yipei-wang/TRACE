import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from utils import masking, func, nC2_swap, kendall_distance

IMAGE_SIZE=224



def greedy(model, image, label, options, mode = 'MoRF'):
    '''
    Parameters:
    - model (torch.nn.Module): the model to be explained
    - image (torch.Tensor): 3 x IMAGE_SIZE x IMAGE_SIZE
    - label (int): the label of the image
    - mode  (str): MoRF (Most Relevant First) or LeRF (Least Relevant First) 
    
    Returns:
    - traj: (list): a list of the deletion trajectoy
    - Prob: (list): the predicted probability of the corresponding class w.r.t. the deletion process
    '''  
    
    channel = image.shape[-3]
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            raise ValueError("only 1 image at a time")

            
    grid = options.edge**2
    patch_size = IMAGE_SIZE//options.edge
    
    all_idx = set(range(grid))
    traj = []
    Prob = []
    model.eval()
    with torch.no_grad():
        new_image = image.detach().clone()
        pred = model(new_image[None])[:, label].item()
        prob = torch.softmax(model(new_image[None]), dim = 1)[:, label].item()
        Prob.append(prob)
        while len(all_idx) > 1: 
            masked = []
            ids = []
            for i in all_idx:
                masked.append(masking(new_image, options.reference, i, edge = options.edge))
                ids.append(i)
            masked = torch.stack(masked)
            prob = torch.softmax(model(masked), dim = 1)[:, label]
            
            if mode == 'Le':
                argmax = prob.argmax().item()
                traj.append(ids[argmax])
                all_idx.remove(traj[-1])
                Prob.append(prob.max().item())
                new_image = masked[prob.argmax().item()].detach().clone()
            elif mode == 'Mo':
                traj.append(ids[prob.argmin().item()])
                all_idx.remove(traj[-1])
                Prob.append(prob.min().item())
                new_image = masked[prob.argmin().item()].detach().clone()
            else:
                raise ValueError("mode has to be either 'Mo' or 'Le'")  
             
                
#         masked = torch.zeros(1,channel,IMAGE_SIZE,IMAGE_SIZE).to(image.device)
        if options.reference == 'zero':
            masked = torch.zeros(1,channel,IMAGE_SIZE,IMAGE_SIZE).to(image.device)
        elif options.reference == 'mean':
            masked = torch.ones(1,channel,IMAGE_SIZE,IMAGE_SIZE).to(image.device)*image.mean()
        elif options.reference == 'patch_mean':
            masked = F.interpolate(
                F.interpolate(image.view(1,3,IMAGE_SIZE,IMAGE_SIZE), size=(7,7), mode='area'),
                size=(IMAGE_SIZE,IMAGE_SIZE), mode='area')
        elif options.reference == 'blur':
            blurrer = transforms.GaussianBlur(kernel_size=(31, 31), sigma=(56, 56))
            masked = blurrer(image)[None]
            
        
        prob = torch.softmax(model(masked), dim = 1)[:, label].item()
        Prob.append(prob)
        traj.append(all_idx.pop())
        
    return np.array(traj), np.array(Prob)



def simulated_annealing(
    options,
    model,
    image,
    label,
    traj_init, 
    candidates,
    T = 0.05, 
    eta = 0.998, 
    threshold = 0., 
    verbose = False
):
    
    # Initialization
    sol_0 = np.array(traj_init).copy()
    ct_array = []
    best = sol_0.copy()

    score_best = -func(model, image, label, best, options)
    score_0 = -func(model, image, label, sol_0, options)

    print('start: ', score_best)
    for t in range(1, options.n_iteration + 1):
        temp = sol_0.copy()
        sol_1 = nC2_swap(sol_0, candidates)

        # energy
        score_1 = -func(model, image, label, sol_1, options)
        delta_t = score_1 - score_0
        
        if delta_t < 0:
            sol_0 = sol_1.copy()
            score_0 = score_1
            ct_array.append(1)
        else:
            p = np.exp(-delta_t/T)
            if np.random.uniform(0, 1) < p:
                sol_0 = sol_1.copy()
                score_0 = score_1
                ct_array.append(1)
            else:
                ct_array.append(0)

        if score_best > score_0:
            best = sol_0.copy()
            score_best = score_0
            print('t: {}, score: {:.4f}, T: {:.4f} kendall onestep: {:.4f}, kendall overall: {:.4f}'.format(
                t, score_best, T, kendall_distance(temp, sol_1),
                kendall_distance(sol_0.argsort(), np.array(traj_init).argsort())
            ))
        elif ct_array[-1] == 1 and verbose:
            print('t: {}, score: {:.4f}/{:.4f}, T: {:.4f} kendall onestep: {:.4f}, kendall overall: {:.4f}'.format(
                t, score_0, score_best, T, kendall_distance(temp, sol_1),
                kendall_distance(sol_0.argsort(), np.array(traj_init).argsort())
            ))

        if T > threshold:
            T *= eta

    return best