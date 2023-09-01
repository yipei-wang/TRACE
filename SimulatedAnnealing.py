import argparse
parser = argparse.ArgumentParser(description="Template")


parser.add_argument("-gpu", "--GPU_index", default=0, type=int, help="gpu index")
parser.add_argument("-e", "--edge", default=7, type=int, help="image_size/patch_size")
parser.add_argument("-path", "--data_path", default='ILSVRC2012_subset', type=str, help="the path of the input image")
parser.add_argument("-idx", "--image_index", default=0, type=int, help="the index of the image")
parser.add_argument("-model", "--model_name", default='resnet18', type=str, help="the name of the model to be explained")
parser.add_argument("-ref", "--reference", default='zero', type=str, help="the type of reference values")
parser.add_argument("-n", "--n_iteration", default=2000, type=str, help="the number of iterations")
parser.add_argument("-m", "--mode", default='Le-Mo',type=str, help="the mode of the optimization")
parser.add_argument("--verbose", default=False,action='store_true')
parser.add_argument("--compare",default=False,action='store_true')
parser.add_argument("-l", "--layer_name", default='layer4', type=str, help="the layer to be explained by GradCAM when compare==True")


options = parser.parse_args()


import os
import torch
from dataset import load_data
from utils import *
from TRACE import greedy, simulated_annealing
import itertools


torch.manual_seed(0)
device = torch.device(f'cuda:{options.GPU_index}')
IMAGE_SIZE = 224
grid = options.edge**2
candidates = list(itertools.combinations(range(grid), 2))
nC2 = grid*(grid-1)//2


# Load the model if the model actually exists in torchvision.models
model_func = getattr(torchvision.models, options.model_name, None)
try:
    model = model_func(pretrained=True).to(device)
    model.eval()
except:
    raise ValueError(f"Model {options.model_name} is not available in torchvision.models")
get_n_params(model)


image, label = load_data(options)
image = image.to(device)
print(f'Loading the {options.image_index+1}th image.\t The image belongs to class {label}')
with torch.no_grad():
    prob = torch.softmax(model(image.unsqueeze(0)),dim=1)
print('Predicted by %s as class %d with confidence %.3f'%(options.model_name, prob.argmax(1).item(), prob.max().item()))
plot_tensor_image(image)


## Use TRACE_Greedy as the initializations
try:
    with open('results/log_greedy_image%d_edge%d_%s_%s.txt'%(options.image_index,options.edge,options.model_name,options.reference), 'r') as f:
        log = eval(f.read())
    traj_init = np.array(log['traj_le'])
    print("Loading the initial trajectory from logs")
except:
    print("Generating the initial trajectory")
    traj_init = greedy(model, image, label, options, 'Le')[0]
    
    
## Generate trajectories using simulated annealing
traj = simulated_annealing( 
    options,
    model,
    image,
    label,
    traj_init, 
    candidates,
    verbose = options.verbose,
)            
log = {
    'image': options.image_index,
    'edge': options.edge,
    'mode': options.mode,
    'model': options.model_name,
    'reference': options.reference,
    'traj': list(traj),
}
# To save the log
with open('results/log_SA_image%d_edge%d_%s_%s_%s.txt'%(options.image_index, options.edge, options.mode, options.model_name, options.reference), 'w') as f:
    f.write(str(log))
    
prob_le = traj_masking(model, image, label, traj, reference = options.reference, edge = options.edge)[0]
prob_mo = traj_masking(model, image, label, np.flip(traj), reference = options.reference, edge = options.edge)[0]

show_traj(image, traj, options, save='results/TRACE-SA_image%d_edge%d_%s_%s_deletion_process.pdf'%(options.image_index+1, options.edge, options.model_name, options.reference))


# Plot the deletion metric results for the single image

if not options.compare:

    plt.figure()
    plt.plot(prob_le, label='TRACE-Greedy-Le')
    plt.plot(prob_mo, label='TRACE-Greedy-Mo')
    plt.legend()
    plt.grid('grey', linestyle='dashed')
    plt.title('The MoRF/LeRF deletion test of TRACE-Greedy on image %d'%(options.image_index+1))
    plt.xlabel('# of Patches Deleted')
    plt.ylabel('Predicted Probability')
    plt.savefig('results/TRACE-Greedy_test_image%d_edge%d_%s.pdf'%(options.image_index+1,options.edge,options.model_name), bbox_inches='tight')
    plt.show()
    
else:
    
    from torchray.attribution.grad_cam import grad_cam
    from torchray.attribution.gradient import gradient
    from torchray.attribution.excitation_backprop import excitation_backprop
    from captum import attr
    
    IG = attr.IntegratedGradients(model)
    IxG = attr.InputXGradient(model)
    
    input=image.unsqueeze(0)
    saliency_gradient = torch.abs(F.interpolate(gradient(model, input, label), size = (IMAGE_SIZE, IMAGE_SIZE), 
                                      mode = 'area')).squeeze().detach().cpu().numpy()
    saliency_ig = torch.abs(IG.attribute(input, target = label)).mean(1).squeeze().detach().cpu().numpy()
    saliency_ixg = torch.abs(IxG.attribute(input, target = label)).mean(1).squeeze().detach().cpu().numpy()
    
    
    ig_le = deletion_attribution(model,image,label,saliency_ig,options,'LeRF')
    ig_mo = deletion_attribution(model,image,label,saliency_ig,options,'MoRF')
    ixg_le = deletion_attribution(model,image,label,saliency_ixg,options,'LeRF')
    ixg_mo = deletion_attribution(model,image,label,saliency_ixg,options,'MoRF')
    gradient_le = deletion_attribution(model,image,label,saliency_gradient,options,'LeRF')
    gradient_mo = deletion_attribution(model,image,label,saliency_gradient,options,'MoRF')


    fig, ax=plt.subplots(1,2,figsize=(12,5))

    
    
    ax[0].plot(prob_le, 'r-', label='TRACE-Greedy-Le')
    ax[0].plot(ig_le, 'g-', label='IG')
    ax[0].plot(ixg_le, 'b-', label='IxG')
    ax[0].plot(gradient_le, 'y-', label='Gradient')
    ax[0].set_title('LeRF Deletion')
    ax[0].grid('grey', linestyle='dashed')
    ax[0].set_xlabel('# of Patches Deleted')
    ax[0].set_ylabel('Predicted Probability')

    
        
    try:
        saliency_gradcam = F.interpolate(grad_cam(model, input, label, saliency_layer = options.layer_name),size = (IMAGE_SIZE, IMAGE_SIZE), 
                                         mode = 'bilinear', align_corners = True).squeeze().detach().cpu().numpy()
        saliency_ebp = F.interpolate(excitation_backprop(model, input, label, saliency_layer = options.layer_name), size = (IMAGE_SIZE, IMAGE_SIZE), 
                                     mode = 'bilinear', align_corners = True).squeeze().detach().cpu().numpy()
        gradcam_le = deletion_attribution(model,image,label,saliency_gradcam,options,'LeRF')
        gradcam_mo = deletion_attribution(model,image,label,saliency_gradcam,options,'MoRF')
        ebp_le = deletion_attribution(model,image,label,saliency_ebp,options,'LeRF')
        ebp_mo = deletion_attribution(model,image,label,saliency_ebp,options,'MoRF')
        ax[0].plot(gradcam_le, 'k-', label='GradCAM')
        ax[0].plot(ebp_le, 'c-', label='EBP')
        ax[1].plot(gradcam_mo, 'k-', label='GradCAM')
        ax[1].plot(ebp_mo, 'c-', label='EBP')
        
    except:
        print("--layer incorrect! Skipping GradCAM and Excitation BP!")

    ax[1].plot(prob_mo, 'r-', label='TRACE-Greedy-Mo')
    ax[1].plot(ig_mo, 'g-', label='IG')
    ax[1].plot(ixg_mo, 'b-', label='IxG')
    ax[1].plot(gradient_mo, 'y-', label='Gradient')
    ax[1].set_title('MoRF Deletion')
    ax[1].grid('grey', linestyle='dashed')
    ax[1].set_xlabel('# of Patches Deleted')
    ax[1].set_ylabel('Predicted Probability')

    plt.legend()
    plt.suptitle('The MoRF/LeRF deletion test of TRACE-SA on image %d'%(options.image_index+1))
    plt.savefig('results/TRACE-SA_comparison_image%d_edge%d_%s_%s.pdf'%(options.image_index+1,options.edge,options.model_name,options.reference), bbox_inches='tight')
    plt.show()