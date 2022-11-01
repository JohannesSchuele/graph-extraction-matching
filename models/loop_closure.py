
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
#from utils import plot_camera_scene

from models.loop_closure_model import LoopClosureModel
from utils.plot_fcts_3d import plot_camera_scene
class LoopClosureWorldMap(nn.Module):

    def __init__(self, inverse_renderer, world_map, device, print_loss=False):
        super(LoopClosureWorldMap, self).__init__()
        self.world_map = world_map
        self.device = device
        self.print_loss = print_loss
        self._model_loop_closure =None
        self._optimizer = None
        self.inverse_renderer = inverse_renderer


    def forward_optimize(self, weight_normal_loss =0.1, lr = 0.25, iterations = 100):
        self._set_optimization_problem(weight_normal_loss=weight_normal_loss)
        self._set_optimizer(lr=lr)


        loop = tqdm(range(iterations))
        for i in loop:
            print('loop: ', i)
            with torch.autograd.set_detect_anomaly(True):
                self._optimizer.zero_grad()
                loss, cameras = self._model_loop_closure.forward()  # ToDo implement a stop criterium if loss is to low, else it would cause problems for the differentiation!!!
                loss.backward(retain_graph =False)  # ToDo: check retain_graph here! and probably just do it in the first call!
                print('distances of cameras', cameras.T)
                # ToDo check if retain_graoh in the first call is sufficient, and makes everything more efficient
            self._optimizer.step()

            if i ==0:
                cameras_init = cameras.clone()
            # plot and print status message
            if i % 5 == 0 or i == iterations - 1:
                status = 'iteration=%3d; camera_distance=1.3e' % (i)
                plot_camera_scene(cameras_init, cameras, status)

            optimized_world_map = self._model_loop_closure.update_world_map_with_latest_fragments_and_camera_positions()
            self.world_map = optimized_world_map

        return optimized_world_map




    def forward(self):
        return True


    def _set_optimizer(self, lr = 0.25):   # ToDo implement if statements to choose the solver
        self._optimizer = torch.optim.Adam(self._model_loop_closure.parameters(), lr=lr)
        parameters_trainable, parameters_all = self.count_parameters(self._model_loop_closure)
        print('The loop closure model, contains ', parameters_trainable, 'trainable parameters and in total ', parameters_all, 'parameters!')
        # import torch
        # from torchvision import models
        # from torchsummary import summary
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # vgg = models.vgg16().to(device)
        # summary(vgg, (3, 224, 224))


    def _set_optimization_problem(self, weight_normal_loss =0.1):
        self._model_loop_closure = LoopClosureModel(world_map=self.world_map,
                                                    inverse_renderer=self.inverse_renderer,
                                                    meshes_world =self.world_map.mesh_model,
                                                    weight_normal_loss= weight_normal_loss,
                                                    print_loss=self.print_loss,
                                                    device=self.device
                                                    )

    @staticmethod
    def count_parameters(model):
        parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameters_all = pytorch_total_params = sum(p.numel() for p in model.parameters())
        return parameters_trainable, parameters_all

