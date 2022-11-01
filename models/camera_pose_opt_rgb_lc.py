
import torch
from tqdm.notebook import tqdm
import torch.nn as nn
#from utils import plot_camera_scene

from models.loop_closure_model import LoopClosureModel
from utils.plot_fcts_3d import plot_camera_scene
from models.camera_pose_opt_rgb_lc_model import CameraPoseModel_RGB
class CameraPose_RGB(nn.Module):

    def __init__(self, renderer, seen_images, losses=None, device='cpu', print_loss=False):
        super().__init__()
        self.seen_images = seen_images
        self.device = device
        self.print_loss = print_loss
        self._model_loop_closure =None
        self._optimizer = None
        self.renderer = renderer
        if losses is not None:
            self.losses = losses
        else:
            self.losses = {
                "rgb": {"weight": 0.5, "values": []}}


    def forward(self, mesh_model_vertex_rgb, last_images_2bused=None, lr = 0.25, iterations = 100):
        self._set_optimization_problem(mesh_model_vertex_rgb=mesh_model_vertex_rgb, last_images_2bused= last_images_2bused)
        self._set_optimizer(lr=lr)


        loop = tqdm(range(iterations))
        for i in loop:
            with torch.autograd.set_detect_anomaly(True):
                self._optimizer.zero_grad()
                loss, cameras = self.CameraPoseModel_RGB.forward(iteration= i)  # ToDo implement a stop criterium if loss is to low, else it would cause problems for the differentiation!!!
                loss.backward(retain_graph =False)  # ToDo: check retain_graph here! and probably just do it in the first call!
                # ToDo check if retain_graoh in the first call is sufficient, and makes everything more efficient
            self._optimizer.step()

            if i == 0:
                cameras_init = cameras.clone()
            # plot and print status message
            if i % 5 == 0 or i == iterations - 1:
                status = 'iteration=%3d; camera_distance=1.3e' % (i)
                plot_camera_scene(cameras_init, cameras, status)

            self.seen_images, r_rotation_batch, t_translation_batch = self.update_camp_positions_in_seen_images()


        return self.seen_images, r_rotation_batch, t_translation_batch



    def update_camp_positions_in_seen_images(self):

        r_rotation_batch, t_translation_batch = self._model_loop_closure.get_rotation_translation_parameters
        r_rotation_batch.requires_grad = False
        t_translation_batch.requires_grad = False
        last_images_2bused = r_rotation_batch.shape[0]
        for i, graph_image_s in enumerate(self.seen_images[: last_images_2bused]):
            graph_image = graph_image_s[2]
            graph_image.update_cam_position(r_rotation_batch[i], t_translation_batch[i])

            self.seen_images[i][2] = graph_image

        return self.seen_images, r_rotation_batch, t_translation_batch


    def _set_optimizer(self, lr = 0.25):   # ToDo implement if statements to choose the solver
        self._optimizer = torch.optim.Adam(self.CameraPoseModel_RGB.parameters(), lr=lr)
        parameters_trainable, parameters_all = self.count_parameters(self.CameraPoseModel_RGB)
        print('The loop closure model, contains ', parameters_trainable, 'trainable parameters and in total ', parameters_all, 'parameters!')


    def _set_optimization_problem(self, mesh_model_vertex_rgb, last_images_2bused=None):

        self.CameraPoseModel_RGB =  CameraPoseModel_RGB(seen_images=self.seen_images,
                                                    renderer=self.renderer,
                                                    mesh_model_vertex_rgb = mesh_model_vertex_rgb,
                                                    losses=self.losses,
                                                    last_images_2bused = last_images_2bused,
                                                    device=self.device
                                                    )

    @ property
    def get_last_images_applied(self):
        return self.CameraPoseModel_RGB.last_images_applied


    @staticmethod
    def count_parameters(model):
        parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        parameters_all = pytorch_total_params = sum(p.numel() for p in model.parameters())
        return parameters_trainable, parameters_all

