
from CNN_chocolate import ConvNet
from _modules.cleanEEGNet import cleanEEGNet
import _modules.params as p
import torch
from torch.autograd import Variable
from torchinfo import summary




light_mod = cleanEEGNet().load_from_checkpoint("/home/mbrugnara/cleanEEGNet/best model/models-epoch=15-valid_loss=0.00-v1.ckpt").to(p.device)
light_mod.eval()
light_mod.freeze()

torch_model = light_mod.model
state_dict = torch_model.state_dict()

torch.save(state_dict,"/home/mbrugnara/cleanEEGNet/best model/bcdnet.pth")

dummy_input = Variable(torch.randn(1,1,62,2560))

trained_model = ConvNet()
summary(trained_model, input_size=(1, 1, 62, 2560))


trained_model.load_state_dict(torch.load("/home/mbrugnara/cleanEEGNet/best model/bcdnet.pth"))

trained_model.eval()
torch.onnx.export(trained_model, dummy_input, "/home/mbrugnara/cleanEEGNet/best model/bcdnet.onnx")

