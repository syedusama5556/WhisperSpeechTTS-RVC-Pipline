from torchvision.models import vgg16

from dlas.trainer.networks import register_model
from dlas.utils.util import opt_get


@register_model
def register_torch_vgg16(opt_net, opt):
    """ return a ResNet 18 object
    """
    return vgg16(**opt_get(opt_net, ['kwargs'], {}))
