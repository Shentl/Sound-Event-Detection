from .models_4conv import Crnn_4conv
from .try1 import Try1_4conv  # 4 conv, GRU num_layers
from .try2 import Try2_4conv  # 4 conv, GRU hidden_size
from .try3 import Try3_4conv  # 4 conv, square kernel_size
from .try4 import Try4_4conv  # stack at the freq/time_dim, time_dim is better
from .try5 import Try5_4conv  # 4 conv, flx kernel_size, GRU num_layers=2, better now 13...
from .try6 import Try6_4conv  # 4 conv, flx kernel_size, GRU num_layers=2, stack at the freq/time_dim
from .try7 import Try7_4conv  # for n_mels = 256
from .models_baseline import Crnn


MODELS = {
    "4conv": Crnn_4conv,
    "try1": Try1_4conv,
    "try2": Try2_4conv,
    "try3": Try3_4conv,
    "try4": Try4_4conv,
    "try5": Try5_4conv,
    "try6": Try6_4conv,
    "try7": Try7_4conv,
    'models_baseline': Crnn
}


def load_model(name="mid"):
    assert name in MODELS.keys(), f"Model name can only be one of {MODELS.keys()}."
    print(f"Using model: '{name}'")
    return MODELS[name]


'''
model_cfg = config['model']
model = load_model(config['model_name'])(**model_cfg)
print(model)
'''
