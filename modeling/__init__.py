from modeling.model import ft_net

def build_model(num_classes):
    model = ft_net(num_classes)
    return model