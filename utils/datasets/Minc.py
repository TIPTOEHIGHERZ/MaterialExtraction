from .DTD import DTDLoader


class MincLoader(DTDLoader):
    def __init__(self, fp, batch_size=32, relative_dir=None, device='cpu'):
        super().__init__(fp, batch_size, relative_dir, device)

        return
