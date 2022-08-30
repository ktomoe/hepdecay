import torch
import torch.nn as nn

from multiml import logger, const
from multiml.task.pytorch import PytorchBaseTask


class MyGraphTask(PytorchBaseTask):
    def set_hps(self, params):
        super().set_hps(params)

        torch.backends.cudnn.benchmark = True

        self._storegate.to_memory('features', phase='all')
        self._storegate.to_memory('label_graphs', phase='all')
        self._storegate.to_memory('label_edges', phase='all')
        self._storegate.set_mode('numpy')

        # loss
        if '2hdm' in self._data_id:
            self._loss = [
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6, 7.2, 9, 7.2, 3, 9])),
                nn.CrossEntropyLoss(),
            ]
        elif 'zprime' in self._data_id:
            self._loss = [
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6, 9, 4.5, 18, 4.5, 4.5])),
                nn.CrossEntropyLoss(),
            ]

    @logger.logging
    def execute(self):
        """ Execute a task.
        """
        self.compile()

        model_feature = self._model_args['features']

        dataloaders = self.prepare_dataloaders()
        result = self.fit(dataloaders=dataloaders, dump=True)

        pred = self.predict(dataloader=dataloaders['test'])
        self.update(data=pred, phase='test')

        suffix = f'_{self._model_args["features"]}_{self._loss_weights[0]}-{self._loss_weights[1]}'

        self._storegate.set_mode('zarr')
        self._storegate.to_storage('pred_graphs', output_var_names='pred_graphs'+suffix, phase='test')
        self._storegate.to_storage('pred_edges', output_var_names='pred_edges'+suffix, phase='test')
        self._storegate.to_storage('attn_graphs', output_var_names='attn_graphs'+suffix, phase='test')
        self._storegate.to_storage('attn_edges', output_var_names='attn_edges'+suffix, phase='test')
        self._storegate.set_mode('numpy')

class MyMLPTask(PytorchBaseTask):
    def set_hps(self, params):
        super().set_hps(params)

        torch.backends.cudnn.benchmark = True

        self._storegate.to_memory('features', phase='all')
        self._storegate.to_memory('label_graphs', phase='all')
        self._storegate.to_memory('label_edges', phase='all')
        self._storegate.set_mode('numpy')

        # inputs
        if 'low' in self._data_id:
            self._model_args['inputs'] = 30

        elif 'high' in self._data_id:
            self._model_args['inputs'] = 34
