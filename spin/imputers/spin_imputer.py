from typing import Type, Mapping, Callable, Optional, Union, List

import torch
from torchmetrics import Metric
from tsl.imputers import Imputer
from tsl.predictors import Predictor

from ..utils import k_hop_subgraph_sampler


class SPINImputer(Imputer):

    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = True,
                 whiten_prob: Union[float, List[float]] = 0.2,
                 n_roots_subgraph: Optional[int] = None,
                 n_hops: int = 2,
                 max_edges_subgraph: Optional[int] = 1000,
                 cut_edges_uniformly: bool = False,
                 prediction_loss_weight: float = 1.0,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(SPINImputer, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scale_target=scale_target,
                                          whiten_prob=whiten_prob,
                                          prediction_loss_weight=prediction_loss_weight,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)
        self.n_roots = n_roots_subgraph
        self.n_hops = n_hops
        self.max_edges_subgraph = max_edges_subgraph
        self.cut_edges_uniformly = cut_edges_uniformly

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.training and self.n_roots is not None:
            batch = k_hop_subgraph_sampler(batch, self.n_hops, self.n_roots,
                                           max_edges=self.max_edges_subgraph,
                                           cut_edges_uniformly=self.cut_edges_uniformly)
        return super(SPINImputer, self).on_after_batch_transfer(batch,
                                                                dataloader_idx)

    def training_step(self, batch, batch_idx):
        injected_missing = (batch.original_mask - batch.mask)
        if 'target_nodes' in batch:
            injected_missing = injected_missing[..., batch.target_nodes, :]
        # batch.input.target_mask = injected_missing
        y_hat, y, loss = self.shared_step(batch, mask=injected_missing)

        # Logging
        self.train_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        if 'target_nodes' in batch:
            torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        # batch.input.target_mask = batch.eval_mask
        y_hat, y, val_loss = self.shared_step(batch, batch.eval_mask)

        # Logging
        self.val_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # batch.input.target_mask = batch.eval_mask
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        y, eval_mask = batch.y, batch.eval_mask
        test_loss = self.loss_fn(y_hat, y, eval_mask)

        # Logging
        self.test_metrics.update(y_hat.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument('--whiten-prob', type=float, default=0.05)
        parser.add_argument('--prediction-loss-weight', type=float, default=1.0)
        parser.add_argument('--n-roots-subgraph', type=int, default=None)
        parser.add_argument('--n-hops', type=int, default=2)
        parser.add_argument('--max-edges-subgraph', type=int, default=1000)
        parser.add_argument('--cut-edges-uniformly', type=bool, default=False)
        return parser
