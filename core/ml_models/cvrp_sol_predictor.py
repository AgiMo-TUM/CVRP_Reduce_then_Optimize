"""Wrapper for solution-arc prediction models (imitation learning / classification)."""

from functools import partial

import torch

from core.ml_models.base_learner import BaseLearner
from core.ml_models.gnn import GraphNNAtt
from core.ml_models.losses import loss_arcs_multiclass
from core.utils.kpi import eval_arc_prediction_accuracy
from core.utils.kpi import get_accuracy


class BaseSolArcPredictor(BaseLearner):
    """Base learner for solution-arc prediction models.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model to be trained.
    class_weight : float or list, optional
        Class weights to be used.
    adam_params : dict, optional
        Dictionary of Adam parameters.
    lr_schedule : dict, optional
        Dictionary of learning-rate scheduler parameters.
    input_transformer : object, optional
        Transformation applied to ``data.x`` and ``data.edge_attr`` before
        the forward pass (e.g. a fitted standardiser).
    """

    def __init__(
        self,
        model,
        class_weight=None,
        adam_params=None,
        lr_schedule=None,
        input_transformer=None,
    ):
        super(BaseSolArcPredictor, self).__init__(model, adam_params, lr_schedule)

        self.multi_class = self.model.output_dim > 1
        self.class_weight = class_weight

        if not self.multi_class:
            if self.class_weight is not None:
                self.class_weight = torch.FloatTensor([self.class_weight])
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weight)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.loss_fn = self.loss_fn.to(device)
            self.evaluate = self.evaluate_binary
        else:
            if self.class_weight is not None:
                self.class_weight = torch.FloatTensor(self.class_weight)
            self.loss_fn = partial(loss_arcs_multiclass, arc_cw=self.class_weight)
            self.evaluate = self.evaluate_multiclass

        self.input_transformer = input_transformer

    def forward_pass(self, data):
        """Run a forward pass and compute the classification loss."""
        arc_predictions_raw, arc_predictions = self.predict_arcs(data)

        true_arc_list = (
            data.y.float() if not self.multi_class else data.y
        ).to(arc_predictions_raw.device)

        loss = self.loss_fn(arc_predictions_raw, true_arc_list)
        return loss, arc_predictions, loss

    def predict_arcs(self, data, train=True):
        """Run the underlying model and return raw + post-processed predictions.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Graph(s) with fields ``x``, ``edge_index``, ``edge_attr``,
            and (optionally) ``y``.
        train : bool, optional
            If False, set the model to eval mode before prediction.

        Returns
        -------
        predictions_raw : torch.Tensor
            Raw model outputs (logits).
        predictions : torch.Tensor
            Sigmoid (binary) or log-softmax (multi-class) of the logits.
        """
        if not train:
            self.model.eval()

        if self.input_transformer is not None:
            x_norm, edge_attr_norm = self.input_transformer.transform(
                [data.x, data.edge_attr]
            )
            data.x, data.edge_attr = x_norm, edge_attr_norm

        predictions_raw = self.model(data.x, data.edge_attr, data.edge_index)

        if self.multi_class:
            predictions = torch.nn.functional.log_softmax(predictions_raw, dim=-1)
        else:
            predictions = torch.sigmoid(predictions_raw)

        return predictions_raw, predictions

    def evaluate_binary(self, data_loaders):
        """Evaluate binary-classification performance over the given loaders.

        Parameters
        ----------
        data_loaders : dict {name: DataLoader}.

        Returns
        -------
        Tuple of (per-loader-metrics dict, last loss, accuracy, recall,
        precision, fscore).
        """
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        p = dict()

        running_loss = running_acc = running_rec = running_prec = running_f = 0
        n = 0

        for loader_name, data_loader in data_loaders.items():
            n = len(data_loader)
            running_loss = running_acc = running_rec = running_prec = running_f = 0

            for batch in data_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        loss, outputs, total_loss = self.forward_pass(batch)

                threshold = 0.5
                predictions = (outputs > threshold).int().detach().cpu().numpy()

                accuracy, recall, precision, fscore = eval_arc_prediction_accuracy(
                    predictions, batch.y.cpu().numpy()
                )
                running_loss += total_loss.item()
                running_acc += accuracy
                running_rec += recall
                running_prec += precision
                running_f += fscore
                torch.cuda.empty_cache()

            p[f"{loader_name}_loss"] = running_loss / n
            p[f"{loader_name}_accuracy"] = running_acc / n
            p[f"{loader_name}_recall"] = running_rec / n
            p[f"{loader_name}_precision"] = running_prec / n
            p[f"{loader_name}_fscore"] = running_f / n

        return (
            p,
            running_loss / n,
            running_acc / n,
            running_rec / n,
            running_prec / n,
            running_f / n,
        )

    def evaluate_multiclass(self, data_loaders):
        """Evaluate multi-class classification performance."""
        self.model.eval()
        p = dict()
        for loader_name, data_loader in data_loaders.items():
            n = len(data_loader)
            running_loss = 0
            running_acc = 0
            for batch in data_loader:
                loss, outputs, _ = self.forward_pass(batch)
                loss = loss.item()
                _, predictions = outputs.max(-1)
                predictions = predictions.cpu().numpy()
                accuracy = get_accuracy(predictions, batch.y.cpu().numpy())
                running_loss += loss
                running_acc += accuracy
            p[f"{loader_name}_loss"] = running_loss / n
            p[f"{loader_name}_accuracy"] = running_acc / n
        return p


class GCNNSolArcPredictor(BaseSolArcPredictor):
    """GNN-based solution-arc predictor."""

    def __init__(self, model_config, directed=True, **kwargs):
        input_dims = (model_config.node_dim, model_config.arc_dim)
        hidden_dim = model_config.hidden_layer_dim
        conv_hidden_dim = model_config.get("conv_hidden_layer_dim", hidden_dim)
        num_conv_layers = model_config.num_conv_layers
        num_dense_layers = model_config.num_dense_layers

        conv_dims = [(conv_hidden_dim, conv_hidden_dim) for _ in range(num_conv_layers)]
        dense_dims = [hidden_dim for _ in range(num_dense_layers)]
        output_dim = model_config.arc_output_dim

        model = GraphNNAtt(
            input_dims,
            conv_dims,
            dense_dims,
            output_dim,
            directed=directed,
        )

        super(GCNNSolArcPredictor, self).__init__(model, **kwargs)
