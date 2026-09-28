"""PyTorch implementation of an attention-based GNN for arc prediction.

The forward pass has two flavours, selected by the ``directed`` flag passed
into :class:`GraphNNAtt` (and propagated to each :class:`GraphLayerAtt`):

- ``directed=True``:  message passing aggregates only at the destination
  endpoint of each arc (asymmetric).
- ``directed=False``: message passing aggregates at both endpoints
  (symmetric / undirected). Use this when the underlying graph is undirected.

The flag is read from ``configs/training/config.yaml`` (key ``directed``) by
the caller and forwarded into the model constructor.
"""

import torch
import torch.nn.functional as F


class GraphLayerAtt(torch.nn.Module):
    """Graph convolutional layer with attention.

    Parameters
    ----------
    dims_in : tuple
        2-element tuple containing input dimension for nodes and arcs.
    dims_out : tuple
        2-element tuple containing output dimension for nodes and arcs.
    directed : bool, optional
        Selects the forward implementation. Default False.
    weight_init_func : callable, optional
        Weight initialisation function.
    """

    def __init__(self, dims_in, dims_out, directed=False, weight_init_func=None):
        super(GraphLayerAtt, self).__init__()

        in_v, in_a = dims_in
        out_v, out_a = dims_out

        self.directed = directed

        self.dense_vv = torch.nn.Linear(in_v, out_v)
        self.dense_va = torch.nn.Linear(in_a, out_v, bias=False)
        self.attention_va = torch.nn.Linear(in_a, 1)

        self.dense_aa = torch.nn.Linear(in_a, out_a)
        self.dense_av = torch.nn.Linear(in_v, out_a, bias=False)

        if weight_init_func is not None:
            weight_init_func(self.dense_vv.weight)
            weight_init_func(self.dense_va.weight)
            weight_init_func(self.dense_aa.weight)
            weight_init_func(self.dense_av.weight)

    def forward(self, x_v, x_a, arc_index):
        if self.directed:
            return self._forward_directed(x_v, x_a, arc_index)
        return self._forward_undirected(x_v, x_a, arc_index)

    def _forward_directed(self, x_v, x_a, arc_index):
        """Asymmetric message passing: aggregate only at destination nodes."""
        if x_a.dim() == 3:
            x_a = x_a.squeeze(-1)
        if x_v.dim() == 3:
            x_v = x_v.squeeze(-1)
        x_a = x_a.to(dtype=self.attention_va.weight.dtype)

        src, dst = arc_index
        h_self = self.dense_vv(x_v)
        node_dtype = h_self.dtype
        device = h_self.device
        num_nodes = x_v.size(0)

        e = self.attention_va(x_a).squeeze(-1)

        max_per_dst = torch.full(
            (num_nodes,), float("-inf"), device=device, dtype=e.dtype
        )
        max_per_dst = max_per_dst.index_reduce(0, dst, e, reduce="amax")

        exp_e = torch.exp(e - max_per_dst[dst])
        denom = torch.zeros(
            num_nodes, device=device, dtype=exp_e.dtype
        ).index_add(0, dst, exp_e)

        attn = (exp_e / denom[dst]).unsqueeze(-1)
        dense_xa = self.dense_va(x_a)
        msg_from_arcs = (dense_xa * attn).to(node_dtype)

        h_msg = torch.zeros_like(h_self).index_add(0, dst, msg_from_arcs)
        h_v = h_msg + h_self

        h_a = (
            self.dense_aa(x_a)
            + self.dense_av(x_v[src])
            + self.dense_av(x_v[dst])
        )
        return h_v, h_a

    def _forward_undirected(self, x_v, x_a, arc_index):
        """Symmetric message passing: aggregate at both source and destination."""
        if x_a.dim() == 3:
            x_a = x_a.squeeze(-1)
        if x_v.dim() == 3:
            x_v = x_v.squeeze(-1)

        src, dst = arc_index
        num_nodes = x_v.size(0)
        x_a = x_a.to(dtype=self.attention_va.weight.dtype)

        h_self = self.dense_vv(x_v)
        node_dtype = h_self.dtype
        device = h_self.device

        e = self.attention_va(x_a).squeeze(-1)

        max_per_dst = torch.full(
            (num_nodes,), float("-inf"), device=device, dtype=e.dtype
        )
        max_per_dst = max_per_dst.index_reduce(0, dst, e, reduce="amax")

        max_per_src = torch.full(
            (num_nodes,), float("-inf"), device=device, dtype=e.dtype
        )
        max_per_src = max_per_src.index_reduce(0, src, e, reduce="amax")

        exp_e = torch.exp(e - max_per_dst[dst])
        exp_e_src = torch.exp(e - max_per_src[src])

        denom = torch.zeros(
            num_nodes, device=device, dtype=exp_e.dtype
        ).index_add(0, dst, exp_e)
        denom_src = torch.zeros(
            num_nodes, device=device, dtype=exp_e_src.dtype
        ).index_add(0, src, exp_e_src)

        attn = (exp_e / denom[dst]).unsqueeze(-1)
        attn_src = (exp_e_src / denom_src[src]).unsqueeze(-1)

        dense_xa = self.dense_va(x_a)
        msg_from_arcs = (dense_xa * attn).to(node_dtype)
        msg_from_arcs_src = (dense_xa * attn_src).to(node_dtype)

        h_msg = torch.zeros_like(h_self).index_add(0, dst, msg_from_arcs)
        h_msg_src = torch.zeros_like(h_self).index_add(0, src, msg_from_arcs_src)

        h_v = h_msg + h_self + h_msg_src

        h_a = (
            self.dense_aa(x_a)
            + self.dense_av(x_v[src])
            + self.dense_av(x_v[dst])
        )
        return h_v, h_a


class GraphNNAtt(torch.nn.Module):
    """Attention GNN that predicts an output value per arc.

    Parameters
    ----------
    dims_in : tuple
        2-element tuple of input dimensions for nodes and arcs.
    conv_dims : list of tuple
        Output dimensions of each graph convolutional layer.
    dense_dims : list of int
        Output dimensions of each post-conv dense layer.
    dim_out : int
        Arc output dimension.
    directed : bool, optional
        If True, use the asymmetric (directed) forward pass in every
        :class:`GraphLayerAtt`. If False, use the symmetric (undirected) one.
        Default False.
    weight_init_func : callable, optional
        Weight initialisation function.
    dropout : float, optional
        Dropout probability for the dense head.
    """

    def __init__(
        self,
        dims_in,
        conv_dims,
        dense_dims,
        dim_out,
        directed=False,
        weight_init_func=None
    ):
        super(GraphNNAtt, self).__init__()

        self.directed = directed

        self.conv = torch.nn.ModuleList()

        self.conv.append(
            GraphLayerAtt(
                dims_in,
                conv_dims[0],
                directed=directed,
                weight_init_func=weight_init_func,
            )
        )


        for i in range(1, len(conv_dims)):
            self.conv.append(
                GraphLayerAtt(
                    conv_dims[i - 1],
                    conv_dims[i],
                    directed=directed,
                    weight_init_func=weight_init_func,
                )
            )


        self.dense = torch.nn.ModuleList()

        if len(dense_dims) >= 1:
            self.dense.append(torch.nn.Linear(conv_dims[-1][-1], dense_dims[0]))

            for i in range(1, len(dense_dims)):
                self.dense.append(torch.nn.Linear(dense_dims[i - 1], dense_dims[i]))

        self.output_dim = dim_out
        if len(dense_dims) >= 1:
            self.out = torch.nn.Linear(dense_dims[-1], dim_out)
        else:
            self.out = torch.nn.Linear(conv_dims[-1][-1], dim_out)


    def forward(self, x_v, x_a, arc_index):
        if x_v.dim() == 3:
            x_v = x_v.squeeze(-1)
        if x_a.dim() == 3:
            x_a = x_a.squeeze(-1)

        for conv_layer in self.conv :
            x_v, x_a = conv_layer(x_v, x_a, arc_index)
            x_v, x_a = F.relu(x_v), F.relu(x_a)

        for dense_layer in self.dense:
            x_a = dense_layer(x_a)
            x_a = F.relu(x_a)

        return self.out(x_a)
