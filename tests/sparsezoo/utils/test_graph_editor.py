import numpy as np
import onnx
import pytest
from onnx.backend.test.case.model import expect

from sparsezoo import ONNXGraph


@pytest.fixture
def non_qdq_model():
    from onnx import TensorProto
    from onnx.helper import (
        make_model, make_node, make_graph,
        make_tensor_value_info)


    # 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
    X = make_tensor_value_info('X', TensorProto.FLOAT, [None, None])
    A = make_tensor_value_info('A', TensorProto.FLOAT, [None, None])
    B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

    # outputs, the shape is left undefined

    Y = make_tensor_value_info('Y', TensorProto.FLOAT, [None])

    # nodes

    # It creates a node defined by the operator type MatMul,
    # 'X', 'A' are the inputs of the node, 'XA' the output.
    node1 = make_node('MatMul', ['X', 'A'], ['XA'])
    node2 = make_node('Add', ['XA', 'B'], ['Y'])

    # from nodes to graph
    # the graph is built from the list of nodes, the list of inputs,
    # the list of outputs and a name.

    graph = make_graph([node1, node2],  # nodes
                       'linear_regression',  # a name
                       [X, A, B],  # inputs
                       [Y])  # outputs

    # onnx graph
    # there is no metadata in this case.

    onnx_model = make_model(graph)
    return onnx_model


@pytest.mark.parametrize(
    "model, expected", [
        (pytest.lazy_fixture("non_qdq_model"), False),
        # TODO: add qdq model
    ]
)
def test_is_qdq(model, expected):
    assert ONNXGraph(model).is_qdq() == expected