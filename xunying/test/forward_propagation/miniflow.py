# -*- coding:utf-8 -*-

"""
Author:xunying/Jasmine
Data:17-4-12
Time:下午2:29
"""
import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        # nodes from which this node receives values
        self.inbound_nodes = inbound_nodes

        # nodes to which this node passes values
        self.outbound_nodes = []

        # a calulated _value
        self.value = None

        # add this node as an outbound node on its inputs
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # these will be implement in a subclass
    def forward(self):
        # forward propaation 基于 inbound_nodes 和　存储的结果self.value计算output value
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # Input节点，没有 inbound nodes,所以不需要输入pass　到这该节点
        Node.__init__(self)

    # example:val0=self.inbound_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        # 设置　该节点self.value值　to the sum of its inbound_nodes
        values=[self.inbound_nodes[one].value for one in range(len(self.inbound_nodes))]
        self.value=sum(values)
        #x_value=self.inbound_nodes[0].value
        #y_value=self.inbound_nodes[1].value
        #self.value = x_value+y_value



def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
