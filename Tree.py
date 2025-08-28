from typing import List
import numpy as np
from sklearn.base import check_array


class Leaf:
    __slots__ = ("prediction", "loss")
    def __init__(self, prediction: int, loss: float):
        self.prediction = prediction
        self.loss = loss

    def __str__(self) -> str:
        return "{ prediction: " + str(self.prediction) + ", loss: " + str(self.loss) + " }"


class Node:
    __slots__ = ("feature", "left_child", "right_child", "loss")

    def __init__(self, feature: int, left_child, right_child, loss=None):
        self.feature     = feature
        self.left_child  = left_child
        self.right_child = right_child
        self.loss        = loss

    def __str__(self) -> str:
        return (
            f"{{ feature: {self.feature} "
            f"[ left: {self.left_child}, right: {self.right_child} ], "
            f"loss: {self.loss} }}"
        )
