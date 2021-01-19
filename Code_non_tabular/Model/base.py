from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import abc
import numpy as np
import six

from typing import Any, Callable


@six.add_metaclass(abc.ABCMeta)
class BaseAlgo(object):
  """Abstract class representing algorithm for off-policy corrections."""

  @abc.abstractmethod
  def solve(self, data, target_policy):
    """Trains or solves for policy evaluation given experience and policy."""

  @abc.abstractmethod
  def estimate_average_reward(self, data, target_policy):
    """Estimates value (average per-step reward) of policy."""

  def close(self):
    pass

