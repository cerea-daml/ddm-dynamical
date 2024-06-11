#!/bin/env python
# -*- coding: utf-8 -*-
#
#
# Created for ddm_dynamical
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Callable, Any

# External modules

# Internal modules


logger = logging.getLogger(__name__)


class RK4Integrator(object):
    """
    RK4Integrator uses a Runge-Kutta fourth-order method to integrate given
    model function in time. This method calculates four different states, which
    are then averaged to one single state increment. To get another type of
    Runge-Kutta scheme, ``steps`` and ``weights`` can be changed.

    Arguments
    ---------
    model : func
        This model function takes a state and returns a new estimated state. The
        returned state should have the same shape as the input state. The model
        should be a time derivative such that it can be integrated. It is
        assumed that the state does not depend on the time itself.
    dt : float, optional
        This is the integration time step. This time step is unitless and
        depends on model's time unit. A positive time step indicates forward
        integration, while a negative shows a backward integration, which might
        be complicated for given model. Default is 0.05.
    """
    def __init__(self, model: Callable, dt: float = 0.05):
        self._model = None
        self._dt = None
        self.model = model
        self.dt = dt
        self.steps = [0, self.dt / 2, self.dt / 2, self.dt]
        self.weights = [1, 2, 2, 1]
        self._weights_sum = sum(self.weights)
        self._weights = [w / self._weights_sum for w in self.weights]

    @property
    def model(self) -> Callable:
        """
        This model function takes a state and returns a new estimated state. The
        returned state should have the same shape as the input state. The model
        should be a time derivative such that it can be integrated. It is
        assumed that the state does not depend on the time itself.
        """
        return self._model

    @model.setter
    def model(self, new_model: Callable):
        if callable(new_model):
            self._model = new_model
        else:
            raise TypeError('Given model is not callable!')

    @property
    def dt(self) -> float:
        """
        This integration time step specifies the step width for the integration
        and is unit less, and depends on model's time unit. A positive time step
        indicates forward integration, while a negative shows a backward
        integration, which might be complicated for given model.
        """
        return self._dt

    @dt.setter
    def dt(self, new_dt: float):
        if not isinstance(new_dt, (float, int)):
            raise TypeError('Given time step is not a float!')
        elif new_dt == 0:
            raise ValueError('Given time step is zero!')
        else:
            self._dt = new_dt

    def _calc_increment(self, state: Any) -> Any:
        """
        This method estimates the increment based estimated slope and set time
        step.

        Parameters
        ----------
        state : any
            This state is used to estimate the slope.

        Returns
        -------
        est_inc : any
            This increment is estimated by multiplying estimated slope with
            set time step.
        """
        est_inc = self._estimate_slope(state) * self.dt
        return est_inc

    def _estimate_slope(self, state: Any) -> Any:
        """
        This method estimates the slope based on given state. This slope is
        used to calculate the increment.

        Parameters
        ----------
        state : any
            This state is used as initial state to estimates the slopes.

        Returns
        -------
        averaged_slope : any
            This slope is averaged based on estimated slopes. These slopes are
            calculated based on given state.
        """
        averaged_slope = state * 0
        curr_slope = state * 0
        for k, ts in enumerate(self.steps):
            model_state = state + curr_slope * ts
            curr_slope = self.model(model_state)
            averaged_slope += self._weights[k] * curr_slope
        return averaged_slope

    def integrate(self, state: Any) -> Any:
        """
        This method integrates given model by set time step. Given state is used
        as initial state and passed to model.

        Parameters
        ----------
        state : any
            This state is used as initial state for the integration. This state
            is passed to set model.

        Returns
        -------
        int_state : any
            This state is integrated by given model. The integrated state is the
            initial state plus an increment estimated based on this integrator
            and set model.
        """
        estimated_inc = self._calc_increment(state)
        int_state = state + estimated_inc
        return int_state
