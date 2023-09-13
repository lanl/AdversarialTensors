#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 19:35:02 2023

@author: cagri
"""
from lightning.pytorch.callbacks import TQDMProgressBar

class MyProgressBar(TQDMProgressBar):
    """
    A subclass of TQDMProgressBar to customize the progress bar behavior.

    This class disables the progress bar during the validation phase.
    """
    def init_validation_tqdm(self):
        """
        Initialize the TQDM progress bar for the validation phase.

        Returns
        -------
        bar : tqdm.tqdm
            The TQDM progress bar instance for validation.
        """
        bar = super().init_validation_tqdm()
        bar.disable = True
        return bar

