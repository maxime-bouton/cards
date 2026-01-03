Installation
============

Quickstart
----------

The Python package can be installed on ``ubuntu`` with ``cuda`` GPU support within an existing ``conda``-compatible environment (e.g., using either ``pixi``, ``mamba`` or ``conda``).
Example installation commands can be found below.

.. code-block:: bash

    # installation within a mamba environment "my_samplers"
    mamba env create -n my_samplers
    mamba install cards -c pthouvenin

    # installation within an existing pixi environment
    pixi workspace channel add pthouvenin
    pixi add cards


Retrieving pre-trained weights for DRUNet and DnCNN
---------------------------------------------------

A distributed implementation is provided for the ``DRUNet``, ``DnCNN`` and ``DDFB`` deep denoisers.
Pre-trained weights are not embedded into the ``cards`` ``conda``-package.
The weights need to be retrieved separately, using for instance the commands detailed below.

.. code-block:: bash

    mkdir -p data/weights && cd data/weights

    # * DDFB
    mkdir ddfb && cd ddfb
    wget https://github.com/maxime-bouton/cards/blob/main/data/weights/ddfb/ddfb_nch3_nla20_nfe64.pth

    # * retrieving weights for DRUNet and DnCNN from https://github.com/cszn/KAIR
    # (see https://drive.google.com/drive/folders/13kfr3qny7S2xwG9h7v95F5mkWs0OmU0D
    # and https://github.com/cszn/DPIR/tree/master/model_zoo)
    #
    # DRUNet (gray and color images)
    cd ../ && mkdir drunet && cd drunet
    wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth && mv drunet_gray.pth drunet_nch1.pth

    wget https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth && mv drunet_color.pth drunet_nch3.pth

    # DnCNN (gray and color images)
    cd ../ && mkdir dncnn && cd dncnn
    wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_gray_blind.pth && mv dncnn_gray_blind.pth dncnn_nch1.pth

    wget https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth && mv dncnn_color_blind.pth dncnn_nch3.pth
