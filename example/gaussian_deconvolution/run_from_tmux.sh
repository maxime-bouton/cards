#!/bin/bash
# Run one of the provided bash scripts in the background using a TMUX session.
# Requires tmux (more details here: https://github.com/tmux/tmux/wiki)
#
# ! The instruction below creates a new tmux session, named "jcgs", in a
# ! detached  mode runing the bash script. The session terminates as soon as
# ! the task is complete.
#
# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)
#
# reference: P.-A. Thouvenin, A. Repetti, P. Chainais - **A distributed Gibbs
# Sampler with Hypergraph Structure for High-Dimensional Inverse Problems**,
# [arxiv preprint 2210.02341](http://arxiv.org/abs/2210.02341), October 2022.

# ! - to join a detached session, type "tmux a -t jcgs" in a terminal
# ! - in a tmux session, press:
# !     - Ctrl+D to kill the session
# !     - Ctrl+B, then D, to detach from the tmux session
# !     - Ctrl+C to interrupt the job currently running in the tmux session

# ! First activate the appropriate conda environment in the terminal from which
# ! the tmux job is run with the command
# ! conda activate jcgs-review

tmux new -s benchmark -d 'submit_local.sh'
