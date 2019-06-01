# RBM-polymer
Restricted Boltzmann Machine of lattice polymer model

Requirements: Python==2.7.13  mxnet==1.0.0

Example command for training:
    python RBM-binary.py -v 6 -n 20 -l 2 -d conf1.txt -w w20 -t 10000 -b 500 -i 20 -m 0.5 -rg enum_rgs.txt -kl enum_states.txt -x 0

Example command for generating:
    python RBM-binary.py -v 6 -n 20 -l 2 -d conf1.txt -w w20 -g 1 -k 100 -x 0
