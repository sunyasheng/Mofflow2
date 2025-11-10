cp -r .env_workstation .env


PYTHONPATH=$PYTHONPATH:$(pwd)  python preprocess/extract_sequence.py
