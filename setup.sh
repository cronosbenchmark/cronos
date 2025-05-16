conda create -n cronos python=3.12

conda activate cronos

conda run -n cronos pip install numpy
conda run -n cronos pip install pandas
conda run -n cronos pip install einops
conda run -n cronos pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n cronos pip install biopython
conda run -n cronos pip3 install -U scikit-learn
conda run -n cronos pip install -U matplotlib
conda run -n cronos pip install seaborn
conda run -n cronos pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
conda run -n cronos pip install wandb
conda run -n cronos pip install pillow
conda run -n cronos pip install umap-learn
conda run -n cronos pip install POT
conda run -n cronos pip install loguru
conda run -n cronos pip install omegaconf imblearn
conda run -n cronos pip install flash-attn --no-build-isolation
conda run -n cronos pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.6/
conda run -n cronos pip install -r requirements.txt
conda run -n cronos pip install scikit-image accelerate