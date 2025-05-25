# test_module.py
from compactformer.patchtst import PatchTSTMinimal, PatchTSTStandard, PatchTSTFull
from compactformer.datasets import PatchDataset
from compactformer.signals import generate_noisy_smooth_signals
from scripts import run_patchtst_grid
from scripts import run_informer_grid
from scripts import run_autoformer_grid

print("All imports successful!")

# Optionally: instantiate a class to verify everything is wired
model = PatchTSTMinimal(patch_length=8, horizon=4, d_model=8, num_heads=2, dim_feedforward=32, num_layers=2)
print(model)


"""

# Example hyperparameters
patch_length = 24
horizon = 12
d_model = 16
num_heads = 2

model_autoformer_std = StandardAutoformer(
    patch_length=patch_length,
    horizon=horizon,
    d_model=d_model,
    num_heads=num_heads
)

print(model_autoformer_std)


"""



"""
patch_lengths = [4, 8, 12, 16, 20]
horizons = [4, 8, 12, 16, 20]
results_dir = "simulation_results"
epochs = 600
batch_size = 32
"""



#run_patchtst_grid.main(epochs=2,batch_size=32, patch_lengths=[4],horizons=[4])

#run_informer_grid.main(epochs=2,batch_size=32, patch_lengths=[4],horizons=[4])

#run_autoformer_grid.main(epochs=2,batch_size=32, patch_lengths=[4],horizons=[4])
