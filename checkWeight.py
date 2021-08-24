from _modules.datamodule import EEGDataModule
import _modules.params as p

data = EEGDataModule(1)
data.setup()
print(data.check_balance())
