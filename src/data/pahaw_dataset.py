from torch.utils.data import Dataset
from src.visualization.visualizer import Visualizer, PlotType
from pandas import DataFrame

class PaHaWDataset(Dataset):

    def __init__(self, mapping, df:DataFrame = None, visualizer: Visualizer = None):
        self.mapping = mapping
        self.df = df.set_index("ID", drop=False) if df is not None else None        
        self.patient_ids = sorted(mapping.keys())
        self.visualizer = visualizer

    def __len__(self):
        return sum(len(tasks) for tasks in self.mapping.values())

    def __getitem__(self, key):

        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], int):
            patient_id, task_idx = key

            if patient_id not in self.mapping:
                raise IndexError(f"Patient ID {patient_id} not found")

            return self.mapping[patient_id][task_idx]

        if self.df is not None:
            return self.df[key]

        raise KeyError("Invalid key")
    
    def __getattr__(self, name):
        if self.df is not None and hasattr(self.df, name):
            return getattr(self.df, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name}")
    
    def get(self, column):
        if self.df is None:
            raise ValueError("No DataFrame provided")
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")

        return self.df[column]

    def visualize(self, patient_id, task_idx,
                  plot_type: PlotType,
                  **kwargs):

        if self.visualizer is None:
            raise ValueError("No visualizer provided")

        sample = self[patient_id, task_idx]
        self.visualizer.plot(plot_type, sample, **kwargs)