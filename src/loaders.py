import os
import numpy as np
from scipy.io import loadmat

# Registro de metadatos de clases y estadísticas
DATASET_REGISTRY = {
    "Indian": {
        "class_stats": {
            'Alfalfa': 46, 'Corn-notill': 1428, 'Corn-mintill': 830, 'Corn': 237,
            'Grass-pasture': 483, 'Grass-trees': 730, 'Grass-pasture-mowed': 28,
            'Hay-windrowed': 478, 'Oats': 20, 'Soybean-notill': 972,
            'Soybean-mintill': 2455, 'Soybean-clean': 593, 'Wheat': 205,
            'Woods': 1265, 'Buildings-Grass-Trees-Drives': 386, 'Stone-Steel-Towers': 93
        }
    },
    "Pavia": {
        "class_stats": {
            'Asphalt': 6631, 'Meadows': 18649, 'Gravel': 2099, 'Trees': 3064,
            'Metal Sheets': 1345, 'Bare Soil': 5029, 'Bitumen': 1330,
            'Bricks': 3682, 'Shadows': 947
        }
    },
    "Salinas": {
        "class_stats": {
            'Brocoli green weeds 1': 12009, 'Brocoli green weeds 2': 3726, 'Fallow': 1976,
            'Fallow rough plow': 1394, 'Fallow smooth': 2678, 'Stubble': 3959,
            'Celery': 3579, 'Grapes untrained': 11271, 'Soil vinyard develop': 6203,
            'Corn senesced green weeds': 3278, 'Lettuce romaine 4wk': 1068,
            'Lettuce romaine 5wk': 1927, 'Lettuce romaine 6wk': 916,
            'Lettuce romaine 7wk': 1070, 'Vinyard untrained': 7268, 'Vinyard vertical trellis': 1807
        }
    },
    "Jasper Ridge": {
        "class_stats": {
            'Tree': 0, 'Water': 0, 'Soil': 0, 'Road': 0
        }
    },
    "Urban": {
        "class_stats": {
            'Asphalt': 0, 'Grass': 0, 'Tree': 0, 'Roof': 0
        }
    },
    "Samson": {
        "class_stats": {
            'Soil': 0, 'Tree': 0, 'Water': 0
        }
    }
}

DATASET_CONFIG = {
    'Pavia': {
        'file':     'Pavia_University.mat',
        'key':      'paviaU',
        'gt_file':  'Pavia_University_gt.mat',
        'gt_key':   'paviaU_gt',
        'rgb_bands': (102, 56, 31),
    },
    'Indian': {
        'file':     'Indian_pines_corrected.mat',
        'key':      'indian_pines_corrected',
        'gt_file':  'Indian_pines_gt.mat',
        'gt_key':   'indian_pines_gt',
        'rgb_bands': (50, 27, 17),
    },
    'Salinas': {
        'file':     'Salinas_corrected.mat',
        'key':      'salinas_corrected',
        'gt_file':  'Salinas_gt.mat',
        'gt_key':   'salinas_gt',
        'rgb_bands': (50, 27, 17),
    },
    'Jasper Ridge': {
        'rgb_bands': (27, 15, 7),
    },
    'Urban': {
        'rgb_bands': (50, 27, 17),
    },
    'Samson': {
        'rgb_bands': (81, 48, 20),
    },
}


class DatasetLoader:
    def __init__(self, dataset_name, data_dir='data'):
        if dataset_name not in DATASET_CONFIG:
            raise ValueError(
                f"Dataset '{dataset_name}' no soportado. Opciones: {list(DATASET_CONFIG.keys())}"
            )
        self.cfg = DATASET_CONFIG[dataset_name]
        self.data_dir = data_dir
        self.dataset_name = dataset_name

    def load(self):
        """Returns (image_normalized, gt, metadata)."""
        mat = loadmat(os.path.join(self.data_dir, self.cfg['file']))
        image = mat[self.cfg['key']].astype(np.float64)
        image /= image.max()

        gt = None
        gt_path = os.path.join(self.data_dir, self.cfg['gt_file'])
        if os.path.exists(gt_path):
            gt_mat = loadmat(gt_path)
            gt = gt_mat[self.cfg['gt_key']]

        rows, cols, bands = image.shape
        metadata = {
            'dataset': self.dataset_name,
            'rows': rows,
            'cols': cols,
            'bands': bands,
        }
        return image, gt, metadata
