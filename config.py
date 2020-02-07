# """"""""""""""""""""""""""""""""""""""""""""
# """""""""""""""""" Paths """""""""""""""""""
# """"""""""""""""""""""""""""""""""""""""""""
EXPERIMENTS_PATH = "~/experiments/tumor/"
DATA_PATH = "~/datasets/tumor/"

INFERENCE_IMAGES = "~/datasets/tumor/x_test/images/"
INFERENCE_CLINICAL = "~/datasets/tumor/x_test/features/clinical_data.csv"


# """"""""""""""""""""""""""""""""""""""""""""
# """"""""""""" Structured Data """"""""""""""
# """"""""""""""""""""""""""""""""""""""""""""
DROP_COLS = ["SourceDataset"]
CAT_COLS = [
    "Histology",
]
REG_COLS = [
    "age",
    "Mstage",
    "Nstage",
    "Tstage",
]
