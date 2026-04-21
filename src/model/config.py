from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "donnees_finales.parquet"


N_SPLITS: int = 3

NATIONALITY_MIN_COUNT: int = 100

TARGET: str = "officialTime"

COLUMNS_TO_DROP: list[str] = [
    "bib",
    "firstName",
    "lastName",
    "realTime",        
    "pace",            
    "averageSpeed",   
    "generalRanking", 
    "sexRanking",       
    "categoryRanking",  
]

SPLIT_COLUMNS_TO_DROP_SUFFIXES: list[str] = [
    "rankGeneral",
    "rankSex",
    "rankCategory",
    "location",      
    "position",     
    "pace",         
    "realTime",     
    "officialTime", 
    "distance",
    
]

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

RIDGE_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ELASTICNET_ALPHAS: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0]
ELASTICNET_L1_RATIOS: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

N_BOOTSTRAP: int = 500
CONFIDENCE_LEVEL: float = 0.95