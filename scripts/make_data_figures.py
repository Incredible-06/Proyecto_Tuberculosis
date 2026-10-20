from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tbxdata.visualize import generate_figures
if __name__=='__main__':
    print(generate_figures(Path('.'),Path('TBX11K'),Path('reports/phase1/figures')))
