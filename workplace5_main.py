"""
workplace5_main.py
第4回シミュレーション実行スクリプト
"""
from workplace5_simulation import WorkplaceSimulation5

if __name__ == "__main__":
    sim = WorkplaceSimulation5(config_path="workplace5_config.yaml")
    sim.run()
