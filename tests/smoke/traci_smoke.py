"""Minimal traci smoke test: start SUMO, step 10 times, close cleanly."""
import os
import sys
from pathlib import Path

if "SUMO_HOME" not in os.environ:
    sys.exit("SUMO_HOME not set")
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci
import sumolib

HERE = Path(__file__).parent
SUMO_BIN = sumolib.checkBinary("sumo")
CFG = str(HERE / "dummy.sumocfg")

traci.start([SUMO_BIN, "-c", CFG, "--no-step-log", "true", "--no-warnings", "true"])
for i in range(10):
    traci.simulationStep()
    t = traci.simulation.getTime()
    print(f"step {i}: sim_time={t}")
traci.close()
print("smoke test OK")
