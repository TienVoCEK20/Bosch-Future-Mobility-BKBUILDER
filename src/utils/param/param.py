from enum import Enum

class RunStates(Enum):
    Moving      = 0
    HardTurn    = 1
    SwitchLane  = 2
    Stopping    = 3
    
runState = RunStates.Moving