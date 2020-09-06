#!/usr/bin/python3

from statemachine import StateMachine, State

class UGVLocalMachine(StateMachine):
    zero = State('Zero', initial=True)
    point1 = State('Point1')
    wait1 = State('Wait1')
    wait2 = State('Wait2')
    point2 = State('Point2')

    phase1 = zero.to(point1)
    phase2 = point1.to(wait1)
    phase3 = wait1.to(wait2)
    phase4 = wait2.to(point2)

class SuicideLocalMachine(StateMachine):
    zero = State('Zero', initial=True)
    suicide1 = State('Suicide1')
    suicide2 = State('Suicide2')
    suicide3 = State('Suicide3')
    suicideZZ = State('SuicideZZ')

    # Used: phase_i_2, phase3, phase4
    phase_i_1 = zero.to(suicide1)
    phase_i_2 = zero.to(suicide2)
    phase2 = suicide1.to(suicide2)
    phase3 = suicide2.to(suicide3)
    phase4 = suicide3.to(suicide2)
    phaseZZ_3 = suicideZZ.to(suicide3)
    phase2_ZZ = suicide2.to(suicideZZ)
    phase3_ZZ = suicide3.to(suicideZZ)



class DroneLocalMachine(StateMachine):
    zero = State('Zero', initial=True)
    scanner1 = State('Scanner1')
    scanner2 = State('Scanner2')
    scanner3 = State('Scanner3')

    phase1 = zero.to(scanner1)
    phase2 = scanner1.to(scanner2)
    phase3 = scanner2.to(scanner1)
    ### phase 4 & 5 are provision to future
    phase4 = scanner2.to(scanner3)
    phase5 = scanner3.to(scanner1)