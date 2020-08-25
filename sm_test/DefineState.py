#!/usr/bin/python3

from statemachine import StateMachine, State

class TrafficLightMachine(StateMachine):
    green = State('Green', initial=True)
    yellow = State('Yellow')
    red = State('Red')

    slowdown = green.to(yellow)
    stop = yellow.to(red)
    go = red.to(green)

class MyModel(object):
    def __init__(self, state):
        self.state = state


#########
#
# from sm_test.DefineState import TrafficLightMachine
#
# >>> traffic_light = TrafficLightMachine()
# >>> traffic_light.current_state
# State('Green', identifier='green', value='green', initial=True)
# >>> traffic_light.current_state == traffic_light.green
# True
# >>> traffic_light.is_green
# True
# [s.identifier for s in traffic_light.states]
# ['green', 'red', 'yellow']
# [t.identifier for t in traffic_light.transitions]
# ['go', 'slowdown', 'stop']
# traffic_light.slowdown()
# traffic_light.current_state == traffic_light.green
# False
# traffic_light.current_state
# State('Yellow', identifier='yellow', value='yellow', initial=False)

# from sm_test.DefineState import TrafficLightMachine, MyModel
# >>> obj = MyModel(state='red')
# >>> traffic_light = TrafficLightMachine(obj)
# >>> traffic_light.is_red
# True
# >>> obj.state
# 'red'
# >>> obj.state = 'green'
# >>> traffic_light.is_green
# True
# >>> traffic_light.slowdown()
# >>> obj.state
# 'yellow'
# >>> traffic_light.is_yellow
# True