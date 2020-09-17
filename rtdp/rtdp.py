import random
import copy
MAX_STEPS = 1000
MAX_DEPTH = 10

def transition_function(s,a,next_state):
    probability = 0.0
    return probability

def is_terminal_state(s):
    raise NotImplementedError

def estimated_heuristic_value_function(s):
    return 0.0

def heuristic_value_function(s):
    return 0.0

def update_value(value_function, s):
    return value_function(s)

def greedy_action(value_function, s):
    action = 0
    return action

def choose_next_state(s, a):
    """
    s - state
    a - action
    """
    next_state = 0
    #TODO
    # bounded_gaps = [transition_function() for b_s_tag in ]
    return next_state

def rtdp(admissible_value_function, initial_states):
    """
    admissible_value_function - value function to serve as initial estimation for states
    initial_states            - set of possible initial states    
    """
    converged = False
    step = 0
    visited_stack = []
    estimated_heuristic_value_function = admissible_value_function
    while not converged and step < MAX_STEPS:
        depth = 0
        visited_stack.clear()
        s = copy.deepcopy(random.choice(initial_states))
        while not is_terminal_state(s) and not s is None and depth < MAX_DEPTH:
            depth += 1
            visited_stack.append(s)
            estimated_heuristic_value = update_value(estimated_heuristic_value_function, s)
            a = greedy_action(estimated_heuristic_value_function, s)
            s = choose_next_state(s,a)
        
        while len(visited_stack) > 0:
            s = visited_stack.pop(-1)
            estimated_heuristic_value = update_value(estimated_heuristic_value_function, s)
    return estimated_heuristic_value_function



