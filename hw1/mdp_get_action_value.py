def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    possible_states = mdp.get_next_states(state, action)

    Q_function_value = 0
    for possible_state, possible_state_prob in possible_states.items():
        Q_function_value += possible_state_prob * (mdp.get_reward(state, action, possible_state) + gamma * state_values[possible_state])

    return Q_function_value
