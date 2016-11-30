import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()################################## S is the set of all states
              mdp.getPossibleActions(state)#################### A is the set of all actions
              mdp.getTransitionStatesAndProbs(state, action)### P is state transition function specifying P(s'|s,a)
              mdp.getReward(state, action, nextState)########## R is a reward function R(s,a,s')
              mdp.isTerminal(state)############################ Theta a threshold, Theta>0
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        """
        Value iteration starts at the "end" and then works backward,
        refining an estimate of either Q* or V*. There is really no end,
        so it uses an arbitrary end point.
        Let Vk be the value function assuming there are k stages to go,
        and let Qk be the Q-function assuming there are k stages to go.
        These can be defined recursively.
        Value iteration starts with an arbitrary function V0 and uses
        the following equations to get the functions for k+1 stages
        to go from the functions for k stages to go.

        initialize V(s) arbitrarily
		loop until policy good enough
				loop for s in the set of all states
						loop for a in the set of all actions
								 compute Q value from values(state, action)
			                      get the best seen value for that state and action
			            end loop
				end loop
        """

        k = 0
        while k < self.iterations:
            V = util.Counter()
            for s in mdp.getStates():
                if not mdp.isTerminal(s):
                    sVal = util.Counter()
                    for a in mdp.getPossibleActions(s):
                        sVal[a] = self.computeQValueFromValues(s, a)
                    V[s] = max(sVal.values())
            k += 1
            self.values = V.copy()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        v = 0
        for valuePair in self.mdp.getTransitionStatesAndProbs(state,action):
            v += valuePair[1] * (self.mdp.getReward(state, action, valuePair[0]) + self.discount * self.values[valuePair[0]])

        return v

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #mdp.isTerminal(state)  ############################ Theta a threshold, Theta>0
        if mdp.isTerminal(state): return None
        #mdp.getPossibleActions(state)#################### A is the set of all actions
        A = mdp.getPossibleActions(state)
        #If the list is empty...
        if not A: return None
        #another dictionary to hold the values...
        v = util.Counter()

        for i in A:
            v[i] = self.getQValue(state, i)
        return v.argMax()


        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
