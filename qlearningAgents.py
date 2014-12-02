from game import *
from learningAgents import ReinforcementAgent
# import all classes from featureExtractors.py
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QVals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.QVals[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        maxQ = 0.0 if len(actions) == 0 else -9999999
        for a in actions:
            Q = self.getQValue(state, a)
            if Q > maxQ:
                maxQ = Q
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        maxQ = -9999999
        action = None
        for a in actions:
            Q = self.getQValue(state, a)
            if Q > maxQ:
                maxQ = Q
                action = a
        return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return action
        # pick random action based on epsilon
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # current curr value for state and action
        currVal = self.QVals[(state, action)]
        actions = self.getLegalActions(nextState)
        maxNextQ = 0 if len(actions) == 0 else -9999999999
        # get the maxiumum qvalue for actions avilable in state
        for a in actions:
            nextQ = self.QVals[(nextState, a)]
            if nextQ > maxNextQ:
                maxNextQ = nextQ
        updateTerm = self.alpha*(reward + self.discount*maxNextQ - currVal)
        self.QVals[(state, action)] = currVal + updateTerm

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        # our featureExtractor - assigns a single feature to every (state, action) pair

        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        # weights { (state, action): 4.2, .... }

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # print "IS IT GETTING Q VALUE?"
        # import pdb; pdb.set_trace()
        # print self.featExtractor[(state,action)]
        # print self.featExtractor.getFeatures(state, action)
        answer = 0
        for k in self.featExtractor.getFeatures(state, action).keys():
          answer += self.weights[k] * self.featExtractor.getFeatures(state, action)[k]

        return answer
        # self.featExtractor[(state,action)]

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        currVal = self.getQValue(state, action)
        actions = self.getLegalActions(nextState)
        maxNextQ = 0 if len(actions) == 0 else -9999999999
        # get the maxQValue
        for a in actions:
            nextQ = self.getQValue(nextState, a)
            # nextQ = self.QVals[(nextState, a)]
            if nextQ > maxNextQ:
                maxNextQ = nextQ
        difference = (reward + self.discount*maxNextQ) - currVal
        for k in self.featExtractor.getFeatures(state, action).keys():
          self.weights[k] = self.weights[k] +(self.alpha*difference*self.featExtractor.getFeatures(state, action)[k])


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
