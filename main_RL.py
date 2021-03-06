# -*- coding: utf-8 -*-


import time
import numpy as np
import matplotlib.pyplot as plt
import vrep
import contexttimer
from environment import Robot
import random

class Agent(object):
    '''
    This class is responsible for our robot representation
    '''
    def __init__(self, robot, alpha, gamma, epsilon, q_init):
        self.robot = robot
        self.num_actions_0 = 3
        self.num_actions_1 = 3
        self.num_actions = self.num_actions_0 * self.num_actions_1
        self.angles_0 = np.linspace(0, np.pi/2, self.num_actions_0)
        self.angles_1 = np.linspace(0, np.pi/2, self.num_actions_1)

        # look-up table from action to angles
        self.angles_lut = np.array(np.meshgrid(self.angles_1, self.angles_0,
                                   indexing='ij')).reshape(2, -1).T

        self.num_states_0 = 5  # angle of joint 0
        self.num_states_1 = 5  # angle of joint 1
        self.num_states = self.num_states_0 * self.num_states_1
        self.state_bins = [
            np.linspace(0, np.pi/2, self.num_states_0, endpoint=False)[1:],
            np.linspace(0, np.pi/2, self.num_states_1, endpoint=False)[1:]]

        self.q_table = np.full((self.num_states, self.num_actions), q_init) #todo CHECK IT
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon

    def choose_action(self, state):
        # ToDo: implement choose action function
        # TODO implement here serch policy
        r = random.random()

        if r > self.epsilon:
            action = np.argmax(self.q_table[state])
            # print(self.q_table[state])
            # print( 'max ' + str(self.q_table[state].argmax() ) + str(self.q_table[ self.q_table[state].argmax() ] ) )

            #TODO error here
         #   action = random.choice(range(len(self.q_table[state])))

            #greede select
            # pass
        else:
            action = random.choice( range ( len ( self.q_table[state]) ) )
            # pass
            #explore select:

        return action

    def do_action(self, action):
        angles = self.angles_lut[action]
        self.robot.set_joint_angles(angles)
        self.robot.proceed_simulation()

    def observe_state(self):
        angles = self.robot.get_joint_angles()
        return self.calc_state(angles)

    def calc_state(self, angles):
        state_0 = np.digitize([angles[0]], self.state_bins[0])[0]
        state_1 = np.digitize([angles[1]], self.state_bins[1])[0]
        state = state_0 * self.num_states_1 + state_1
        return state

    def play(self):
        action = self.choose_action(self.state)
        self.do_action(action)

        state_new = self.observe_state()

        position_new = self.robot.get_body_position()
        x_forward = position_new[0] - self.position[0]
        reward = x_forward - 0.001

        # update Q-table
        self.update_q(self.state, action, reward, state_new)

        self.state = state_new
        self.position = position_new

    def update_q(self, state, action, reward, state_new):
        #state_new = state
        # old_state = self.q_table[state][action]
        q_sa = self.q_table[state, action]
        self.q_table[state, action] = q_sa + self.alpha * (reward + self.gamma * np.max(self.q_table[state_new]) - q_sa)
        # self.q_table[state][action] += self.alpha * (reward + gamma * np.argmax(self.q_table[state_new]) - self.q_table[state][action])
        #ToDo: implement q-value update
		# pass

    def initialize_episode(self):
        self.robot.restart_simulation()
        self.robot.initialize_pose()
        self.position = self.robot.get_body_position()
        angles = self.robot.get_joint_angles()
        self.state = self.calc_state(angles)


def plot(body_trajectory, joints_trajectory, return_history, q_table):
    fig = plt.figure(figsize=(9, 4))
    T = len(body_trajectory)

    # plot an xyz trajectory of the body
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(222)
    ax4 = plt.subplot(224)
    ax1.grid()
    ax1.set_color_cycle('rgb')
    ax1.plot(np.arange(T) * 0.05, np.array(body_trajectory))
    ax1.set_title('Position of the body')
    ax1.set_ylabel('position [m]')
    ax1.legend(['x', 'y', 'z'], loc='best')

    # plot a trajectory of angles of the joints
    ax2.grid()
    ax2.set_color_cycle('rg')
    ax2.plot(np.arange(T) * 0.05, np.array(joints_trajectory))
    ax2.set_title('Angle of the joints')
    ax2.set_xlabel('time in simulation [s]')
    ax2.set_ylabel('angle [rad]')
    ax2.legend(['joint_0', 'joint_1'], loc='best')

    # plot a history of returns of each episode
    ax3.grid()
    ax3.plot(return_history)
    ax3.set_title('Returns (total rewards) of each episode')
    ax3.set_xlabel('episode')
    ax3.set_ylabel('position [m]')

    # show Q-table
    ax4.matshow(q_table.T, cmap=plt.cm.gray)
    ax4.set_title('Q-table')
    ax4.xaxis.set_ticks_position('bottom')
    ax4.set_xlabel('state')
    ax4.set_ylabel('action')
    plt.tight_layout()
    plt.show()
    plt.draw()


if __name__ == '__main__':
    try:
        client_id
    except NameError:
        client_id = -1
    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(-1)
    client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    assert client_id != -1, 'Failed connecting to remote API server'

    # print ping time
    sec, msec = vrep.simxGetPingTime(client_id)
    print "Ping time: %f" % (sec + msec / 1000.0)

    robot = Robot(client_id)
    #learning rate alpha
    alpha = 1 # as we have to init with some values,
    # but don't know which exactly to chose therfore we taking learning rate as hight as possible
    # to 'intialize' transitons with revard function's values
    # and rapidly decrease it to feasible values to preserve convergence
    final_alpha = 0.2

    num_episodes = 20
    len_episode = 150
    return_history = []

    #  discount factor gamma,
    gamma = 0.7 # decrease of weight for unpleasant values
    # have to preserve not too hight, to prevent of learning mistakes like a positive revard
    #  epsilon-greedy rate epsilon
    epsilon = 0.3 # explore / exploit partition, lower part - for explore
    epsilon_end = 0.005 #final epsillon value to make our crawler move greedily in at the last iterations
    epsilon_decrease_rate = (epsilon_end / epsilon) ** (1.0/num_episodes)

    #  initil value of q-table q_init
    q_init = 0.0 # ignore init policy, instead rew
    agent = Agent(robot, alpha, gamma, epsilon, q_init)




    try:
        for episode in range(num_episodes):
            print "start simulation # %d" % episode

            with contexttimer.Timer() as timer:
                agent.initialize_episode() #nothing to place here, just init values
                body_trajectory = []
                joints_trajectory = []
                body_trajectory.append(robot.get_body_position())
                joints_trajectory.append(robot.get_joint_angles())

                for t in range(len_episode):
                    agent.play()
                    print agent.state

                    body_trajectory.append(robot.get_body_position())
                    joints_trajectory.append(robot.get_joint_angles())

            position = body_trajectory[-1]
            return_history.append(position[0])
            q_table = agent.q_table
            agent.epsilon = agent.epsilon * epsilon_decrease_rate
            agent.alpha = final_alpha + agent.alpha * 0.7
            print( "EPSILLON" + str (agent.epsilon) )

          #  plot(body_trajectory, joints_trajectory, return_history, q_table)
            print
            print "Body position: ", position
            print "Elapsed time (wall-clock): ", timer.elapsed
            print

    except KeyboardInterrupt:
        print "Terminated by `Ctrl+c` !!!!!!!!!!"

    plt.grid()
    plt.plot(return_history)
    plt.title('Return (total reward in a episode)')
    plt.xlabel('episode')
    plt.ylabel('position [m]')
    plt.show()

    e = vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
