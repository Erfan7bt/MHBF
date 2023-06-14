import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Gridworld:
    """
    A class that implements a quadratic NxN gridworld.

    Attributes
    ----------
    N : int
        Gridworld size.
    reward_position : tuple
        Reward location (x, y).
    reward_at_target : float
        Reward administered at the target location.
    reward_at_wall : float
        Reward administered at when bumping into walls.
    obstacle : bool
        Whether there is an obstacle in the room.
    epsilon : float
        Probability at which the agent chooses a random action. This makes sure the
        agent explores the grid.
    eta : float
        Learning rate.
    gamma : float
        Discount factor - quantifies how far into the future a reward is still
        considered important for the current action.
    lambda_eligibility : float
        The decay factor for the eligibility trace. 0 corresponds to no eligibility
        trace at all.
    x_position : int
        Position in x-dimension for the current time step.
    x_position_old : int
        Position in x-dimension for the previous time step.
    y_position : int
        Position in y-dimension for the current time step.
    y_position_old : int
        Position in y-dimension for the previous time step.
    action : int
        Action for the current time step (0: down, 1: up, 2:right, 3: left).
    action_old : int
        Action for the previous time step (0: down, 1: up, 2:right, 3: left).
    latency_list : list
        A list of latencies for the last learning run. Length of the list is the number
        of trials in that run. This is reset at each new run (if `N_runs > 1` in
        `run()`).
    latencies : numpy.ndarray
        Array of size `N_trials` during the last `run()` call. The values are the
        average latencie for each trial, averaged over all learning runs (`N_runs`).
    _wall_touch : bool
        Whether the last action ended up in an obstacle.


    Methods
    -------
    run(N_trials=10,N_runs=1)
        Run 'N_trials' trials. A trial is finished, when the agent reaches the reward
        location.
    visualize_trial()
        Run a single trial with graphical output.
    reset()
        Make the agent forget everything she has learned.
    plot_Q()
        Plot of the Q-values.
    learning_curve()
        Plot the time it takes the agent to reach the target as a function of trial
        number.
    navigation_map()
        Plot the movement direction with the highest Q-value for all positions.
    _learn_run(N_trials=10)
        Run a learning period consisting of `N_trials` trials.
    _update_state()
        Update the state according to the old state and the current action.
    _reward()
        Evaluates how much reward should be administered when performing the
        chosen action at the current location
    _is_wall()
        This function returns whether the given position is within an obstacle.
    _arrived()
        Check if the agent has arrived at the `reward_position`.
    _run_trial()
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.
    _update_Q()
        Update the current estimate of the Q-values according to SARSA.
    _choose_action()
        Choose the next action based on the current estimate of the Q-values.
    """

    def __init__(
        self,
        N,
        reward_position=(0, 0),
        epsilon=0.5,
        obstacle=True,
        lambda_eligibility=0.0,
    ):
        """
        Creates a quadratic NxN gridworld.

        Parameters
        ----------
        N : int
            Size of the gridworld.
        reward_position : tuple
            Tuple of (x_coordinate, y_coordinate) for the reward location.
        epsilon : float
            Epsilon in the epsilon-greedy policy. That is the probability at which the
            agent chooses a random action.
        obstacle : bool
            Whether to add a wall to the gridworld or not.
        lambda_eligibility : float
            The decay factor for the eligibility trace. The default is 0, which
            corresponds to no eligibility trace at all.
        """

        # gridworld size
        self.N = N

        # reward location
        self.reward_position = reward_position

        # reward administered t the target location and when
        # bumping into walls
        self.reward_at_target = 1.0
        self.reward_at_wall = -0.5

        # probability at which the agent chooses a random
        # action. This makes sure the agent explores the grid.
        self.epsilon = epsilon

        # learning rate
        self.eta = 0.1

        # discount factor - quantifies how far into the future
        # a reward is still considered important for the
        # current action
        self.gamma = 0.99

        # the decay factor for the eligibility trace the
        # default is 0., which corresponds to no eligibility
        # trace at all.
        self.lambda_eligibility = lambda_eligibility

        # is there an obstacle in the room?
        self.obstacle = obstacle

        # initialize the state and action variables
        self.x_position = None
        self.x_position_old = None
        self.y_position = None
        self.y_position_old = None
        self.action = None
        self.action_old = None

        # list that contains the times it took the agent to reach the target for all
        # trials, serves to track the progress of learning
        self.latency_list = []

        # draw animation of agent exploring?
        self._visualize = False

        # whether the last action bumped into a wall
        self._wall_touch = None

        # initialize the Q-values etc.
        self._init_run()


    def run(self, N_trials=10, N_runs=1):
        """
        Let an agent learn for `N_trails` trials for `N_run` runs.

        Parameters
        ----------
        N_trials : int
            Number of trials in a single run.
        N_runs : int
            Number of runs (with newly initalized Q-values). If `N_runs > 1`, the
            latencies stored in `self.latencies` will be an average over all runs.
        """
        self.latencies = np.zeros(N_trials)

        for _ in range(N_runs):
            self._init_run()
            latencies = self._learn_run(N_trials=N_trials)
            self.latencies += latencies / N_runs

    def visualize_trial(self):
        """
        Run a single trial with a graphical display that shows in
                red   - the position of the agent
                blue  - walls/obstacles
                green - the reward position

        Note that for the simulation, exploration is reduced -> self.epsilon=0.1
        """
        # store the old exploration/exploitation parameter
        epsilon = self.epsilon

        # favor exploitation, i.e. use the action with the
        # highest Q-value most of the time
        self.epsilon = 0.1

        self._init_visualization()
        self._run_trial()

        # restore the old exploration/exploitation factor
        self.epsilon = epsilon

        return self._finish_visualization()

    def learning_curve(self, log=False, filter_t=1.0):
        """
        Show a running average of the time it takes the agent to reach the target location.

        Parameters
        ----------
        log : bool
            Whether to use a logarithmic scale on the y-axis.
        filter_t : float
            Timescale of the running average.
        """
        plt.figure()
        plt.xlabel("trials")
        plt.ylabel("time to reach target")
        latencies = np.array(self.latency_list)
        # calculate a running average over the latencies with a averaging time 'filter_t'
        for i in range(1, latencies.shape[0]):
            latencies[i] = latencies[i - 1] + (latencies[i] - latencies[i - 1]) / float(
                filter_t
            )

        if not log:
            plt.plot(self.latencies)
        else:
            plt.semilogy(self.latencies)

    def navigation_map(self):
        """
        Plot the direction with the highest Q-value for every position.
        Useful only for small gridworlds, otherwise the plot becomes messy.
        """
        self.x_direction = np.zeros((self.N, self.N))
        self.y_direction = np.zeros((self.N, self.N))

        self.actions = np.argmax(self.Q[:, :, :], axis=2)
        self.y_direction[self.actions == 0] = 1.0
        self.y_direction[self.actions == 1] = -1.0
        self.y_direction[self.actions == 2] = 0.0
        self.y_direction[self.actions == 3] = 0.0

        self.x_direction[self.actions == 0] = 0.0
        self.x_direction[self.actions == 1] = 0.0
        self.x_direction[self.actions == 2] = 1.0
        self.x_direction[self.actions == 3] = -1.0

        plt.figure(figsize=(self.N, self.N))
        plt.quiver(self.x_direction, self.y_direction, pivot="mid")
        plt.axis([-0.5, self.N - 0.5, -0.5, self.N - 0.5])

    def reset(self):
        """
        Reset the Q-values (and the latency_list).

        Instant amnesia - the agent forgets everything she has learned before
        """
        self._init_run()

    def plot_Q(self):
        """
        Plot the dependence of the Q-values on position. The figure consists of 4
        subgraphs, each of which shows the Q-values colorcoded for one of the actions.
        """
        plt.figure(figsize=(0.75 * self.N, 0.75 * self.N))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(
                self.Q[:, :, i], interpolation="nearest", origin="lower", vmax=1.1
            )
            if i == 0:
                plt.title("Up")
            elif i == 1:
                plt.title("Down")
            elif i == 2:
                plt.title("Right")
            else:
                plt.title("Left")

            plt.colorbar()

    def _init_run(self):
        """
        Initialize the Q-values, eligibility trace, position etc.
        """
        # initialize the Q-values and the eligibility trace
        self.Q = 0.01 * np.random.rand(self.N, self.N, 4) + 0.1
        self.e = np.zeros((self.N, self.N, 4))

        # reset the latency list
        self.latency_list = []
        # reset state and action variables
        self.x_position = None
        self.x_position_old = None
        self.y_position = None
        self.y_position_old = None
        self.action = None
        self.action_old = None

    def _check_position_initialized(self, err_msg=""):
        """
        Check if the position is initialized. If not, raise an error.

        Raises
        ------
        RuntimeError
            If `self.x_position` or `self.y_position` are not initialized.
        """
        if self.x_position is None or self.y_position is None:
            raise RuntimeError(
                f"{err_msg} `self.x_position` or `self.y_position` is `None`. You "
                "need to initialize your state before you can use it."
            )

    def _check_action_initialized(self, err_msg=""):
        """
        Check if the action is initialized. If not, raise an error.

        Raises
        ------
        RuntimeError
            If `self.action` is `None`.
        """
        if self.action is None:
            raise RuntimeError(
                f"{err_msg} `self.action` is `None`. You need to choose an action "
                "first."
            )

    def _learn_run(self, N_trials=10):
        """
        Run a learning period consisting of N_trials trials.

        Note: The Q-values are not reset. Therefore, running this routine
        several times will continue the learning process. If you want to run
        a completely new simulation, call reset() before running it.

        Parameters
        ----------
        N_trials : int
            Number of trials.

        Returns
        -------
        latencies : numpy.ndarray
            1D array of size `N_trials` with latencies for each trial.
        """
        for _ in range(N_trials):
            # run a trial and store the time it takes to the target
            latency = self._run_trial()
            self.latency_list.append(latency)

        return np.array(self.latency_list)

    def _run_trial(self):
        """
        Run a single trial on the gridworld until the agent reaches the reward position.
        Return the time it takes to get there.
        """
        raise NotImplementedError(
            "`_run_trial` is not implemented. Please implement the function and "
            "overwrite the `Gridworld._run_trial` method."
        )

    def _update_Q(self):
        """
        Update the current estimate of the Q-values according to SARSA.
        """
        raise NotImplementedError(
            "`_update_Q` is not implemented. Please implement the function and "
            "overwrite the `Gridworld._update_Q` method."
        )

    def _choose_action(self):
        """
        Choose the next action based on the current estimate of the Q-values.
        The parameter epsilon determines, how often agent chooses the action
        with the highest Q-value (probability 1-epsilon). In the rest of the cases
        a random action is chosen.
        """
        raise NotImplementedError(
            "`_choose_action` is not implemented. Please implement the function and "
            "overwrite the `Gridworld._choose_action` method."
        )

    def _arrived(self):
        """
        Check if the agent has arrived at the `reward_position`.

        Returns
        -------
        bool
            Whether the agent has arrived.
        """
        self._check_position_initialized(err_msg="Can't check if agent has arrived.")

        return (
            self.x_position == self.reward_position[0]
            and self.y_position == self.reward_position[1]
        )

    def _reward(self):
        """
        Evaluates how much reward should be administered when performing the
        chosen action at the current location

        Returns
        -------
        float
            The reward.

        Raises
        ------
        RuntimeError
            If function is called before the first state update.
        """
        if self._arrived():
            return self.reward_at_target

        # make sure there was a state update before evaluating the reward
        # (else we don't know if the last action resulted in a wall bump)
        if self._wall_touch is None:
            raise RuntimeError(
                "Can't evaluate the reward before updating at least once the state."
            )

        if self._wall_touch:
            return self.reward_at_wall
        else:
            return 0.0

    def _update_state(self):
        """
        Update the state according to the old state and the current action.
        """
        # make sure the action is set
        self._check_action_initialized(err_msg="Can't update state using action.")

        # make sure the state is initialized
        self._check_position_initialized(err_msg="Can't update state using action.")

        # remember the old position of the agent
        self.x_position_old = self.x_position
        self.y_position_old = self.y_position

        # update the agents position according to the action
        if self.action == 0:
            # move down
            self.x_position += 1
        elif self.action == 1:
            # move up
            self.x_position -= 1
        elif self.action == 2:
            # move right
            self.y_position += 1
        elif self.action == 3:
            # move left
            self.y_position -= 1
        else:
            print("There must be a bug. This is not a valid action!")

        # reset agent position if she bumped into a wall
        self._wall_touch = self._is_wall()
        if self._wall_touch:
            self.x_position = self.x_position_old
            self.y_position = self.y_position_old

        # visualize the current state
        self._visualize_current_state()

    def _is_wall(self, x_position=None, y_position=None):
        """
        This function returns whether the given position is within an obstacle.

        If you want to put the obstacle somewhere else, this is what you have to modify.
        The default is a wall that starts in the middle of the room and ends at the
        top wall.

        If no position is given, the current position of the agent is evaluated.

        Parameters
        ----------
        x_position : int or None
            X position to check. If `None`, use `self.x_position`.
        y_position : int or None
            Y position to check. If `None`, use `self.y_position`.

        Returns
        -------
        bool
            Wheter the given position is within an obstacle.

        Raises
        ------
        RuntimeError
            If `x_position`, `y_position`, `self.x_position` and `self.y_position` are
            all `None`.
        """
        if x_position is None or y_position is None:
            self._check_position_initialized(
                err_msg="Can't check if position is in wall."
            )
            x_position = self.x_position
            y_position = self.y_position

        # check if the agent is trying to leave the gridworld
        if (
            x_position < 0
            or x_position >= self.N
            or y_position < 0
            or y_position >= self.N
        ):
            return True

        # check if the agent has bumped into an obstacle in the room
        if self.obstacle:
            if y_position == int(self.N / 2) and x_position > self.N / 2:
                return True

        # if none of the above is the case, this position is not a wall
        return False

    def _visualize_current_state(self):
        """
        Show the gridworld. The squares are colored in
        red - the position of the agent - turns yellow when reaching the target or running into a wall
        blue - walls
        green - reward
        """
        if self._visualize:
            self._display = np.copy(self._display)
            # set the agents color
            # set the old position back to black
            self._display[self.x_position_old, self.y_position_old, :] = 0
            # set the new position to red
            self._display[self.x_position, self.y_position, 0] = 1
            if self._wall_touch:
                # set the current position to white if it would have ran into a wall
                self._display[self.x_position, self.y_position, :] = 1

            # update the figure
            self._append_image(self._display)

    def _init_visualization(self):
        import __main__ as main

        self._notebook = not hasattr(main, "__file__")

        self._visualize = True
        # create the figure
        self._anifig = plt.figure()
        self._aniax = self._anifig.add_subplot(1, 1, 1)
        # initialize the content of the figure (RGB at each position)
        self._anim = []
        self._display = np.zeros((self.N, self.N, 3))

        # position of the agent
        self._display[self.x_position, self.y_position, 0] = 1
        # set the reward locations
        self._display[self.reward_position[0], self.reward_position[1], [0, 1]] = 1

        for x in range(self.N):
            for y in range(self.N):
                if self._is_wall(x_position=x, y_position=y):
                    self._display[x, y, 2] = 1.0

        self._append_image(self._display)

    def _append_image(self, display):
        display = (
            self._aniax.imshow(display, interpolation="nearest", origin="lower"),
        )
        self._anim.append(display)

    def _finish_visualization(self):
        self._visualize = False
        if self._notebook:
            anim = animation.ArtistAnimation(self._anifig, self._anim, blit=True)
            return anim
        else:
            fps = 5
            _ani = animation.ArtistAnimation(
                self._anifig,
                self._anim,
                interval=1000.0 / fps,
                blit=True,
                repeat_delay=1000,
            )
            plt.show()
