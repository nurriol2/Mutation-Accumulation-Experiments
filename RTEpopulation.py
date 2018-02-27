# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import abc
import copy
import functools
import numpy as np


class rtePopulationInterface(abc.ABC):
    '''
    An abstract base class to serve as an interface for making objects to
    simulate the evolution of a population of individuals with
    retrotransposons (RTE's).
    '''

    # __repr should always be implemented to aid debugging
    @abc.abstractmethod
    def __repr__(self):
        pass

    # the full distribution is a distribution of the population over three
    # variables: # of active RTE's, # of inactive RTE's, and # of compensating
    # mutations. It should be in the form of a 3-d numpy array with axes
    # corresponding to the distribution in the preceding order along with 3
    # matching 1-D numpy arrays for each axis that give what the values of
    # the given state variable is along the given axis.
    @abc.abstractmethod
    def full_distribution(self):
        pass

    # the distribution of the population over # of active RTE's, all other
    # variables marginalized over. Should be two 1-D numpy arrays. One for the
    # possible values of # of active RTE's and one for the size of each
    # corresponding subpopulation.
    @abc.abstractmethod
    def active_rte_dist(self):
        pass

    # the mean # of active RTE's in the population. Should be #.
    @abc.abstractmethod
    def mean_active_rtes(self):
        pass

    # the distribution of the population over # of inactive RTE's, all other
    # variables marginalized over. Should be two 1-D numpy arrays. One for the
    # possible values of # of inactive RTE's and one for the size of each
    # corresponding subpopulation.
    @abc.abstractmethod
    def inactive_rte_dist(self):
        pass

    # the mean # of inactive RTE's in the population. Should be #.
    @abc.abstractmethod
    def mean_inactive_rtes(self):
        pass

    # the distribution of the population over # of compensating mutations, all
    # other variables marginalized over. Should be two 1-D numpy arrays. One
    # for the possible values of # of compensating mutations and one for the
    # size of each corresponding subpopulation.
    @abc.abstractmethod
    def compensating_mutations_dist(self):
        pass

    # the mean # of compensating mutations in the population. Should be #.
    @abc.abstractmethod
    def mean_compensating_mutations(self):
        pass

    # the distribution of the population over fitnesses.
    @abc.abstractmethod
    def fitness_dist(self):
        pass

    # the mean fitness of the population.
    @abc.abstractmethod
    def mean_fitness(self):
        pass

    # the distribution of the population over growth rates.
    @abc.abstractmethod
    def growth_rate_dist(self):
        pass

    # the mean growth rate of the population (birth - death).
    @abc.abstractmethod
    def mean_growth_rate(self):
        pass

    # the current number of individuals in the population
    @abc.abstractmethod
    def pop_size(self):
        pass

    # evolve the population forward in time.
    @abc.abstractmethod
    def update(self):
        pass


class rtePopulationIBM(rtePopulationInterface):

    def __init__(self, individual_list, fitness_function,
                 population_growth_function, time=0, alpha=1):
        self.population = {index: ind for index, ind in
                           enumerate(individual_list)}
        self.max_who = len(self.population) - 1
        self.fitness_function = self._define_indiv_fitness(fitness_function)
        self.population_growth_function = population_growth_function
        self.time = time
        self.alpha = alpha

    @classmethod
    def array_init(cls, population_distribution, min_active, min_inactive,
                   min_compensating, fitness_function,
                   population_growth_function, p_full_insert, p_partial_insert,
                   p_deactivate, p_compensate, time=0, alpha=1):
        population_distribution = np.atleast_3d(population_distribution)
        max_active = min_active + population_distribution.shape[0] - 1
        max_inactive = min_inactive + population_distribution.shape[1] - 1
        max_compensating = min_compensating + \
            population_distribution.shape[2] - 1
        ind_list = []
        for i in range(min_active, max_active + 1):
            for j in range(min_inactive, max_inactive + 1):
                for k in range(min_compensating, max_compensating + 1):
                    coordinates = (i - min_active, j - min_inactive,
                                   k - min_compensating)
                    for n in range(population_distribution[coordinates]):
                        ind = individual(i, j, k, p_full_insert,
                                         p_partial_insert, p_deactivate,
                                         p_compensate)
                        ind_list.append(ind)
        return cls(ind_list, fitness_function, population_growth_function,
                   time, alpha)

    def __repr__(self):
        return repr(self.population)

    def full_distribution(self):
        '''
        The full distribution is a distribution of the population over three
        variables: # of active RTE's, # of inactive RTE's, and # of
        compensating mutations. It should be in the form of a 3-d numpy array
        with axes # corresponding to the distribution in the preceding order
        along with 3 matching 1-D numpy arrays for each axis that give what the
        values of the given state variable is along the given axis.
        '''
        bounds = self._find_bounds_state_variables()
        active_rtes = np.arange(bounds[0], bounds[1]+1)
        inactive_rtes = np.arange(bounds[2], bounds[3]+1)
        compensating_mutations = np.arange(bounds[4], bounds[5]+1)
        distribution = np.zeros((bounds[1]+1-bounds[0],
                                 bounds[3]+1-bounds[2],
                                 bounds[5]+1-bounds[4]), dtype=int)
        for indiv in self.population.values():
            i = indiv.active - bounds[0]
            j = indiv.inactive - bounds[2]
            k = indiv.compensating - bounds[4]
            distribution[i, j, k] = distribution[i, j, k]+1
        return active_rtes, inactive_rtes, compensating_mutations, distribution

    def _find_bounds_state_variables(self):
        '''Return maximum and minimum numbers of active transposons, inactive
        transposons, and compensating mutations in the population.'''
        max_active = 0
        min_active = np.inf
        max_inactive = 0
        min_inactive = np.inf
        max_compensating = 0
        min_compensating = np.inf
        for ind in self.population.values():
            if ind.active > max_active:
                max_active = ind.active
            if ind.active < min_active:
                min_active = ind.active
            if ind.inactive > max_inactive:
                max_inactive = ind.inactive
            if ind.inactive < min_inactive:
                min_inactive = ind.inactive
            if ind.compensating > max_compensating:
                max_compensating = ind.compensating
            if ind.compensating < min_compensating:
                min_compensating = ind.compensating
        return min_active, max_active, min_inactive, max_inactive, \
            min_compensating, max_compensating

    def active_rte_dist(self):
        active, _, __, distribution = self.full_distribution()
        active_distribution = np.sum(np.sum(distribution, 1), 1)
        return active, active_distribution

    def mean_active_rtes(self):
        active, active_distribution = self.active_rte_dist()
        return np.sum(active*active_distribution)/np.sum(active_distribution)

    def inactive_rte_dist(self):
        _, inactive, __, distribution = self.full_distribution()
        inactive_distribution = np.sum(np.sum(distribution, 0), 1)
        return inactive, inactive_distribution

    def mean_inactive_rtes(self):
        inactive, inactive_distribution = self.inactive_rte_dist()
        return np.sum(inactive*inactive_distribution) / \
            np.sum(inactive_distribution)

    def compensating_mutations_dist(self):
        _, __, compensating, distribution = self.full_distribution()
        compensating_distribution = np.sum(np.sum(distribution, 0), 0)
        return compensating, compensating_distribution

    def mean_compensating_mutations(self):
        compensating, compensating_distribution = \
            self.compensating_mutations_dist()
        return np.sum(compensating*compensating_distribution) / \
            np.sum(compensating_distribution)

    def fitness_dist(self):
        fitnesses = np.array([self.fitness_function(indiv) for indiv in
                              self.population.values()])
        fitness_dist, edges = np.histogram(fitnesses, bins='auto')
        return edges[:-1], fitness_dist

    def mean_fitness(self):
        return np.mean(np.array([self.fitness_function(indiv) for indiv in
                                 self.population.values()]))

    def growth_rate_dist(self):
        fitnesses, fitness_dist = self.fitness_dist()
        r = self.mean_growth_rate()
        return fitnesses - self.mean_fitness() + r, fitness_dist

    def mean_growth_rate(self):
        r = self.population_growth_function(self.pop_size(), self.time)
        return r

    def pop_size(self):
        return len(self.population)

    def reproduce(self, who):
        indiv = self.population[who]
        new_who = self.max_who + 1
        if new_who not in self.population:
            self.population[new_who] = indiv.reproduce()
            self.population[new_who].parent = who
        else:
            raise KeyError('''this individual already exists! Error in
                           numbering scheme''')
        self.max_who = new_who

    def kill(self, who):
        return self.population.pop(who)

    def _hatch(self, who, n):
        children = [self.population[who].reproduce() for i in range(n)]
        for child in children:
            child.parent = who
        for i in range(1, n+1):
            new_who = self.max_who + i
            if new_who not in self.population:
                self.population[new_who] = children[i-1]
            else:
                raise KeyError('''this individual already exists!
                               Error in numbering scheme''')
        self.max_who = self.max_who + n

    def update(self, dt):
        '''Evolve population forward in time by dt.

        Any individuals who die will be returned so their information can be
        saved if desired.
        '''
        if not bool(self.population):  # checks if population dict is empty
            raise StopIteration('the population has no individuals in it')
        mean_f = self.mean_fitness()
        r = self.mean_growth_rate()
        self._mutations(dt)
        dead = self._deaths(dt, mean_f, r)
        self._births(dt, mean_f, r)
        dead_list = []
        for who in dead:
            dead_list.append(self.kill(who))
        self.time = self.time + dt
        return dead_list

    def _mutations(self, dt):
        for indiv in self.population.values():
            indiv.mutate(dt)

    def _deaths(self, dt, mean_f, r):
        death_dict = {}
        for who, indiv in self.population.items():
            f = self.fitness_function(indiv)
            if f - mean_f + r > 0:
                death_rate = self.alpha * dt
            else:
                death_rate = - (f - mean_f + r - self.alpha) * dt
            death_rate = min(death_rate, 1.0)
            dead = np.random.binomial(1, death_rate)
            death_dict[who] = dead
        dead_list = []
        for who, is_dead in death_dict.items():
            if is_dead:
                dead_list.append(who)
        return dead_list

    def _births(self, dt, mean_f, r):
        birth_dict = {}
        for who, indiv in self.population.items():
            f = self.fitness_function(indiv)
            if f - mean_f + r > 0:
                birth_rate = (f - mean_f + r + self.alpha) * dt
            else:
                birth_rate = self.alpha * dt
            births = np.random.poisson(birth_rate)
            birth_dict[who] = births
        for who, births in birth_dict.items():
            self._hatch(who, births)

    def _define_indiv_fitness(self, fitness_function):
        def individual_fitness(indiv, fitness_function):
            return fitness_function(active=indiv.active,
                                    inactive=indiv.inactive,
                                    compensating=indiv.compensating)
        return functools.partial(individual_fitness,
                                 fitness_function=fitness_function)


class individual(object):

    def __init__(self, active, inactive, compensating, p_full_insert,
                 p_partial_insert, p_deactivate, p_compensate, parent=None):

        if active < 0:
            raise ValueError('The number of active transposons must be >= 0.')
        self.active = int(active)

        if inactive < 0:
            raise ValueError('The number of inactive transposons must be >=0.')
        self.inactive = inactive

        if compensating < 0:
            raise ValueError('''The number of compensating mutations must be
                             >=0.''')
        if compensating > (inactive+active):
            raise ValueError('''The number of compensating mutations must be
                             less than or equal to the number of
                             retroelements.''')
        self.compensating = compensating

        if p_full_insert < 0:
            raise ValueError('p_full_insert must be >=0')
        self.p_full_insert = p_full_insert

        if p_partial_insert < 0:
            raise ValueError('p_partial_insert must be >=0')
        self.p_partial_insert = p_partial_insert

        if p_deactivate < 0:
            raise ValueError('p_deactivate must be >=0')
        self.p_deactivate = p_deactivate

        if p_compensate < 0:
            raise ValueError('p_deactivate must be >=0')
        self.p_compensate = p_compensate

        try:
            self.step_size = .5*min(1/p_compensate,
                                    .1/(p_full_insert + p_deactivate))
        except ZeroDivisionError:
            self.step_size = 1

        self.parent = parent

    def __repr__(self):
        stract = "active retroelements: " + str(self.active)
        strinact = "inactive retroelemnts: " + str(self.inactive)
        strcomp = "compensating mutations: " + str(self.compensating)
        strprobs = "p_full_insert: " + str(self.p_full_insert) + \
            ', p_partial_insert: ' + str(self.p_partial_insert) + \
            ", p_deactivate: " + str(self.p_deactivate) + \
            ', p_compensate: ' + str(self.p_compensate)
        return stract + '\n' + strinact + '\n' + strcomp + '\n' + strprobs + \
            '\n'

    def reproduce(self):
        '''individuals reproduce by making a copy of themselves'''
        return copy.deepcopy(self)

    def mutate(self, dt):
        '''Mutate the individual for time dt. dt can be large. The step size
        will be taken to be small enough to avoid egregious errors.'''
        dt_left = dt
        while dt_left > 0:
            step_size = min(dt_left, self.step_size)
            self._dmutate(step_size)
            dt_left = dt_left - step_size

    def _dmutate(self, dt):
        '''Do not call this method directly. It is for small timesteps'''
        full_inserts = np.random.poisson(self.active*self.p_full_insert*dt)
        partial_inserts = np.random.poisson(self.active *
                                            self.p_partial_insert * dt)
        deactivations = np.random.binomial(self.active, self.p_deactivate*dt)
        if (self.active + self.inactive - self.compensating) > 0:
            compensations = np.random.binomial(self.active + self.inactive -
                                               self.compensating,
                                               self.p_compensate*dt)
        else:
            compensations = 0
        self.active = self.active + full_inserts - deactivations
        self.inactive = self.inactive + partial_inserts + deactivations
        self.compensating = self.compensating + compensations


class rtePopulationArray(rtePopulationInterface):

    def __init__(self, population_distribution, min_active, min_inactive,
                 min_compensating, fitness_function,
                 population_growth_function, p_full_insert, p_partial_insert,
                 p_deactivate, p_compensate, time=0, alpha=1):

        self.population_distribution = np.atleast_3d(population_distribution)
        num_actives = self.population_distribution.shape[0]
        num_inactives = self.population_distribution.shape[1]
        num_compensatings = self.population_distribution.shape[2]
        self.actives = np.arange(min_active, min_active +
                                 num_actives).reshape(num_actives, 1, 1)
        self.inactives = np.arange(min_inactive, min_inactive +
                                   num_inactives).reshape(1, num_inactives, 1)
        self.compensatings = \
            np.arange(min_compensating, min_compensating +
                      num_compensatings).reshape(1, 1, num_compensatings)

        self.fitness_function = fitness_function
        self.population_growth_function = population_growth_function

        if p_full_insert < 0:
            raise ValueError('p_full_insert must be >=0')
        self.p_full_insert = p_full_insert

        if p_partial_insert < 0:
            raise ValueError('p_partial_insert must be >=0')
        self.p_partial_insert = p_partial_insert

        if p_deactivate < 0:
            raise ValueError('p_deactivate must be >=0')
        self.p_deactivate = p_deactivate

        if p_compensate < 0:
            raise ValueError('p_deactivate must be >=0')
        self.p_compensate = p_compensate

        self.time = time
        self.alpha = alpha

    def __repr__(self):
        actives_str = repr(self.actives)
        inactives_str = repr(self.inactives)
        compensating_str = repr(self.compensatings)
        full_distribution_str = repr(self.population_distribution)
        return_str = "active_RTE's:\n" + actives_str + "\n" + \
            "inactive_RTE's:\n" + inactives_str + "\n" + \
            "compensating mutations:\n" + compensating_str + "\n" + \
            "population distribution:\n" + full_distribution_str + "\n"
        return return_str

    def full_distribution(self):
        return self.actives, self.inactives, self.compensatings, \
            self.population_distribution

    def active_rte_dist(self):
        active_distribution = \
            np.sum(np.sum(self.population_distribution, 1), 1)
        return self.actives.flatten(), active_distribution

    def mean_active_rtes(self):
        active, active_distribution = self.active_rte_dist()
        return np.sum(active.astype('float64') * active_distribution) / \
            np.sum(active_distribution)

    def inactive_rte_dist(self):
        inactive_distribution = \
            np.sum(np.sum(self.population_distribution, 0), 1)
        return self.inactives.flatten(), inactive_distribution

    def mean_inactive_rtes(self):
        inactive, inactive_distribution = self.inactive_rte_dist()
        return np.sum(inactive.astype('float64')*inactive_distribution) / \
            np.sum(inactive_distribution)

    def compensating_mutations_dist(self):
        compensating_distribution = \
            np.sum(np.sum(self.population_distribution, 0), 0)
        return self.compensatings.flatten(), compensating_distribution

    def mean_compensating_mutations(self):
        compensating, compensating_distribution = \
            self.compensating_mutations_dist()
        return \
            np.sum(compensating.astype('float64') *
                   compensating_distribution)/np.sum(compensating_distribution)

    def fitness_dist(self):
        fitnesses = self.fitness_function(self.actives, self.inactives,
                                          self.compensatings).flatten()
        fitness_dist = self.population_distribution.flatten()
        sort_order = fitnesses.argsort()
        fitnesses = fitnesses[sort_order]
        fitness_dist = fitness_dist[sort_order]
        unique_fitnesses = []
        unique_fitness_dist = []
        current_fitness_class = fitnesses[0]
        current_total = 0
        for i, fit in enumerate(fitnesses):
            if fit != current_fitness_class:
                unique_fitnesses.append(current_fitness_class)
                unique_fitness_dist.append(current_total)
                current_total = 0
                current_fitness_class = fit
            current_total = current_total + fitness_dist[i]
        unique_fitnesses.append(current_fitness_class)
        unique_fitness_dist.append(current_total)
        return np.array(unique_fitnesses), np.array(unique_fitness_dist)

    def mean_fitness(self):
        fitnesses, fitness_distribution = self.fitness_dist()
        return np.sum(fitnesses * fitness_distribution) / \
            np.sum(fitness_distribution)

    def growth_rate_dist(self):
        fitnesses, fitness_dist = self.fitness_dist()
        r = self.mean_growth_rate()
        return fitnesses - self.mean_fitness() + r, fitness_dist

    def mean_growth_rate(self):
        r = self.population_growth_function(self.pop_size(), self.time)
        return r

    def pop_size(self):
        return np.sum(self.population_distribution)

    def update(self, dt):
        ''' Evolve the population forward in time by an increment dt.

        The timestep will be chopped up for mutations if the mutation rate is
        too high. However it won't do so for births and deaths. dt should be
        significantly less than 1 to get sensible results.
        '''
        if np.sum(self.population_distribution) == 0:
            raise StopIteration('This population has no individuals in it')
        self._mutations(dt)
        mean_f = self.mean_fitness()
        r = self.mean_growth_rate()
        births = self._births(dt, mean_f, r)
        deaths = self._deaths(dt, mean_f, r)
        self.population_distribution = self.population_distribution + \
            births - deaths
        self.time = self.time + dt

    def _mutations(self, dt):
        dt_left = dt
        while dt_left > 0:
            step_size = min(dt_left, self.step_size())
            self._dmutate(step_size)
            dt_left = dt_left - step_size

    def step_size(self):
        max_active = self.actives[-1, 0, 0]
        max_inactive = self.inactives[0, -1, 0]
        min_compensating = self.compensatings[0, 0, 0]
        step_size = \
            .05/((self.p_full_insert +
                  self.p_partial_insert +
                  self.p_deactivate) * max_active +
                 self.p_compensate *
                 (max_active+max_inactive-min_compensating))
        return step_size

    def _dmutate(self, dt):
        '''Do not call this method directly. It is for small timesteps.

        The simulation results will break in a possibly unnoticeable manner if
        dt is too big.'''
        p_active_up = self.p_full_insert*dt*self.actives
        active_increase = np.random.binomial(self.population_distribution,
                                             p_active_up)
        self.population_distribution = self.population_distribution - \
            active_increase
        active_increase = np.pad(active_increase, ((2, 0), (0, 1), (0, 1)),
                                 'constant')

        p_inactive_up = self.p_partial_insert*dt*self.actives
        inactive_increase = np.random.binomial(self.population_distribution,
                                               p_inactive_up/(1-p_active_up))
        self.population_distribution = self.population_distribution - \
            inactive_increase
        inactive_increase = np.pad(inactive_increase,
                                   ((1, 1), (1, 0), (0, 1)), 'constant')

        p_deactivate_up = self.p_deactivate*dt*self.actives
        deactivate_increase = \
            np.random.binomial(self.population_distribution,
                               p_deactivate_up/(1-p_active_up-p_inactive_up))
        self.population_distribution = \
            self.population_distribution - deactivate_increase
        deactivate_increase = np.pad(deactivate_increase,
                                     ((0, 2), (1, 0), (0, 1)), 'constant')

        p_compensate_up = \
            np.maximum(self.p_compensate * dt *
                       (self.actives+self.inactives-self.compensatings), 0)
        compensate_increase = \
            np.random.binomial(self.population_distribution, p_compensate_up /
                               (1-p_active_up-p_inactive_up-p_deactivate_up))
        self.population_distribution = \
            self.population_distribution - compensate_increase
        compensate_increase = np.pad(compensate_increase,
                                     ((1, 1), (0, 1), (1, 0)), 'constant')

        self.actives = \
            np.pad(self.actives,
                   ((1, 1), (0, 0), (0, 0)),
                   'linear_ramp',
                   end_values=((self.actives[0, 0, 0]-1,
                                self.actives[-1, 0, 0]+1),
                               (0, 0),
                               (0, 0)))
        self.inactives = \
            np.pad(self.inactives,
                   ((0, 0), (0, 1), (0, 0)),
                   'linear_ramp',
                   end_values=((0, 0),
                               (0, self.inactives[0, -1, 0]+1),
                               (0, 0)))
        self.compensatings = \
            np.pad(self.compensatings,
                   ((0, 0), (0, 0), (0, 1)),
                   'linear_ramp',
                   end_values=((0, 0),
                               (0, 0),
                               (0, self.compensatings[0, 0, -1]+1)))
        self.population_distribution = \
            np.pad(self.population_distribution,
                   ((1, 1), (0, 1), (0, 1)), 'constant')
        self.population_distribution = self.population_distribution + \
            active_increase + inactive_increase + deactivate_increase + \
            compensate_increase
        self._trim_updates()

    #removes unnecessary zeros padding out the distribution matrix
    def _trim_updates(self):
        while np.sum(self.population_distribution[0, :, :]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, 0, 0)
            self.actives = np.delete(self.actives, 0, 0)
        while np.sum(self.population_distribution[-1, :, :]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, -1, 0)
            self.actives = np.delete(self.actives, -1, 0)
        while np.sum(self.population_distribution[:, 0, :]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, 0, 1)
            self.inactives = np.delete(self.inactives, 0, 1)
        while np.sum(self.population_distribution[:, -1, :]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, -1, 1)
            self.inactives = np.delete(self.inactives, -1, 1)
        while np.sum(self.population_distribution[:, :, 0]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, 0, 2)
            self.compensatings = np.delete(self.compensatings, 0, 2)
        while np.sum(self.population_distribution[:, :, -1]) == 0:
            self.population_distribution = \
                np.delete(self.population_distribution, -1, 2)
            self.compensatings = np.delete(self.compensatings, -1, 2)

    def _births(self, dt, mean_f, r):
        fitnesses = self.fitness_function(self.actives, self.inactives,
                                          self.compensatings)
        test = fitnesses - mean_f + r
        high_f = (test > 0) * self.population_distribution
        low_f = (test <= 0) * self.population_distribution
        high_births = np.random.poisson(high_f * (test + self.alpha) *
                                        (test > 0) * dt)
        low_births = np.random.poisson(low_f*self.alpha*dt)
        return high_births + low_births

    def _deaths(self, dt, mean_f, r):
        fitnesses = self.fitness_function(self.actives, self.inactives,
                                          self.compensatings)
        test = fitnesses - mean_f + r
        high_f = (test > 0) * self.population_distribution
        low_f = (test <= 0) * self.population_distribution
        high_deaths = np.random.binomial(high_f, self.alpha*dt)
        low_deaths = np.random.binomial(low_f, - (test - self.alpha) *
                                        (test <= 0) * dt)
        return high_deaths + low_deaths


def test_fitness_function(active, inactive, compensating):
    return -.01 * (active + inactive - compensating)


def test_growth_rate_function(N, t):
    return (1 - N/1000)
