# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import abc
import copy
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

    # evolve the population forward in time.
    @abc.abstractmethod
    def update(self):
        pass


class rtePopulationIBM(rtePopulationInterface):
    

    def __init__(self, individual_list, fitness_function,
                 population_growth_function):
        self.population = {index: ind for index, ind in
                           enumerate(individual_list)}
        self.max_who = len(self.population) - 1
        self.fitness_function = fitness_function
        self.population_growth_function = population_growth_function
        self.time = 0
        self.alpha = .5

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
        r = self.population_growth_function(len(self.population), self.time)
        return r

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

    def hatch(self, who, n):
        children = [self.population[who].reproduce() for i in range(n)]
        for child in children:
            child.parent = who
        for i in range(1,n+1):
            new_who = self.max_who + i
            if new_who not in self.population:
                self.population[new_who] = children[i-1]
            else:
                raise KeyError('''this individual already exists!
                               Error in numbering scheme''')
        self.max_who = self.max_who + n

    def update(self, dt):
        '''Evolve population forward in time by dt. Any individuals who die
        will be returned so their information can be saved if desired.'''
        mean_f = self.mean_fitness()
        r = self.mean_growth_rate()
        self._mutations(dt)
        dead = self._deaths(dt, mean_f, r)
        self._births(dt, mean_f, r)
        self.time = self.time + dt
        return dead

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
            death_dict[who]=dead
        dead_list =[]
        for who, is_dead in death_dict.items():
            if is_dead:
                dead_list.append(self.kill(who))
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
            birth_dict[who]=births
        for who, births in birth_dict.items():
            self.hatch(who, births)


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

        self.step_size = .5*min(1/p_compensate,
                                .1/(p_full_insert + p_deactivate))

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
        partial_inserts = np.random.poisson(self.active*self.p_partial_insert
                                            * dt)
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

def test_fitness_function(indiv):
    return -.01 * (indiv.active + indiv.inactive - indiv.compensating)

def test_growth_rate_function(N, t):
    return (1 - N/1000)
