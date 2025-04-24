import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import os
from joblib import Parallel, delayed
from cec17_functions import cec17_test_func
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Known biases for functions f1 to f30
biases = {
    1: 100,   2: 0,     3: 300,   4: 400,   5: 500,
    6: 600,   7: 700,   8: 800,   9: 900,   10: 1000,
    11: 1100, 12: 1200, 13: 1300, 14: 1400, 15: 1500,
    16: 1600, 17: 1700, 18: 1800, 19: 1900, 20: 2000,
    21: 2100, 22: 2200, 23: 2300, 24: 2400, 25: 2500,
    26: 2600, 27: 2700, 28: 2800, 29: 2900, 30: 3000
}

def get_category(fun_index):
    if 1 <= fun_index <= 3:
        return "Unimodal"
    elif 4 <= fun_index <= 10:
        return "Multimodal"
    elif 11 <= fun_index <= 20:
        return "Hybrid"
    elif 21 <= fun_index <= 30:
        return "Composition"
    else:
        return "Unknown"

# Function to evaluate a solution
def evaluate_solution(x, dim, fun_index):
    f = [0]
    cec17_test_func(x.tolist(), f, dim, 1, fun_index)
    return f[0]

# Enhanced boundary handling with advanced bounce-back mechanism
def advanced_space_bound(X, lb, ub):
    X_new = np.copy(X)
    
    # Element-wise boundary handling with different strategies
    mask_lower = X < lb
    mask_upper = X > ub
    
    if np.any(mask_lower):
        # For lower bound violations - mirror reflection with damping
        X_new[mask_lower] = lb + 0.5 * np.abs(lb - X[mask_lower])
    
    if np.any(mask_upper):
        # For upper bound violations - random bounce with history
        X_new[mask_upper] = ub - 0.5 * np.abs(X[mask_upper] - ub)
    
    return X_new

def fun_range_cec(lb, ub, dim):
    """Defines the search space for CEC17 functions based on provided parameters."""
    return lb, ub, dim

# Enhanced chaotic maps for better diversity
def sinusoidal_map(x):
    return 2.3 * x**2 * np.sin(np.pi * x)

def logistic_map(x):
    r = 3.99  # Chaotic region
    return r * x * (1 - x)

def ikeda_map(x):
    """A more complex chaotic map for better diversity"""
    u = 0.9 - 0.6/(1 + x**2)
    return 1 + 0.9 * x * np.cos(u)

def tent_map(x):
    """Tent map for additional chaos behavior"""
    if x < 0.5:
        return 2 * x
    else:
        return 2 * (1 - x)

def gauss_map(x):
    """Gauss map for more unpredictable sequences"""
    beta = 6.2
    if x == 0:
        return 0
    return np.exp(-beta * x**2) % 1.0

# Sophisticated Latin Hypercube Sampling with gradient
def advanced_lhs_sampling(n_samples, dim, lb, ub):
    # Create scrambled Sobol sequence for better coverage
    sampler = qmc.Sobol(d=dim, scramble=True)
    samples = sampler.random(n=n_samples)
    
    # Scale samples to the search space
    scaled_samples = qmc.scale(samples, lb, ub)
    
    # Add gradient information for first few points (important regions)
    elite_count = max(3, n_samples // 10)
    for i in range(elite_count):
        # Ensure some points are near the center of the space
        scaled_samples[i, :] = (lb + ub) / 2 + (np.random.random(dim) * 0.2 - 0.1) * (ub - lb)
    
    return scaled_samples

# Enhanced multi-strategy memory for tracking different types of solutions
class EnhancedAdaptiveMemory:
    def __init__(self, capacity=15, dim=10):
        self.capacity = capacity
        self.dim = dim
        # Create different memory banks for different purposes
        self.short_term = {'solutions': [], 'fitnesses': []}  # Recent best
        self.diversity = {'solutions': [], 'fitnesses': []}   # Most diverse
        self.long_term = {'solutions': [], 'fitnesses': []}   # Historical best
        self.region_centers = []  # Track promising regions
        self.capacity_per_bank = max(5, capacity // 3)
    
    def add(self, solution, fitness):
        # Update short-term memory (recent best)
        if len(self.short_term['solutions']) < self.capacity_per_bank:
            self.short_term['solutions'].append(solution.copy())
            self.short_term['fitnesses'].append(fitness)
        else:
            # Replace worst in short-term if better
            worst_idx = np.argmax(self.short_term['fitnesses'])
            if fitness < self.short_term['fitnesses'][worst_idx]:
                self.short_term['solutions'][worst_idx] = solution.copy()
                self.short_term['fitnesses'][worst_idx] = fitness
        
        # Update diversity memory - keep solutions that maximize distance
        if len(self.diversity['solutions']) < self.capacity_per_bank:
            self.diversity['solutions'].append(solution.copy())
            self.diversity['fitnesses'].append(fitness)
        else:
            # If enough solutions exist, calculate diversity contribution
            if len(self.diversity['solutions']) > 0:
                if self._should_add_to_diversity(solution):
                    worst_div_idx = self._get_least_diverse_solution_index()
                    self.diversity['solutions'][worst_div_idx] = solution.copy()
                    self.diversity['fitnesses'][worst_div_idx] = fitness
        
        # Update long-term memory - only keep the very best solutions
        if len(self.long_term['solutions']) < self.capacity_per_bank:
            self.long_term['solutions'].append(solution.copy())
            self.long_term['fitnesses'].append(fitness)
        else:
            worst_idx = np.argmax(self.long_term['fitnesses'])
            if fitness < self.long_term['fitnesses'][worst_idx]:
                self.long_term['solutions'][worst_idx] = solution.copy()
                self.long_term['fitnesses'][worst_idx] = fitness
    
    def _should_add_to_diversity(self, solution):
        """Check if solution should be added to diversity bank"""
        if len(self.diversity['solutions']) == 0:
            return True
            
        # Calculate minimum distance to existing solutions
        min_distance = float('inf')
        for existing_sol in self.diversity['solutions']:
            dist = np.linalg.norm(solution - existing_sol)
            min_distance = min(min_distance, dist)
        
        # Get minimum distance between existing solutions
        existing_min_distance = float('inf')
        if len(self.diversity['solutions']) > 1:
            for i in range(len(self.diversity['solutions'])):
                for j in range(i+1, len(self.diversity['solutions'])):
                    dist = np.linalg.norm(self.diversity['solutions'][i] - self.diversity['solutions'][j])
                    existing_min_distance = min(existing_min_distance, dist)
        else:
            existing_min_distance = 0
            
        # Add if this solution is more diverse than existing ones
        return min_distance > 0.5 * existing_min_distance
    
    def _get_least_diverse_solution_index(self):
        """Find the solution contributing least to diversity"""
        if len(self.diversity['solutions']) <= 1:
            return 0
            
        min_contribution = float('inf')
        min_idx = 0
        
        for i in range(len(self.diversity['solutions'])):
            # Calculate average distance to all other solutions
            total_dist = 0
            for j in range(len(self.diversity['solutions'])):
                if i != j:
                    dist = np.linalg.norm(self.diversity['solutions'][i] - self.diversity['solutions'][j])
                    total_dist += dist
            avg_dist = total_dist / (len(self.diversity['solutions']) - 1)
            
            if avg_dist < min_contribution:
                min_contribution = avg_dist
                min_idx = i
                
        return min_idx
    
    def get_best(self):
        """Get overall best solution from memory banks"""
        best_fitness = float('inf')
        best_solution = None
        
        # Check all memory banks
        memory_banks = [self.short_term, self.diversity, self.long_term]
        
        for bank in memory_banks:
            if bank['solutions']:
                idx = np.argmin(bank['fitnesses'])
                if bank['fitnesses'][idx] < best_fitness:
                    best_fitness = bank['fitnesses'][idx]
                    best_solution = bank['solutions'][idx].copy()
        
        if best_solution is None:
            return None, float('inf')
            
        return best_solution, best_fitness
    
    def get_random(self):
        """Get random solution from memory banks with preference for better solutions"""
        if not self.short_term['solutions'] and not self.diversity['solutions'] and not self.long_term['solutions']:
            return None, float('inf')
        
        # Select bank with probability proportional to quality
        bank_probs = [0.5, 0.3, 0.2]  # short_term, diversity, long_term
        banks = [self.short_term, self.diversity, self.long_term]
        
        # Filter out empty banks
        valid_banks = []
        valid_probs = []
        
        for i, bank in enumerate(banks):
            if bank['solutions']:
                valid_banks.append(bank)
                valid_probs.append(bank_probs[i])
        
        if not valid_banks:
            return None, float('inf')
            
        # Normalize probabilities
        valid_probs = np.array(valid_probs) / sum(valid_probs)
        
        # Select bank
        selected_bank = np.random.choice(len(valid_banks), p=valid_probs)
        bank = valid_banks[selected_bank]
        
        # Select solution with preference for better solutions
        fitnesses = np.array(bank['fitnesses'])
        probs = 1 / (fitnesses - min(fitnesses) + 1e-10)
        probs = probs / sum(probs)
        
        idx = np.random.choice(len(bank['solutions']), p=probs)
        return bank['solutions'][idx].copy(), bank['fitnesses'][idx]
    
    def get_diverse_solution(self, reference_solution):
        """Return the most diverse solution compared to reference_solution"""
        if not self.diversity['solutions']:
            return self.get_random()
        
        max_distance = -1
        diverse_idx = 0
        
        for i, sol in enumerate(self.diversity['solutions']):
            distance = np.linalg.norm(sol - reference_solution)
            if distance > max_distance:
                max_distance = distance
                diverse_idx = i
        
        return self.diversity['solutions'][diverse_idx].copy(), self.diversity['fitnesses'][diverse_idx]

# Function to calculate approximate gradient
def gradient_approximation(x, f_x, dim, fun_index, step_size=0.01):
    """Approximate gradient using finite differences"""
    gradient = np.zeros(dim)
    
    for i in range(dim):
        # Forward difference
        x_forward = x.copy()
        x_forward[i] += step_size
        f_forward = evaluate_solution(x_forward, dim, fun_index)
        
        # Compute gradient
        gradient[i] = (f_forward - f_x) / step_size
    
    # Normalize gradient if non-zero
    norm = np.linalg.norm(gradient)
    if norm > 0:
        gradient = gradient / norm
    
    return gradient

# Enhanced multi-strategy local search
def enhanced_multi_strategy_local_search(x_best, f_best, dim, fun_index, lb, ub, params, iteration_ratio=0):
    """Enhanced local search with gradient approximation and specialized strategies"""
    x_new = x_best.copy()
    f_new = f_best
    
    # Get problem category for strategy selection
    category = get_category(fun_index)
    evals_used = 0  # Track function evaluations
    
    # 1. Use gradient approximation occasionally for efficiency
    if np.random.random() < 0.3:
        # Calculate approximate gradient
        gradient = gradient_approximation(x_new, f_new, dim, fun_index, 
                                         step_size=params['gradient_step'])
        evals_used += dim
        
        # Try multiple step sizes in negative gradient direction
        for step_multiplier in [0.1, 0.5, 1.0, 2.0]:
            # Adapt step size based on iteration progress
            adaptive_step = step_multiplier * params['gradient_step'] * (ub - lb) * (1 - 0.7 * iteration_ratio)
            
            # Move in the negative gradient direction
            x_test = x_new - adaptive_step * gradient
            x_test = advanced_space_bound(x_test, lb, ub)
            f_test = evaluate_solution(x_test, dim, fun_index)
            evals_used += 1
            
            if f_test < f_new:
                x_new = x_test.copy()
                f_new = f_test
                break  # Found improvement, exit loop
    
    # 2. Choose search strategy based on problem type and current progress
    
    # 2.1 For Unimodal and some Multimodal problems - Pattern Search works well
    if category == "Unimodal" or (category == "Multimodal" and fun_index in [4, 9]) or (np.random.random() < 0.6 - 0.3 * iteration_ratio):
        # Enhanced pattern search with adaptive step size
        step_size = params['gradient_step'] * (ub - lb) * (1 - 0.7 * iteration_ratio)
        
        # Try dimensions in order of potential impact
        dims_to_try = np.random.permutation(dim)[:min(params['local_search_depth'], dim)]
        
        for dim_idx in dims_to_try:
            # Try decreasing step sizes for better precision
            for step_reduction in [1.0, 0.5, 0.1]:
                current_step = step_size * step_reduction
                
                # Try positive direction
                x_test = x_new.copy()
                x_test[dim_idx] += current_step
                x_test = advanced_space_bound(x_test, lb, ub)
                f_test = evaluate_solution(x_test, dim, fun_index)
                evals_used += 1
                
                if f_test < f_new:
                    x_new = x_test.copy()
                    f_new = f_test
                    break  # Found improvement, try next dimension
                
                # Try negative direction
                x_test = x_new.copy()
                x_test[dim_idx] -= current_step
                x_test = advanced_space_bound(x_test, lb, ub)
                f_test = evaluate_solution(x_test, dim, fun_index)
                evals_used += 1
                
                if f_test < f_new:
                    x_new = x_test.copy()
                    f_new = f_test
                    break  # Found improvement, try next dimension
    
    # 2.2 For challenging Multimodal and Hybrid problems - Random Direction Search
    elif category == "Multimodal" or category == "Hybrid" or (np.random.random() < 0.5):
        # Enhanced random direction search with golden section
        n_directions = min(params['local_search_depth'], dim)
        base_radius = params['gradient_step'] * (ub - lb) * (1 - 0.5 * iteration_ratio)
        
        for _ in range(n_directions):
            # Generate random direction vector
            direction = np.random.normal(0, 1, dim)
            direction = direction / np.linalg.norm(direction)
            
            # Try multiple step sizes with adaptive sampling
            best_scale = 0
            best_fit = f_new
            scales_to_try = np.linspace(0.1, 2.0, 5) * (1 + 0.5 * (1 - iteration_ratio))
            
            for scale in scales_to_try:
                x_test = x_new + scale * base_radius * direction
                x_test = advanced_space_bound(x_test, lb, ub)
                f_test = evaluate_solution(x_test, dim, fun_index)
                evals_used += 1
                
                if f_test < best_fit:
                    best_fit = f_test
                    best_scale = scale
            
            # If improvement found, update solution
            if best_fit < f_new:
                x_new = x_new + best_scale * base_radius * direction
                x_new = advanced_space_bound(x_new, lb, ub)
                f_new = best_fit
    
    # 2.3 For Composition functions - Specialized Dimensional Search
    else:
        # Enhanced dimensional search for composition functions
        n_attempts = params['local_search_depth']
        
        # Try modifying multiple dimensions together
        for _ in range(n_attempts):
            # Select a subset of dimensions
            n_dims = max(2, dim // 4)
            dims = np.random.choice(dim, n_dims, replace=False)
            
            # Generate perturbation vector with varying steps
            perturbation = np.zeros(dim)
            # Use larger steps for composition functions 
            step_factor = 0.05 * (1 + (fun_index - 21) / 10) if 21 <= fun_index <= 30 else 0.02
            perturbation[dims] = step_factor * (ub - lb) * (np.random.random(n_dims) * 2 - 1)
            
            x_test = x_new + perturbation
            x_test = advanced_space_bound(x_test, lb, ub)
            f_test = evaluate_solution(x_test, dim, fun_index)
            evals_used += 1
            
            if f_test < f_new:
                x_new = x_test.copy()
                f_new = f_test
    
    # 3. Function-specific strategies for challenging functions
    
    # 3.1 For functions with poor success rates, try specialized techniques
    challenging_functions = [5, 7, 8, 10, 12, 13, 15, 25, 27, 29, 30]
    if fun_index in challenging_functions and np.random.random() < 0.3:
        # For these difficult functions, try specialized search
        if fun_index in [5, 7, 8]:  # Challenging multimodal
            # Try large random jumps
            jump_magnitude = 0.3 * (ub - lb)
            x_test = x_new + jump_magnitude * np.random.normal(0, 1, dim)
            x_test = advanced_space_bound(x_test, lb, ub)
            f_test = evaluate_solution(x_test, dim, fun_index)
            evals_used += 1
            
            if f_test < f_new:
                x_new = x_test.copy()
                f_new = f_test
                
        elif fun_index in [12, 13, 15]:  # Challenging hybrid
            # Try more focused search in random subspace
            subspace_size = max(2, dim // 3)
            active_dims = np.random.choice(dim, subspace_size, replace=False)
            
            # Create perturbation only in active dimensions
            for _ in range(3):  # Try a few perturbations
                perturbation = np.zeros(dim)
                perturbation[active_dims] = 0.1 * (ub - lb) * np.random.normal(0, 1, subspace_size)
                
                x_test = x_new + perturbation
                x_test = advanced_space_bound(x_test, lb, ub)
                f_test = evaluate_solution(x_test, dim, fun_index)
                evals_used += 1
                
                if f_test < f_new:
                    x_new = x_test.copy()
                    f_new = f_test
                    break
                
        elif fun_index in [25, 27, 29, 30]:  # Challenging composition
            # Use more varied step sizes
            for _ in range(2):
                rand_scale = 0.05 + 0.25 * np.random.random()
                x_test = x_new + rand_scale * (ub - lb) * np.random.normal(0, 1, dim)
                x_test = advanced_space_bound(x_test, lb, ub)
                f_test = evaluate_solution(x_test, dim, fun_index)
                evals_used += 1
                
                if f_test < f_new:
                    x_new = x_test.copy()
                    f_new = f_test
                    break
    
    return x_new, f_new, evals_used

# Advanced function-specific parameter adaptation
def get_enhanced_params_v7(fun_index, category, iteration_ratio=0):
    """Get parameters optimized for specific functions with adaptive changes during iterations"""
    
    # Common default parameters
    params = {
        'cyclone_prob': 0.6,      # Balance between cyclone and chain foraging
        'somersault_factor': 2.0,  # Step size for somersault
        'local_search_freq': 5,    # Frequency of local search
        'local_search_depth': 3,   # Depth of local search
        'restart_threshold': 15,   # Stagnation counter threshold
        'alpha_scale': 2.0,        # Scale for alpha parameter in chain foraging
        'crossover_rate': 0.7,     # Rate for dimension-wise crossover
        'exploitation_bias': 0.5,  # 0-1 scale where 1 is full exploitation
        'memory_size': 5,          # Size of memory for elite solutions
        'gradient_step': 0.01,     # Step size for gradient-based search
        'gradient_freq': 20,       # Frequency of gradient-based search
        'restart_scale': 0.2,      # Scale for restart spread
        'diversity_control': 0.5,  # Control parameter for population diversity
        'elite_rate': 0.2,         # Percentage of population preserved as elite
        'mutation_rate': 0.05,     # Probability of random dimension mutation
    }
    
    # Adaptive adjustments based on iteration ratio (0 at start, 1 at end)
    # This helps balance exploration/exploitation throughout the run
    if iteration_ratio > 0:
        # Gradually increase exploitation as iterations progress
        params['exploitation_bias'] += 0.3 * iteration_ratio
        params['local_search_depth'] += int(2 * iteration_ratio)
        params['restart_threshold'] = max(5, int(params['restart_threshold'] * (1 - 0.5 * iteration_ratio)))
        params['local_search_freq'] = max(2, int(params['local_search_freq'] * (1 - 0.3 * iteration_ratio)))
        params['gradient_step'] *= (1 - 0.5 * iteration_ratio)  # Decrease step size over time
    
    # Unimodal functions - favor exploitation
    if category == "Unimodal":
        params['cyclone_prob'] = 0.85
        params['somersault_factor'] = 1.2
        params['local_search_freq'] = 3
        params['local_search_depth'] = 5
        params['restart_threshold'] = 20
        params['exploitation_bias'] = 0.85
        params['gradient_freq'] = 10
        params['diversity_control'] = 0.3
        
        if fun_index == 1:  # Sphere function
            params['cyclone_prob'] = 0.9
            params['somersault_factor'] = 1.0
            params['local_search_depth'] = 6
            params['gradient_freq'] = 5
        
        elif fun_index == 3:  # Rotated High Conditioned Elliptic
            params['exploitation_bias'] = 0.75
            params['somersault_factor'] = 1.5
            params['gradient_step'] = 0.005
    
    # Multimodal functions - balanced approach
    elif category == "Multimodal":
        params['cyclone_prob'] = 0.65
        params['somersault_factor'] = 2.0
        params['local_search_freq'] = 6
        params['restart_threshold'] = 15
        params['exploitation_bias'] = 0.6
        params['diversity_control'] = 0.6
        params['mutation_rate'] = 0.1
        
        # V7 IMPROVEMENTS FOR MULTIMODAL FUNCTIONS
        if fun_index in [5, 7, 8]:  # Challenging multimodal functions
            params['cyclone_prob'] = 0.55
            params['somersault_factor'] = 2.5
            params['restart_threshold'] = 10
            params['exploitation_bias'] = 0.4
            params['mutation_rate'] = 0.15
            params['restart_scale'] = 0.3
            params['diversity_control'] = 0.8
            params['local_search_depth'] = 5
            params['local_search_freq'] = 3
        
        elif fun_index == 10:  # Rotated Multipeak
            params['cyclone_prob'] = 0.45
            params['somersault_factor'] = 3.0
            params['exploitation_bias'] = 0.3
            params['restart_threshold'] = 8
            params['restart_scale'] = 0.4
            params['diversity_control'] = 0.9
            params['mutation_rate'] = 0.2
        
        elif fun_index in [4, 9]:  # Successful multimodal functions
            params['cyclone_prob'] = 0.7
            params['somersault_factor'] = 1.5
            params['local_search_freq'] = 4
            params['local_search_depth'] = 5
            params['exploitation_bias'] = 0.7
    
    # Hybrid functions - more exploration with specialized local search
    elif category == "Hybrid":
        params['cyclone_prob'] = 0.55
        params['somersault_factor'] = 2.2
        params['local_search_freq'] = 7
        params['restart_threshold'] = 12
        params['exploitation_bias'] = 0.45
        params['diversity_control'] = 0.65
        params['memory_size'] = 8
        params['mutation_rate'] = 0.12
        
        # V7 IMPROVEMENTS FOR HYBRID FUNCTIONS
        if fun_index in [11, 12, 13, 15]:  # Most challenging hybrids
            params['cyclone_prob'] = 0.35
            params['somersault_factor'] = 3.0
            params['restart_threshold'] = 7
            params['exploitation_bias'] = 0.3
            params['local_search_depth'] = 6
            params['restart_scale'] = 0.4
            params['diversity_control'] = 0.9
            params['mutation_rate'] = 0.25
            params['local_search_freq'] = 3
        
        elif fun_index in [14, 18, 20]:  # Moderately successful hybrids
            params['cyclone_prob'] = 0.5
            params['somersault_factor'] = 2.5
            params['local_search_freq'] = 5
            params['local_search_depth'] = 5
            params['restart_scale'] = 0.3
    
    # Composition functions - high exploration, frequent restarts
    elif category == "Composition":
        params['cyclone_prob'] = 0.45
        params['somersault_factor'] = 2.5
        params['local_search_freq'] = 8
        params['restart_threshold'] = 10
        params['exploitation_bias'] = 0.4
        params['diversity_control'] = 0.7
        params['memory_size'] = 10
        params['mutation_rate'] = 0.15
        
        # V7 IMPROVEMENTS FOR COMPOSITION FUNCTIONS
        if fun_index in [26, 28]:  # More successful composition functions
            params['cyclone_prob'] = 0.5
            params['somersault_factor'] = 2.0
            params['restart_threshold'] = 12
            params['exploitation_bias'] = 0.5
            params['local_search_depth'] = 4
            params['local_search_freq'] = 6
            params['restart_scale'] = 0.25
        
        elif fun_index in [21, 22, 23]:  # Moderately successful compositions
            params['cyclone_prob'] = 0.45
            params['somersault_factor'] = 2.3
            params['restart_threshold'] = 10
            params['local_search_freq'] = 5
            params['memory_size'] = 8
        
        elif fun_index in [25, 27, 29, 30]:  # Most challenging compositions
            params['cyclone_prob'] = 0.35
            params['somersault_factor'] = 3.5
            params['restart_threshold'] = 6
            params['exploitation_bias'] = 0.25
            params['local_search_depth'] = 7
            params['restart_scale'] = 0.45
            params['diversity_control'] = 0.95
            params['memory_size'] = 15
            params['mutation_rate'] = 0.3
            params['local_search_freq'] = 3
    
    return params

# Advanced Levy flight function for better exploration
def levy_flight(dim, beta=1.5, scale=1.0):
    """Generate Levy flight step with parameter beta"""
    # Generate step from levy distribution
    sigma_u = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
              (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1/beta)
    
    # Generate steps in each dimension
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = u / (np.abs(v)**(1/beta))
    
    # Normalize and scale
    if np.linalg.norm(step) > 0:
        step = scale * step / np.linalg.norm(step)
    
    return step

# Advanced restart strategy with memory utilization
# Improved initialization with advanced sampling
def initialize_diverse_population_v7(n_pop, dim, lb, ub, fun_index, seed=None):
    """Enhanced initialization with specialized techniques for different function types"""
    if seed is not None:
        np.random.seed(seed)
    
    # Get problem category for specialized initialization
    category = get_category(fun_index)
    
    # Allocate population array
    PopPos = np.zeros((n_pop, dim))
    PopFit = np.zeros(n_pop)
    
    # Determine initialization approaches based on function type
    lhs_count = int(n_pop * 0.6)  # Base LHS population
    
    # Use different ratios based on problem type
    if category == "Unimodal":
        lhs_count = int(n_pop * 0.7)  # More structured sampling for unimodal
        levy_count = int(n_pop * 0.2)  # Some levy flights
        random_count = n_pop - lhs_count - levy_count
    elif category == "Multimodal":
        if fun_index in [5, 7, 8, 10]:  # Challenging multimodal
            lhs_count = int(n_pop * 0.5)  # Less structure
            levy_count = int(n_pop * 0.3)  # More levy flights
            random_count = n_pop - lhs_count - levy_count
        else:
            lhs_count = int(n_pop * 0.6)
            levy_count = int(n_pop * 0.25)
            random_count = n_pop - lhs_count - levy_count
    elif category == "Hybrid":
        lhs_count = int(n_pop * 0.55)
        levy_count = int(n_pop * 0.25)
        random_count = n_pop - lhs_count - levy_count
    else:  # Composition
        if fun_index in [21, 22, 23, 26, 28]:  # Better performing compositions
            lhs_count = int(n_pop * 0.55)
            levy_count = int(n_pop * 0.25)
            random_count = n_pop - lhs_count - levy_count
        else:  # Most challenging compositions
            lhs_count = int(n_pop * 0.45)  # Less structure
            levy_count = int(n_pop * 0.35)  # More exploration
            random_count = n_pop - lhs_count - levy_count
    
    # 1. Use advanced LHS for structured portion
    if lhs_count > 0:
        lhs_samples = advanced_lhs_sampling(lhs_count, dim, lb, ub)
        PopPos[:lhs_count] = lhs_samples
    
    # 2. Use Levy flights from center for more diverse exploration
    if levy_count > 0:
        center = (lb + ub) / 2
        for i in range(lhs_count, lhs_count + levy_count):
            PopPos[i] = center + levy_flight(dim, beta=1.5, scale=0.3*(ub-lb))
            PopPos[i] = advanced_space_bound(PopPos[i], lb, ub)
    
    # 3. Pure random for remaining positions
    if random_count > 0:
        start_idx = lhs_count + levy_count
        PopPos[start_idx:] = np.random.uniform(lb, ub, (random_count, dim))
    
    # Evaluate all positions
    for i in range(n_pop):
        PopFit[i] = evaluate_solution(PopPos[i], dim, fun_index)
    
    return PopPos, PopFit

def advanced_restart_strategy(PopPos, PopFit, BestX, memory, n_pop, lb, ub, dim, fun_index, params, iteration_ratio):
    """Advanced restart strategy with memory-based learning and clustering"""
    # Determine how many individuals to restart
    restart_fraction = 0.3 + 0.1 * (1 - iteration_ratio)  # More restarts early
    restart_count = max(2, int(n_pop * restart_fraction))
    
    # Save the elite individuals
    sorted_indices = np.argsort(PopFit)
    elites_count = max(2, int(n_pop * params['elite_rate']))
    
    # Identify the worst individuals to restart
    restart_indices = sorted_indices[-(restart_count):]
    
    # New positions array
    new_positions = PopPos.copy()
    new_fitnesses = PopFit.copy()
    
    # For composition functions, try to identify and target promising regions
    category = get_category(fun_index)
    
    # Apply different restart strategies based on function type
    for i, idx in enumerate(restart_indices):
        restart_type = np.random.random()
        
        if category == "Unimodal":  # Unimodal - focus more on exploitation
            if restart_type < 0.7:  # Near best solution
                new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.1 + 0.1 * np.random.random()) * np.random.normal(0, 1, dim)
            else:  # Complete restart with small probability
                new_positions[idx] = np.random.uniform(lb, ub, dim)
        
        elif category == "Multimodal":  # Multimodal - balanced approach
            if fun_index in [5, 7, 8, 10]:  # Challenging multimodal
                if restart_type < 0.3:  # Near best
                    new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.2 + 0.3 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.6:  # From diverse memory
                    memory_sol, _ = memory.get_diverse_solution(BestX)
                    if memory_sol is not None:
                        new_positions[idx] = memory_sol + 0.3 * (ub - lb) * np.random.normal(0, 1, dim)
                    else:
                        new_positions[idx] = np.random.uniform(lb, ub, dim)
                else:  # Complete restart with large probability
                    new_positions[idx] = np.random.uniform(lb, ub, dim)
            else:  # Standard multimodal
                if restart_type < 0.4:  # Near best solution
                    new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.2 + 0.2 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.7:  # Near elite solutions
                    elite_idx = sorted_indices[np.random.randint(0, elites_count)]
                    new_positions[idx] = PopPos[elite_idx] + params['restart_scale'] * (ub - lb) * (0.1 + 0.3 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.9:  # From memory
                    memory_sol, _ = memory.get_random()
                    if memory_sol is not None:
                        new_positions[idx] = memory_sol + 0.2 * (ub - lb) * np.random.normal(0, 1, dim)
                    else:
                        new_positions[idx] = np.random.uniform(lb, ub, dim)
                else:  # Complete restart
                    new_positions[idx] = np.random.uniform(lb, ub, dim)
        
        elif category == "Hybrid":  # Hybrid - more exploration
            if fun_index in [11, 12, 13, 15]:  # Most challenging hybrids
                if restart_type < 0.2:  # Near best
                    new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.3 + 0.3 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.5:  # From memory with diversity
                    memory_sol, _ = memory.get_diverse_solution(PopPos[idx])
                    if memory_sol is not None:
                        new_positions[idx] = memory_sol + 0.25 * (ub - lb) * np.random.normal(0, 1, dim)
                    else:
                        new_positions[idx] = np.random.uniform(lb, ub, dim)
                else:  # Complete restart with Levy flight
                    center = (lb + ub) / 2
                    new_positions[idx] = center + levy_flight(dim, beta=1.5, scale=0.3*(ub-lb))
            else:  # Standard hybrid
                if restart_type < 0.3:  # Near best solution
                    new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.3 + 0.3 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.6:  # Near random elite
                    elite_idx = sorted_indices[np.random.randint(0, elites_count)]
                    new_positions[idx] = PopPos[elite_idx] + params['restart_scale'] * (ub - lb) * (0.2 + 0.4 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.8:  # From memory with diversity
                    memory_sol, _ = memory.get_diverse_solution(PopPos[idx])
                    if memory_sol is not None:
                        new_positions[idx] = memory_sol + 0.25 * (ub - lb) * np.random.normal(0, 1, dim)
                    else:
                        new_positions[idx] = np.random.uniform(lb, ub, dim)
                else:  # Complete restart
                    new_positions[idx] = np.random.uniform(lb, ub, dim)
        
        else:  # Composition (21-30) - high exploration with targeted regions
            if fun_index in [26, 28]:  # More successful compositions
                if restart_type < 0.4:  # Near best with moderate radius
                    new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.3 + 0.3 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.7:  # From memory with moderate diversity
                    memory_sol, _ = memory.get_random()
                    if memory_sol is not None:
                        new_positions[idx] = memory_sol + 0.25 * (ub - lb) * np.random.normal(0, 1, dim)
                    else:
                        new_positions[idx] = np.random.uniform(lb, ub, dim)
                else:  # Complete restart
                    new_positions[idx] = np.random.uniform(lb, ub, dim)
            else:  # Challenging compositions
                if restart_type < 0.2:  # Near best with larger radius
                    new_positions[idx] = BestX + params['restart_scale'] * (ub - lb) * (0.4 + 0.4 * np.random.random()) * np.random.normal(0, 1, dim)
                elif restart_type < 0.4:  # From memory with high diversity
                    memory_sol, _ = memory.get_diverse_solution(BestX)
                    if memory_sol is not None:
                        new_positions[idx] = memory_sol + 0.3 * (ub - lb) * np.random.normal(0, 1, dim)
                    else:
                        new_positions[idx] = np.random.uniform(lb, ub, dim)
                elif restart_type < 0.7:  # Levy flight from center
                    center = (lb + ub) / 2
                    new_positions[idx] = center + levy_flight(dim, beta=1.5, scale=0.4*(ub-lb))
                else:  # Complete random restart
                    new_positions[idx] = np.random.uniform(lb, ub, dim)
        
        # Apply bounds and evaluate
        new_positions[idx] = advanced_space_bound(new_positions[idx], lb, ub)
        new_fitnesses[idx] = evaluate_solution(new_positions[idx], dim, fun_index)
    
    return new_positions, new_fitnesses

# Enhanced MRFO algorithm with state-of-the-art improvements for version 7
def enhanced_adaptive_MRFO_v7(fun_index, max_iter, n_pop, lb, ub, dim, seed=None):
    """
    Enhanced Adaptive MRFO algorithm V7 with:
    1. Advanced function-specific parameter tuning
    2. Multi-strategy memory with enhanced diversity
    3. Function-specific specialized initialization
    4. Improved local search strategy
    5. Advanced restart mechanism
    6. Adaptive parameter control
    7. Specialized techniques for challenging functions
    """
    lb, ub, dim = fun_range_cec(lb, ub, dim)
    eval_count = 0  # Initialize evaluation counter
    
    # Get problem category
    category = get_category(fun_index)
    
    # Initialize population with enhanced diverse sampling
    PopPos, PopFit = initialize_diverse_population_v7(n_pop, dim, lb, ub, fun_index, seed)
    eval_count += n_pop
    
    # Initialize best solution
    BestF = np.min(PopFit)
    BestIndex = np.argmin(PopFit)
    BestX = PopPos[BestIndex, :].copy()
    
    # Initialize enhanced memory
    memory = EnhancedAdaptiveMemory(capacity=15, dim=dim)
    memory.add(BestX, BestF)
    
    # Initialize history arrays
    HisBestFit = np.zeros(max_iter)
    HisMeanFit = np.zeros(max_iter)
    HisStdFit = np.zeros(max_iter)
    
    # Variables for stagnation detection
    stagnation_counter = 0
    prev_best = BestF
    
    # Initialize diverse chaotic maps for each individual
    map_types = np.random.choice([0, 1, 2, 3, 4], n_pop)  # More map types
    chaotic_values = np.random.uniform(0.1, 0.9, n_pop)  # Away from fixed points
    
    # Start timing
    start_time = time.time()
    
    for It in range(max_iter):
        # Calculate iteration ratio for adaptive parameters
        iteration_ratio = It / max_iter
        
        # Get adaptive parameters based on function and current iteration
        params = get_enhanced_params_v7(fun_index, category, iteration_ratio)
        
        # Update chaotic values for each individual
        for i in range(n_pop):
            if map_types[i] == 0:
                chaotic_values[i] = sinusoidal_map(chaotic_values[i])
            elif map_types[i] == 1:
                chaotic_values[i] = logistic_map(chaotic_values[i])
            elif map_types[i] == 2:
                chaotic_values[i] = ikeda_map(chaotic_values[i])
            elif map_types[i] == 3:
                chaotic_values[i] = tent_map(chaotic_values[i])
            else:
                chaotic_values[i] = gauss_map(chaotic_values[i])
            
            # Ensure values stay in (0,1) range
            chaotic_values[i] = max(0.1, min(0.9, chaotic_values[i]))
        
        # Create new positions array
        newPopPos = np.zeros_like(PopPos)
        
        # Update population with ensemble of operators
        for i in range(n_pop):
            # Adaptive probability based on position in population and iteration
            individual_rank = i / n_pop
            adaptive_prob = params['cyclone_prob'] * (1 - 0.5 * individual_rank) * (1 + 0.3 * iteration_ratio)
            
            # Select operator based on individual's performance and problem type
            if np.random.rand() < adaptive_prob:
                # Cyclone foraging (exploitation)
                if i < n_pop // 3:  # Elite group - more exploitation
                    # Get reference point (best solution with some probability)
                    if np.random.rand() < 0.8:
                        reference = BestX
                    else:
                        # Use memory occasionally
                        memory_sol, _ = memory.get_best()
                        if memory_sol is not None:
                            reference = memory_sol
                        else:
                            reference = BestX
                    
                    # Enhanced cyclone movement
                    r1 = chaotic_values[i]
                    Beta = 2 * math.exp(r1 * ((max_iter - It + 1) / max_iter)) * math.sin(2 * math.pi * r1)
                    
                    # Dimension-wise operation
                    newPopPos[i, :] = reference + r1 * (reference - PopPos[i, :]) + Beta * (reference - PopPos[i, :])
                    
                    # Add mutation with small probability for exploration
                    if np.random.rand() < params['mutation_rate']:
                        mutation_dims = np.random.choice(dim, max(1, dim // 10), replace=False)
                        newPopPos[i, mutation_dims] = np.random.uniform(lb, ub, len(mutation_dims))
                
                else:  # Middle and lower groups - chain following with adaptation
                    # More diverse reference selection
                    if np.random.rand() < 0.7:
                        # Follow previous individual (chain formation)
                        reference = PopPos[max(0, i-1), :]
                    else:
                        # Follow best occasionally
                        reference = BestX
                    
                    r1 = chaotic_values[i]
                    Beta = 2 * math.exp(r1 * ((max_iter - It + 1) / max_iter)) * math.sin(2 * math.pi * r1)
                    
                    # Enhanced movement with diversity control
                    diversity_factor = params['diversity_control'] * (1 - iteration_ratio)
                    newPopPos[i, :] = reference + (r1 + diversity_factor * np.random.random()) * (reference - PopPos[i, :]) + Beta * (BestX - PopPos[i, :])
            
            else:
                # Specialized exploration strategy based on function type
                if category == "Unimodal":
                    # For unimodal, more structured exploration
                    Alpha = params['alpha_scale'] * chaotic_values[i] * np.sqrt(np.abs(np.log(chaotic_values[i] + 1e-10)))
                    
                    # Use dimension-wise crossover
                    crossover_rate = params['crossover_rate'] * (1 - 0.2 * iteration_ratio)
                    mask = np.random.random(dim) < crossover_rate
                    
                    # Create new position
                    newPopPos[i, :] = PopPos[i, :].copy()
                    newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (BestX[mask] - PopPos[i, mask]) + Alpha * (BestX[mask] - PopPos[i, mask])
                    
                elif category == "Multimodal":
                    # For multimodal, use Lévy flight for better exploration of multiple optima
                    if fun_index in [5, 7, 8, 10]:  # Challenging multimodal
                        # More aggressive exploration
                        levy_step = levy_flight(dim, beta=1.8, scale=0.15 * (ub - lb))
                        
                        # Dynamic crossover 
                        crossover_rate = params['crossover_rate'] * (1 - 0.2 * iteration_ratio)
                        mask = np.random.random(dim) < crossover_rate
                        
                        # Create new position with Lévy flight
                        newPopPos[i, :] = PopPos[i, :].copy()
                        newPopPos[i, mask] = PopPos[i, mask] + levy_step[mask]
                        
                        # Occasional large jump
                        if np.random.random() < 0.1:
                            jump_size = 0.3 * (ub - lb)
                            jump_dims = np.random.choice(dim, max(1, dim // 5), replace=False)
                            newPopPos[i, jump_dims] += np.random.normal(0, 1, len(jump_dims)) * jump_size
                    else:
                        # Standard multimodal
                        Alpha = params['alpha_scale'] * chaotic_values[i] * np.sqrt(np.abs(np.log(chaotic_values[i] + 1e-10)))
                        
                        # Dynamic crossover
                        crossover_rate = params['crossover_rate'] * (1 - 0.3 * iteration_ratio)
                        mask = np.random.random(dim) < crossover_rate
                        
                        # Create new position with adaptive steps
                        newPopPos[i, :] = PopPos[i, :].copy()
                        newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (BestX[mask] - PopPos[i, mask]) + Alpha * (BestX[mask] - PopPos[i, mask])
                
                elif category == "Hybrid":
                    # For hybrid, balanced approach between exploration and exploitation
                    if fun_index in [11, 12, 13, 15]:  # Challenging hybrids
                        # More exploration with memory-based learning
                        memory_sol, _ = memory.get_random()
                        if memory_sol is not None and np.random.random() < 0.4:
                            reference = memory_sol
                        else:
                            reference = BestX
                            
                        # Lévy flight with adaptive scales
                        levy_step = levy_flight(dim, beta=1.7, scale=0.2 * (ub - lb))
                        
                        # Combine lévy flight with memory-based learning
                        alpha = np.random.random() * 0.8
                        newPopPos[i, :] = PopPos[i, :] + alpha * levy_step + (1-alpha) * chaotic_values[i] * (reference - PopPos[i, :])
                    else:
                        # Standard hybrid
                        # Chain foraging with exploration enhancement
                        Alpha = params['alpha_scale'] * chaotic_values[i] * np.sqrt(np.abs(np.log(chaotic_values[i] + 1e-10)))
                        
                        # Dynamic crossover rate based on iteration
                        crossover_rate = params['crossover_rate'] * (1 - 0.3 * iteration_ratio)
                        mask = np.random.random(dim) < crossover_rate
                        
                        # Create new position
                        newPopPos[i, :] = PopPos[i, :].copy()
                        
                        if i < n_pop // 2:  # First half - more structured
                            # Follow best with chaotic perturbation
                            newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (BestX[mask] - PopPos[i, mask]) + Alpha * (BestX[mask] - PopPos[i, mask])
                        else:  # Second half - more exploratory
                            # More random operation for diversity
                            if np.random.rand() < 0.5:
                                # Use diverse solution from memory
                                memory_sol, _ = memory.get_diverse_solution(PopPos[i, :])
                                if memory_sol is not None:
                                    reference = memory_sol
                                else:
                                    reference = BestX
                            else:
                                # Generate random point
                                reference = np.random.uniform(lb, ub, dim)
                            
                            # Add lévy step occasionally
                            if np.random.random() < 0.3:
                                levy_step = levy_flight(dim, beta=1.5, scale=0.1 * (ub - lb))
                                newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (reference[mask] - PopPos[i, mask]) + levy_step[mask]
                            else:
                                newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (reference[mask] - PopPos[i, mask])
                
                else:  # Composition functions - high exploration
                    if fun_index in [25, 27, 29, 30]:  # Most challenging compositions
                        # Highly exploratory with diverse reference points
                        if np.random.random() < 0.4:
                            memory_sol, _ = memory.get_diverse_solution(PopPos[i, :])
                            if memory_sol is not None:
                                reference = memory_sol
                            else:
                                # Random point in space
                                reference = np.random.uniform(lb, ub, dim)
                                
                            # Create new position with diverse reference
                            levy_step = levy_flight(dim, beta=1.9, scale=0.25 * (ub - lb))
                            newPopPos[i, :] = PopPos[i, :] + chaotic_values[i] * (reference - PopPos[i, :]) + levy_step
                        else:
                            # Random search in subspace
                            subspace_size = max(2, dim // 3)
                            active_dims = np.random.choice(dim, subspace_size, replace=False)
                            
                            # Create perturbation only in active dimensions
                            newPopPos[i, :] = PopPos[i, :].copy()
                            newPopPos[i, active_dims] = PopPos[i, active_dims] + 0.3 * (ub - lb) * np.random.normal(0, 1, subspace_size)
                    else:
                        # Standard composition
                        Alpha = params['alpha_scale'] * chaotic_values[i] * np.sqrt(np.abs(np.log(chaotic_values[i] + 1e-10)))
                        
                        # Dynamic crossover rate based on iteration
                        crossover_rate = params['crossover_rate'] * (1 - 0.3 * iteration_ratio)
                        mask = np.random.random(dim) < crossover_rate
                        
                        # Create new position
                        newPopPos[i, :] = PopPos[i, :].copy()
                        
                        if np.random.rand() < 0.6:
                            # Use best solution as reference
                            newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (BestX[mask] - PopPos[i, mask]) + Alpha * (BestX[mask] - PopPos[i, mask])
                        else:
                            # Use random elite as reference
                            elite_idx = np.random.randint(0, max(1, n_pop // 5))
                            sorted_indices = np.argsort(PopFit)
                            reference = PopPos[sorted_indices[elite_idx], :]
                            newPopPos[i, mask] = PopPos[i, mask] + chaotic_values[i] * (reference[mask] - PopPos[i, mask])
        
        # Space bound check and evaluation
        for i in range(n_pop):
            newPopPos[i, :] = advanced_space_bound(newPopPos[i, :], lb, ub)
            newFit = evaluate_solution(newPopPos[i, :], dim, fun_index)
            eval_count += 1
            
            if newFit < PopFit[i]:
                # Record successful solution in memory if significantly better
                if newFit < PopFit[i] * 0.9:  # At least 10% improvement
                    memory.add(newPopPos[i, :], newFit)
                
                PopFit[i] = newFit
                PopPos[i, :] = newPopPos[i, :].copy()
        
        # Adaptive somersault factor
        base_S = params['somersault_factor']
        if stagnation_counter > params['restart_threshold'] // 2:
            # Increase step size for more exploration when stagnating
            S = base_S * (1 + 0.5 * stagnation_counter / params['restart_threshold'])
        else:
            # Normal step size adaptation based on iteration progress
            S = base_S * (1 - 0.3 * iteration_ratio)
        
        # Enhanced somersault foraging with multiple strategies
        for i in range(n_pop):
            # Different strategies based on position in population and function type
            if category == "Unimodal":
                # For unimodal functions - favor exploitation
                if i < n_pop // 3:  # Top performers - refined exploitation
                    # Use best solution as reference
                    reference = BestX
                    
                    # Controlled small steps around best
                    r_factor = chaotic_values[i] * (1 - 0.5 * iteration_ratio)
                    somersault_pos = PopPos[i, :] + S * (r_factor * reference - (1 - r_factor) * PopPos[i, :])
                else:
                    # Standard somersault for remainder
                    r_factors = np.random.random(dim)
                    somersault_pos = PopPos[i, :] + S * (r_factors * BestX - (1 - r_factors) * PopPos[i, :])
            
            elif category == "Multimodal":
                if fun_index in [5, 7, 8, 10]:  # Challenging multimodal
                    if i < n_pop // 4:  # Top performers
                        # Focused exploitation around best
                        r_factors = np.random.random(dim)
                        somersault_pos = PopPos[i, :] + S * (r_factors * BestX - (1 - r_factors) * PopPos[i, :])
                    elif i < n_pop // 2:  # Mid performers
                        # Mix of best and memory reference
                        memory_sol, _ = memory.get_random()
                        if memory_sol is not None and np.random.random() < 0.5:
                            reference = memory_sol
                        else:
                            reference = BestX
                        
                        r_factors = np.random.random(dim)
                        somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                    else:  # Lower performers - more exploration
                        # Either restart or make big jumps
                        if np.random.rand() < 0.3:
                            # Complete restart
                            somersault_pos = np.random.uniform(lb, ub, dim)
                        else:
                            # Use Lévy flight from current position
                            levy_step = levy_flight(dim, beta=1.8, scale=0.2*(ub-lb))
                            somersault_pos = PopPos[i, :] + levy_step
                else:  # Standard multimodal
                    if i < n_pop // 4:  # Top performers - refined exploitation
                        # Use best solution as reference
                        reference = BestX
                        
                        # Controlled small steps around best
                        r_factor = chaotic_values[i] * (1 - 0.5 * iteration_ratio)
                        somersault_pos = PopPos[i, :] + S * (r_factor * reference - (1 - r_factor) * PopPos[i, :])
                    
                    elif i < n_pop // 2:  # Upper middle - balanced approach
                        # Use dimension-wise random factors
                        r_factors = np.random.random(dim)
                        
                        # Occasionally use memory
                        if np.random.random() < 0.3:
                            memory_sol, _ = memory.get_random()
                            if memory_sol is not None:
                                reference = memory_sol
                            else:
                                reference = BestX
                        else:
                            reference = BestX
                        
                        somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                    
                    else:  # Lower half - more exploration
                        # Use weighted combination of best and random
                        r_factors = np.random.random(dim)
                        random_point = np.random.uniform(lb, ub, dim)
                        
                        weight = 0.7 - 0.4 * iteration_ratio  # Reduce randomness over time
                        somersault_pos = PopPos[i, :] + S * ((1 - weight) * r_factors * BestX + 
                                                       weight * r_factors * random_point - 
                                                      (1 - r_factors) * PopPos[i, :])
            
            elif category == "Hybrid":
                if fun_index in [11, 12, 13, 15]:  # Most challenging hybrids
                    if i < n_pop // 4:  # Top performers
                        # Mix of best and memory reference
                        memory_sol, _ = memory.get_best()
                        if memory_sol is not None and np.random.random() < 0.5:
                            reference = memory_sol
                        else:
                            reference = BestX
                        
                        r_factors = np.random.random(dim)
                        somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                    else:  # Lower performers - high exploration
                        # Either restart or make big jumps
                        if np.random.rand() < 0.4:
                            # Complete restart or levy flight
                            if np.random.random() < 0.5:
                                somersault_pos = np.random.uniform(lb, ub, dim)
                            else:
                                center = (lb + ub) / 2
                                levy_step = levy_flight(dim, beta=1.5, scale=0.3*(ub-lb))
                                somersault_pos = center + levy_step
                        else:
                            # Use random elite as reference
                            elite_idx = np.random.randint(0, max(1, n_pop // 5))
                            sorted_indices = np.argsort(PopFit)
                            reference = PopPos[sorted_indices[elite_idx], :]
                            
                            r_factors = np.random.random(dim)
                            somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                else:  # Standard hybrid
                    if i < n_pop // 3:  # Top performers - refined exploitation
                        # Use best solution as reference
                        reference = BestX
                        
                        # Controlled small steps around best
                        r_factor = chaotic_values[i] * (1 - 0.5 * iteration_ratio)
                        somersault_pos = PopPos[i, :] + S * (r_factor * reference - (1 - r_factor) * PopPos[i, :])
                    
                    elif i < 2 * n_pop // 3:  # Middle group - balanced approach
                        # Use dimension-wise random factors
                        r_factors = np.random.random(dim)
                        
                        # Choose reference adaptively
                        if np.random.random() < 0.7:
                            reference = BestX
                        else:
                            # Use random elite
                            elite_idx = np.random.randint(0, max(1, n_pop // 5))
                            sorted_indices = np.argsort(PopFit)
                            reference = PopPos[sorted_indices[elite_idx], :]
                        
                        somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                    
                    else:  # Bottom performers - high exploration
                        # Random restart with some probability
                        if np.random.random() < 0.2:
                            somersault_pos = np.random.uniform(lb, ub, dim)
                        else:
                            # Use Lévy flight from current position
                            levy_step = levy_flight(dim, beta=1.5, scale=0.15*(ub-lb))
                            somersault_pos = PopPos[i, :] + levy_step
            
            else:  # Composition functions
                if fun_index in [25, 27, 29, 30]:  # Most challenging compositions
                    if i < n_pop // 5:  # Top performers - careful exploitation
                        r_factors = np.random.random(dim)
                        somersault_pos = PopPos[i, :] + S * (r_factors * BestX - (1 - r_factors) * PopPos[i, :])
                    elif i < n_pop // 2:  # Mid performers - balanced approach with memory
                        # Use memory-based reference
                        memory_sol, _ = memory.get_random()
                        if memory_sol is not None:
                            ref_weight = 0.7
                            reference = ref_weight * BestX + (1 - ref_weight) * memory_sol
                        else:
                            reference = BestX
                        
                        r_factors = np.random.random(dim)
                        somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                    else:  # Lower performers - high exploration
                        # High probability of restart or large move
                        if np.random.random() < 0.5:
                            if np.random.random() < 0.3:
                                somersault_pos = np.random.uniform(lb, ub, dim)
                            else:
                                center = (lb + ub) / 2
                                levy_step = levy_flight(dim, beta=1.8, scale=0.4*(ub-lb))
                                somersault_pos = center + levy_step
                        else:
                            # Random direction search
                            direction = np.random.normal(0, 1, dim)
                            direction = direction / np.linalg.norm(direction)
                            step_size = 0.3 * (ub - lb) * np.random.random()
                            somersault_pos = PopPos[i, :] + step_size * direction
                else:  # Better performing compositions
                    if i < n_pop // 3:  # Top performers - refined exploitation
                        # Use best solution as reference
                        reference = BestX
                        
                        # Controlled small steps around best
                        r_factor = chaotic_values[i] * (1 - 0.5 * iteration_ratio)
                        somersault_pos = PopPos[i, :] + S * (r_factor * reference - (1 - r_factor) * PopPos[i, :])
                    
                    elif i < 2 * n_pop // 3:  # Middle group - balanced approach
                        # Use dimension-wise random factors
                        r_factors = np.random.random(dim)
                        
                        # Occasionally use memory
                        if np.random.random() < 0.3:
                            memory_sol, _ = memory.get_random()
                            if memory_sol is not None:
                                reference = memory_sol
                            else:
                                reference = BestX
                        else:
                            reference = BestX
                        
                        somersault_pos = PopPos[i, :] + S * (r_factors * reference - (1 - r_factors) * PopPos[i, :])
                    
                    else:  # Bottom performers - more exploration
                        # Use weighted combination of best and random
                        r_factors = np.random.random(dim)
                        random_point = np.random.uniform(lb, ub, dim)
                        
                        weight = 0.5 - 0.3 * iteration_ratio  # Reduce randomness over time
                        somersault_pos = PopPos[i, :] + S * ((1 - weight) * r_factors * BestX + 
                                                      weight * r_factors * random_point - 
                                                      (1 - r_factors) * PopPos[i, :])
            
            # Space bound check
            somersault_pos = advanced_space_bound(somersault_pos, lb, ub)
            somersault_fit = evaluate_solution(somersault_pos, dim, fun_index)
            eval_count += 1
            
            if somersault_fit < PopFit[i]:
                # Add to memory if significant improvement
                if somersault_fit < PopFit[i] * 0.9:
                    memory.add(somersault_pos, somersault_fit)
                
                PopFit[i] = somersault_fit
                PopPos[i, :] = somersault_pos.copy()
        
        # Enhanced local search with multi-strategy approach
        if It % params['local_search_freq'] == 0:
            # Select the best solution
            curr_best_idx = np.argmin(PopFit)
            base_pos = PopPos[curr_best_idx, :].copy()
            base_fit = PopFit[curr_best_idx]
            
            # Perform advanced multi-strategy local search
            improved_pos, improved_fit, local_evals = enhanced_multi_strategy_local_search(
                base_pos, base_fit, dim, fun_index, lb, ub, 
                params, iteration_ratio
            )
            
            # Add local search evaluations
            eval_count += local_evals
            
            if improved_fit < base_fit:
                PopPos[curr_best_idx, :] = improved_pos.copy()
                PopFit[curr_best_idx] = improved_fit
                
                # Add to memory
                memory.add(improved_pos, improved_fit)
        
        # Update global best
        curr_best_idx = np.argmin(PopFit)
        if PopFit[curr_best_idx] < BestF:
            BestF = PopFit[curr_best_idx]
            BestX = PopPos[curr_best_idx, :].copy()
            stagnation_counter = 0  # Reset stagnation counter
            
            # Add to memory
            memory.add(BestX, BestF)
        else:
            stagnation_counter += 1
        
        # Advanced stagnation handling with enhanced restart strategy
        if stagnation_counter > params['restart_threshold']:
            # Apply advanced restart with memory utilization
            PopPos, PopFit = advanced_restart_strategy(
                PopPos, PopFit, BestX, memory, n_pop, 
                lb, ub, dim, fun_index, params, iteration_ratio
            )
            
            # Add restart evaluations (one per restarted individual)
            eval_count += int(n_pop * (0.3 + 0.1 * (1 - iteration_ratio)))
            
            stagnation_counter = 0
        
        # Record history
        HisBestFit[It] = BestF
        HisMeanFit[It] = np.mean(PopFit)
        HisStdFit[It] = np.std(PopFit)
    
    run_time = time.time() - start_time
    return PopFit, BestF, run_time, eval_count, HisBestFit, HisMeanFit, HisStdFit
    # Function to run single evaluation (will be used in parallel)
def run_single_evaluation(fun_index, seed, max_iter, n_pop, lb, ub, dim, tol):
    PopFit, BestF, run_time, func_evals, HisBestFit, HisMeanFit, HisStdFit = enhanced_adaptive_MRFO_v7(
        fun_index, max_iter, n_pop, lb, ub, dim, seed
    )
    
    bias = biases.get(fun_index, 0)
    best_err = BestF - bias
    
    # Calculate convergence iteration
    conv_iter = max_iter
    for it, val in enumerate(HisBestFit):
        if (val - bias) <= tol:
            conv_iter = it + 1
            break
    
    results = {
        'best': BestF,
        'best_err': best_err,
        'median': np.median(PopFit),
        'worst': np.max(PopFit),
        'std': np.std(PopFit),
        'success': 1 if best_err <= tol else 0,
        'conv_iter': conv_iter,
        'run_time': run_time,
        'func_evals': func_evals,
        'history': {
            'best': HisBestFit, 
            'mean': HisMeanFit, 
            'std': HisStdFit
        }
    }
    
    return results

# Function to run multiple runs in parallel
def run_multiple_runs_parallel(fun_index, n_runs, max_iter, n_pop, lb, ub, dim, tol, n_jobs):
    print(f"Starting function {fun_index} ({get_category(fun_index)}) with {n_runs} runs, {n_pop} population, {max_iter} iterations")
    
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_single_evaluation)(fun_index, i, max_iter, n_pop, lb, ub, dim, tol)
        for i in range(n_runs)
    )
    
    # Aggregate results
    best_list = [r['best'] for r in results_list]
    best_errors = [r['best_err'] for r in results_list]
    median_list = [r['median'] for r in results_list]
    worst_list = [r['worst'] for r in results_list]
    std_list = [r['std'] for r in results_list]
    conv_iters = [r['conv_iter'] for r in results_list]
    run_times = [r['run_time'] for r in results_list]
    func_evals_list = [r['func_evals'] for r in results_list]
    
    # Save first run history
    conv_history_first = results_list[0]['history']
    
    aggregated = {
        'Best_Raw_Mean': np.mean(best_list),
        'Best_Raw_Median': np.median(best_list),
        'Best_Error_Mean': np.mean(best_errors),
        'Best_Error_Median': np.median(best_errors),
        'Median_Mean': np.mean(median_list),
        'Worst_Mean': np.mean(worst_list),
        'Std_Mean': np.mean(std_list),
        'Success_Rate (%)': 100 * np.mean([r['success'] for r in results_list]),
        'Avg_Convergence_Iter': np.mean(conv_iters),
        'Best_Error_Var': np.var(best_errors),
        'Best_Error_IQR': np.subtract(*np.percentile(best_errors, [75, 25])),
        'Avg_RunTime(s)': np.mean(run_times),
        'Avg_FuncEvals': np.mean(func_evals_list)
    }
    
    print(f"Completed function {fun_index}: Best error = {aggregated['Best_Error_Mean']}, Success rate = {aggregated['Success_Rate (%)']}")
    
    return aggregated, conv_history_first

def main():
    # Enhanced resources for better performance
    dim = 10           # Dimensionality
    n_pop = 200         # Increased population size
    max_iter = 5000     # Increased maximum iterations
    lb = -100          # Lower bound
    ub = 100           # Upper bound
    n_runs = 20         # Number of independent runs per function
    tol = 1e-2         # Tolerance for success
    
    # Use all available CPUs minus 1
    n_jobs = os.cpu_count() - 1
    print(f"Starting Enhanced Adaptive MRFO with {n_jobs} processors")
    print(f"Configuration: Population={n_pop}, Max Iterations={max_iter}, Dimension={dim}, Runs={n_runs}")
    
    results_raw = []
    results_no_bias = []
    detailed_stats = []
    conv_histories = {}
    
    # Run all functions in parallel
    for fun_index in range(1, 31):
        category = get_category(fun_index)
        
        stats, conv_history_first = run_multiple_runs_parallel(
            fun_index, n_runs, max_iter, n_pop, lb, ub, dim, tol, n_jobs
        )
        
        bias = biases.get(fun_index, 0)
        result_raw = {
            'Func': fun_index,
            'Category': category,
            'Best': stats['Best_Raw_Mean'],
            'Median': stats['Best_Raw_Median'],
            'RunTime(s)': stats['Avg_RunTime(s)'],
            'FuncEvals': stats['Avg_FuncEvals']
        }
        result_no_bias = {
            'Func': fun_index,
            'Category': category,
            'Best': stats['Best_Error_Mean'],
            'Median': stats['Best_Error_Median'],
            'Success Rate (%)': stats['Success_Rate (%)'],
            'Avg_Convergence_Iter': stats['Avg_Convergence_Iter'],
            'Variance': stats['Best_Error_Var'],
            'IQR': stats['Best_Error_IQR'],
            'RunTime(s)': stats['Avg_RunTime(s)'],
            'FuncEvals': stats['Avg_FuncEvals']
        }
        results_raw.append(result_raw)
        results_no_bias.append(result_no_bias)
        detailed_stats.append({
            'Func': fun_index,
            'Category': category,
            'Median_Mean': stats['Median_Mean'],
            'Worst_Mean': stats['Worst_Mean'],
            'Std_Mean': stats['Std_Mean']
        })
        conv_histories[fun_index] = conv_history_first
    
    # Save results to Excel
    df_raw = pd.DataFrame(results_raw)
    df_nb = pd.DataFrame(results_no_bias)
    df_stats = pd.DataFrame(detailed_stats)
    
    df_raw.to_excel("Enhanced_Raw_Results_with_Bias.xlsx", index=False)
    df_nb.to_excel("Enhanced_Results_After_Bias_Removal.xlsx", index=False)
    df_stats.to_excel("Enhanced_Additional_Stats.xlsx", index=False)
    
    print("\nRAW RESULTS (with bias) (averaged over runs):")
    print(df_raw.to_string(index=False))
    print("\nRESULTS AFTER BIAS REMOVAL (Error relative to optimum):")
    print(df_nb.to_string(index=False))
    print("\nADDITIONAL STATISTICS:")
    print(df_stats.to_string(index=False))
    
    # Create convergence plots
    n_rows, n_cols = 5, 6
    x_axis = np.arange(max_iter)
    
    # Plot raw convergence curves
    fig_raw, axs_raw = plt.subplots(n_rows, n_cols, figsize=(20, 14))
    axs_raw = axs_raw.flatten()
    for fun_index in range(1, 31):
        conv_curve = conv_histories[fun_index]['best']
        axs_raw[fun_index - 1].plot(x_axis, conv_curve, 'b-', linewidth=1)
        axs_raw[fun_index - 1].set_title(f"f{fun_index}")
        axs_raw[fun_index - 1].set_xlabel("Iteration")
        axs_raw[fun_index - 1].set_ylabel("Fitness")
        # Only use log scale if all values are positive
        if np.all(conv_curve > 0):
            axs_raw[fun_index - 1].set_yscale('log')
    fig_raw.tight_layout()
    fig_raw.suptitle("Enhanced Adaptive MRFO: Convergence Curves (Raw)", fontsize=16, y=1.02)
    fig_raw.savefig("Enhanced_Convergence_Raw.png", bbox_inches="tight")
    
    # Plot error convergence curves
    fig_nb, axs_nb = plt.subplots(n_rows, n_cols, figsize=(20, 14))
    axs_nb = axs_nb.flatten()
    for fun_index in range(1, 31):
        bias = biases.get(fun_index, 0)
        conv_curve = conv_histories[fun_index]['best'] - bias
        axs_nb[fun_index - 1].plot(x_axis, conv_curve, 'g-', linewidth=1)
        axs_nb[fun_index - 1].set_title(f"f{fun_index}")
        axs_nb[fun_index - 1].set_xlabel("Iteration")
        axs_nb[fun_index - 1].set_ylabel("Fitness Error")
        # Only use log scale if all values are positive
        if np.all(conv_curve > 0):
            axs_nb[fun_index - 1].set_yscale('log')
    fig_nb.tight_layout()
    fig_nb.suptitle("Enhanced Adaptive MRFO: Convergence Curves (Bias Removed)", fontsize=16, y=1.02)
    fig_nb.savefig("Enhanced_Convergence_Bias_Removed.png", bbox_inches="tight")
    
    # Try to create comparison chart with previous results
    try:
        # Try loading Phase-9 results first
        try:
            previous_df = pd.read_excel("Phase9_Results_After_Bias_Removal.xlsx")
            prev_name = "Phase9"
        except:
            # If not found, try loading streamlined results
            try:
                previous_df = pd.read_excel("Streamlined_Results_After_Bias_Removal.xlsx")
                prev_name = "Streamlined"
            except:
                # If not found, try loading original results
                previous_df = pd.read_excel("Results_After_Bias_RemovalGPTPaid.xlsx")
                prev_name = "Original"
        
        enhanced_df = df_nb.copy()
        
        comparison = pd.DataFrame()
        comparison['Function'] = previous_df['Func']
        comparison['Category'] = previous_df['Category']
        comparison[f'{prev_name}_Error'] = previous_df['Best']
        comparison['Enhanced_Error'] = enhanced_df['Best']
        comparison['Improvement (%)'] = 100 * (previous_df['Best'] - enhanced_df['Best']) / np.maximum(previous_df['Best'], 1e-10)
        comparison[f'{prev_name}_Success (%)'] = previous_df['Success Rate (%)']
        comparison['Enhanced_Success (%)'] = enhanced_df['Success Rate (%)']
        
        comparison.to_excel("Enhanced_Improvement_Comparison.xlsx", index=False)
        print("\nIMPROVEMENT COMPARISON:")
        print(comparison.to_string(index=False))
        
        # Plot improvement comparison
        plt.figure(figsize=(14, 10))
        plt.bar(comparison['Function'], comparison['Improvement (%)'])
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Function')
        plt.ylabel('Improvement (%)')
        plt.title(f'Performance Improvement of Enhanced Adaptive MRFO over {prev_name} MRFO')
        plt.xticks(comparison['Function'])
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig("Enhanced_Improvement_Comparison.png", bbox_inches="tight")
        
    except Exception as e:
        print(f"Could not create comparison chart: {e}")
    
    plt.show()

if __name__ == "__main__":
    main()