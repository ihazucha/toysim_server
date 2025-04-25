import numpy as np
import timeit
import collections # Use 'collections' consistently

# --- Configuration ---
ARRAY_SIZE = 10000
NUM_ROLLS = 10000
NEW_VALUE = 99999

# --- Setup ---
# Initial data setup is not strictly needed here as timeit handles its own setup,
# but it's okay to keep for potential direct use or clarity.
initial_array = np.arange(ARRAY_SIZE, dtype=np.int64)
initial_deque = collections.deque(initial_array, maxlen=ARRAY_SIZE)

# --- Approach 1: Slicing ---
def roll_with_slicing(arr, value):
    """Rolls array by shifting left and adding value at the end using slicing."""
    new_arr = np.empty_like(arr)
    new_arr[:-1] = arr[1:]
    new_arr[-1] = value
    return new_arr.copy()

# --- Approach 2: np.roll ---
def roll_with_np_roll(arr, value):
    """Rolls array by shifting left and adding value at the end using np.roll."""
    rolled_arr = np.roll(arr, -1)
    rolled_arr[-1] = value
    return rolled_arr.copy()

# --- Approach 3: Deque ---
def roll_with_deque(d, value):
    """Appends value to deque (automatically removing oldest) and converts back to numpy array."""
    d.append(value)
    return np.array(d) # The conversion is part of the cost here

# --- Timing ---

# Define common globals for timeit
common_globals = {
    'np': np,
    'collections': collections,
    'ARRAY_SIZE': ARRAY_SIZE,
    'NEW_VALUE': NEW_VALUE,
    'roll_with_slicing': roll_with_slicing,
    'roll_with_np_roll': roll_with_np_roll,
    'roll_with_deque': roll_with_deque
}

# Time Slicing
slice_setup = f"""
arr = np.arange(ARRAY_SIZE, dtype=np.int64)
"""
slice_stmt = "arr = roll_with_slicing(arr, NEW_VALUE)"
slice_time = timeit.timeit(stmt=slice_stmt, setup=slice_setup, number=NUM_ROLLS, globals=common_globals)
print(f"Slicing approach time:      {slice_time:.6f} seconds for {NUM_ROLLS} rolls")

# Time np.roll
np_roll_setup = f"""
arr = np.arange(ARRAY_SIZE, dtype=np.int64)
"""
np_roll_stmt = "arr = roll_with_np_roll(arr, NEW_VALUE)"
np_roll_time = timeit.timeit(stmt=np_roll_stmt, setup=np_roll_setup, number=NUM_ROLLS, globals=common_globals)
print(f"np.roll approach time:      {np_roll_time:.6f} seconds for {NUM_ROLLS} rolls")

# Time Deque
deque_setup = f"""
initial_array = np.arange(ARRAY_SIZE, dtype=np.int64)
d = collections.deque(initial_array, maxlen=ARRAY_SIZE)
"""
deque_stmt = "arr = roll_with_deque(d, NEW_VALUE)"
deque_time = timeit.timeit(stmt=deque_stmt, setup=deque_setup, number=NUM_ROLLS, globals=common_globals)
print(f"Deque + np.array() approach time: {deque_time:.6f} seconds for {NUM_ROLLS} rolls")

# --- Optional: Deque append only (to isolate conversion cost) ---
deque_append_only_setup = f"""
initial_array = range(ARRAY_SIZE) # Use range for deque init for speed
d = collections.deque(initial_array, maxlen=ARRAY_SIZE)
"""
deque_append_only_stmt = "d.append(NEW_VALUE)"
# Need specific globals for this setup if it differs significantly
deque_append_globals = {
    'collections': collections,
    'ARRAY_SIZE': ARRAY_SIZE,
    'NEW_VALUE': NEW_VALUE
}
deque_append_only_time = timeit.timeit(stmt=deque_append_only_stmt, setup=deque_append_only_setup, number=NUM_ROLLS, globals=deque_append_globals)
print(f"Deque append() only time:   {deque_append_only_time:.6f} seconds for {NUM_ROLLS} rolls")
print(f"\nNote: Deque time includes np.array() conversion overhead.")
print(f"Array Size: {ARRAY_SIZE}, Number of Rolls: {NUM_ROLLS}")