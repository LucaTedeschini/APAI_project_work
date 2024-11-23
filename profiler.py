import os
import sys
import subprocess
import time

try:
    import matplotlib.pyplot as plt
    plt_exists = True
except:
    plt_exists = False

print("MatPlotLib correctly imported" if plt_exists else "MatPlotLib was not imported. Please install it to see the graphs")

# Path to the executable
executable_path = "build/APAI_ff_nn"
if not os.path.exists(executable_path):
    print("Executable does not exists!")
    sys.exit(1)

# Parameters (N = neurons first layer, L = layers, R = sparse coefficient)
N = "100000"
L = "100"
R = "3"
result_dict = {}
for num_threads in range(1,12+1):
    print(f"Testing n_thread: {num_threads}")
    start = time.time()
    results = []
    for i in range(30):
        if i % 5 == 0:
            end = time.time()
            print(f"\t {i}/30 - {round(end-start,5)}s")
            start = time.time()
        # Check the correctness of the position of the input parameters
        p = subprocess.Popen([executable_path, N, L, R], env={"OMP_NUM_THREADS": str(num_threads)}, stdout=subprocess.PIPE)
        results.append(float(p.communicate()[0].decode("utf-8").replace("\n","").replace("Took: ","")))
    result_dict[num_threads] = round(sum(results)/len(results),5)
    print("\t Done")

baseline = result_dict[1]

#Compute speedup
baseline = result_dict[1]  # Baseline is the single-threaded time
speedup_dict = {
    threads: round(baseline / time, 5)
    for threads, time in result_dict.items()
}

# Print results
print("Execution Times (seconds):")
for threads, time in result_dict.items():
    print(f"\t{threads} threads: {time} s")

print("\nSpeedup:")
for threads, speedup in speedup_dict.items():
    print(f"\t{threads} threads: {speedup}x")

x = list(speedup_dict.keys())
y = list(speedup_dict.values())

baseline_x, baseline_y = x[0], y[0]
best_x, best_y = sorted(speedup_dict.items(), key=lambda x: x[1], reverse=True)[0][0], sorted(speedup_dict.items(), key=lambda x: x[1], reverse=True)[0][1]

x.remove(baseline_x)
x.remove(best_x)
y.remove(baseline_y)
y.remove(best_y)



plt.bar(baseline_x, baseline_y, color="green")
plt.bar(best_x, best_y, label="Best", color="yellow")
plt.bar(x, y, label="Speedup", color="blue")
plt.plot(range(1, x[-1]+1), [1] * len(range(1, x[-1]+1)), color="green", label="Baseline")

plt.legend()
plt.show()
