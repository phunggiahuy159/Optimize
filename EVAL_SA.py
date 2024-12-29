#PYTHON 
#PYTHON 
import random
import math
from time import time as tm

def read_input():
    N, Q = map(int, input().split())
    pred = []
    for _ in range(Q):
        pred.append(tuple(map(int, input().split())))
    d = [0] + list(map(int, input().split()))
    M = int(input())
    s = [0] + list(map(int, input().split()))
    K = int(input())
    c = [[0 for i in range(M + 1)] for j in range(N + 1)]
    for _ in range(K):
        task, team, cost = map(int, input().split())
        c[task][team] = cost
    return N, Q, pred, d, M, s, K, c

def TopologicalSort(N, pred):
    edge = [set() for i in range(N + 1)]
    in_degree = [0 for i in range(N + 1)]
    for u, v in pred:
        edge[u].add(v)
        in_degree[v] += 1
    zero_in_degree = []
    for i in range(1, N + 1):
        if in_degree[i] == 0:
            zero_in_degree = [i] + zero_in_degree
    topo = []
    while len(zero_in_degree) > 0:
        node = zero_in_degree.pop()
        topo.append(node)
        for u in edge[node]:
            in_degree[u] -= 1
            if in_degree[u] == 0:
                zero_in_degree = [u] + zero_in_degree
    return topo

class State:
    def __init__(self, l, N, M, s, d, c, pred, before, after):
        self.l = l[:]
        self.N = N
        self.M = M
        self.s = s
        self.d = d
        self.c = c
        self.pred = pred
        self.before = before
        self.after = after
        self.index = self.GetIndex()
        self.evaluation = (0, 0)
        self.start_times = [0] * (N + 1)
        self.team_assignments = [-1] * (N + 1)

    def GetEvaluation(self):
        scheduled = 0
        tmp_s = self.s[:]
        tmp_time = [0 for _ in range(self.N + 1)]
        tmp_completed = [0 for _ in range(self.N + 1)]
        valid = True  

        for task in self.l:
            if tmp_completed[task] == -1:
                for i in self.after[task]:
                    tmp_completed[i] = -1
                continue
            
            for prereq in self.before[task]:
                if tmp_time[task] < tmp_time[prereq] + self.d[prereq]:
                    valid = False
                    break
            
            if not valid:
                tmp_completed[task] = -1
                for i in self.after[task]:
                    tmp_completed[i] = -1
                continue
            
            teams = [(max(tmp_s[i], tmp_time[task]), self.c[task][i], i) for i in range(1, self.M + 1) if self.c[task][i] > 0]
            if len(teams) == 0:
                tmp_completed[task] = -1
                for i in self.after[task]:
                    tmp_completed[i] = -1
                continue

            index = min(range(len(teams)), key=lambda i: teams[i])
            team = teams[index][2]

            tmp_completed[task] = team
            self.team_assignments[task] = team
            scheduled += 1
            tmp_time[task] = max(tmp_s[team], tmp_time[task])
            tmp_s[team] = tmp_time[task] + self.d[task]

            for i in self.after[task]:
                tmp_time[i] = max(tmp_time[i], tmp_time[task] + self.d[task])
            
            self.start_times[task] = tmp_time[task]

        if not valid:
            return float('inf'), float('inf')  

        total_time = 0
        total_cost = 0
        for i in range(1, self.N + 1):
            if tmp_completed[i] not in {-1, 0}:
                total_time = max(total_time, tmp_time[i] + self.d[i])
                total_cost += self.c[i][tmp_completed[i]]

        return total_time, total_cost


    def GetNeighbor(self, task):
        left = 0
        right = self.N - 1
        for t in self.before[task]:
            left = max(left, self.index[t])
        for t in self.after[task]:
            right = min(right, self.index[t])
        BeforeIndex = self.index[task]
        if left >= right - 1:
            return self
        AfterIndex = random.randint(left + 1, right - 1)
        tmp = self.l[:].pop(BeforeIndex)
        self.l[:].insert(AfterIndex, tmp)
        return State(self.l[:], self.N, self.M, self.s, self.d, self.c, self.pred, self.before, self.after)


    def GetIndex(self):
        index = [0 for _ in range(self.N + 1)]
        for i, task in enumerate(self.l):
            index[task] = i
        return index

    
class SimulatedAnnealing:
    def __init__(self):
        pass

    def Solve(self, N, M, s, d, c, pred, time_limit=30, step_limit=120):
        t = tm()
        topo = TopologicalSort(N, pred)
        before = [set() for _ in range(N + 1)]
        after = [set() for _ in range(N + 1)]
        for u, v in pred:
            after[u].add(v)
            before[v].add(u)
        state = State(topo, N, M, s, d, c, pred, before, after)
        step = 0
        temperature = 5000
        cooling_rate = 0.995
        while tm() - t <= time_limit and step <= step_limit:
            step += 1
            candidates = []
            
            for task in range(1, N + 1):
                neighbor = state.GetNeighbor(task)
                neighbor_eval = neighbor.GetEvaluation()
                current_eval = state.GetEvaluation()
                
                delta_eval = neighbor_eval[0] - current_eval[0]
                if delta_eval < 0:
                    candidates.append((neighbor, 'better'))
                elif delta_eval > 0:
                    candidates.append((neighbor, 'worse', math.exp(-delta_eval / temperature)))
            
            # Select one random neighbor from the candidates
            if candidates:
                selected = random.choice(candidates)
                selected_state, category, *prob = selected
                if category == 'better' or (category == 'worse' and random.random() < prob[0]):
                    state = selected_state
                
            temperature *= cooling_rate

        return state

N, Q, pred, d, M, s, K, c = read_input()

solver = SimulatedAnnealing()
state = solver.Solve(N, M, s, d, c, pred, time_limit=70000000)

# assigned_tasks = []
# for task in range(1, N + 1):
#     team = state.team_assignments[task]
#     start_time = state.start_times[task]
#     if team == -1:
#         continue
#     assigned_tasks.append((task, team, start_time))

# print(len(assigned_tasks))
# for task, team, start_time in assigned_tasks:
#     print(task, team, start_time)
total_time, total_cost = state.GetEvaluation()  
# print(total_time)  
