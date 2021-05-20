'''
Module in pyomo for abstract model charging ev
'''
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pandas as pd

import pickle
import gzip

import plotly.io as pio
pio.renderers.default = "browser"


model = pyo.AbstractModel()

model.T = pyo.Set() # set of time
model.F = pyo.Set(ordered=True) # F means set of features (as penalty/reward). see example of input data below: time, state, fast, ....

model.Δt = pyo.Param(within=pyo.NonNegativeReals, default=0.5) # 0.5 hrs
model.battery_capacity = pyo.Param(within=pyo.NonNegativeReals, default= 45.0) # kWh
model.charge_efficiency = pyo.Param(within=pyo.NonNegativeReals, default= 0.90) # percentage/fraction
model.min_SOC = pyo.Param(within=pyo.NonNegativeReals, default=0.0) # percentage/fraction
model.max_SOC = pyo.Param(within=pyo.NonNegativeReals, default=1.0) # percentage/fraction
model.init_SOC = pyo.Param(within=pyo.NonNegativeReals, default=0.5) # percentage/fraction
model.grid_availability = pyo.Param(model.T, within=pyo.NonNegativeReals, default=2) # default: 2 kW for each hour. See below, through input data can be modified.
model.driving_consumption = pyo.Param(model.T, within=pyo.NonNegativeReals, default=1) # kWh per time-step. Modify in input data by a time-series
model.weight = pyo.Param(model.T,model.F, default=1)

model.CHARGE_AT_BATTERY = pyo.Var(model.T, domain=pyo.NonNegativeReals)
model.SOC = pyo.Var(model.T, domain=pyo.NonNegativeReals)

def obj_expression(m):
    return pyo.quicksum(pyo.prod(m.weight[t,f] for f in m.F) * m.SOC[t] for t in m.T)

model.OBJ = pyo.Objective(rule=obj_expression, sense=1) # here sense -1 for maximization and 1 for minimization, default minimization


def SOC(m, t):
    if t == m.T[1]: # first timestep (Pyomo uses 1-based indexing)
        return m.SOC[t] == m.init_SOC + m.CHARGE_AT_BATTERY[t]*m.Δt/m.battery_capacity - m.driving_consumption[t]/m.battery_capacity
    else:
        return m.SOC[t] == m.SOC[t-1] + m.CHARGE_AT_BATTERY[t]*m.Δt/m.battery_capacity - m.driving_consumption[t]/m.battery_capacity
model.Constr_SOC = pyo.Constraint(model.T, rule=SOC)

def final_SOC(m, t):
    if t == m.T[len(m.T)]:
        return m.SOC[t] >= m.init_SOC
    else:
        return pyo.Constraint.Skip
model.Constr_finalSOC = pyo.Constraint(model.T, rule=final_SOC)

def max_SOC(m, t):
    return m.SOC[t] <= m.max_SOC
model.Constr_maxSOC = pyo.Constraint(model.T, rule=max_SOC)

def min_SOC(m, t):
    return m.SOC[t] >= m.min_SOC
model.Constr_minSOC = pyo.Constraint(model.T, rule=min_SOC)

def charge_st_availability(m, t):
    return m.CHARGE_AT_BATTERY[t] <= m.grid_availability[t]*m.charge_efficiency
model.Constr_charge_st_avail = pyo.Constraint(model.T, rule=charge_st_availability)

def charge_st_battery(m, t):
    return m.CHARGE_AT_BATTERY[t] <= m.battery_capacity/m.Δt
model.Constr_charge_st_batt = pyo.Constraint(model.T, rule=charge_st_battery)










# get consumption time-series and grid availability
with gzip.open('file.pickle') as f:
    dc = pickle.load(f)

df = dc['timeseries']
T = list(range(1,len(df)+1))

# this DF is to try to put some penalties or rewards to promote or discourage charging at certain hours or conditions
# for model.weight parameter
feat = pd.DataFrame()
feat.loc[:,'t'] = T
feat = feat.set_index('t')
feat.loc[:,'state'] = df['state'].apply(lambda x: 100 if x == 'home' else 1) # so far 1 does not make any effect. Future change to other values higher or lower than 1 (reward or penalty) based on cerain state
feat.loc[:,'time'] = [1]*len(df) # same here, but in future can be modified based on certain time
feat.loc[:,'fast'] = df['charging_point'].apply(lambda x: 100 if x in ['fast75', 'fast150'] else 1).values # if vehicle charge on fast charging station a penalty of 100 is multiplied to SOC, to make it bigger. As objective function intend to minimize. will dicourage charge at fast chrg station.
featdc = feat.stack().squeeze().to_dict()

cons = pd.DataFrame()
cons.loc[:,'t'] = T
cons = cons.set_index('t')
cons.loc[:,'value'] = df['consumption'].values
consdc = cons.squeeze().to_dict()

avail = pd.DataFrame()
avail.loc[:,'t'] = T
avail = avail.set_index('t')
avail.loc[:,'value'] = df['charging_cap'].values
availdc = avail.squeeze().to_dict()


# Alter default values providing all input values
input_data = {
    None:{
        'T':{None:T},
        'F':{None:feat.columns},
        'Δt':{None: dc['t']},
        'battery_capacity':{None: dc['battery_capacity']},
        'charge_efficiency':{None: dc['charging_eff']},
        'min_SOC':{None:0.1},  # dc['soc_min']
        'max_SOC':{None:1.0},
        'init_SOC':{None:0.5}, # dc['soc_init']
        'grid_availability': availdc,
        'driving_consumption': consdc,
        'weight': featdc,
    }
}

instance = model.create_instance(input_data)

opt = SolverFactory('glpk')

results = opt.solve(instance, tee=False)
results.write()
instance.solutions.load_from(results)

# print("Print values for all variables")
# for v in instance.component_data_objects(pyo.Var):
#     print(str(v), v.value)

df_list = []
for t in instance.T.data():
    frame = pd.DataFrame([{
                        'T': t,
                        'grid_availability':  pyo.value(instance.grid_availability[t]),
                        'CHARGE_AT_BATTERY': pyo.value(instance.CHARGE_AT_BATTERY[t]),
                        'SOC':  pyo.value(instance.SOC[t]),
                        'driving_consumption':  pyo.value(instance.driving_consumption[t]),
                        }])
    df_list.append(frame)

solution = pd.concat(df_list, ignore_index=True).set_index('T')
solution.to_csv('results.csv')


print()
if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
    print("Solver Status: ",  results.solver.status)
    print("Termination Condition: ",  results.solver.termination_condition)
elif (results.solver.termination_condition == TerminationCondition.infeasible):
    print("Solver Status: ",  results.solver.status)
    print("Termination Condition: ", results.solver.termination_condition)
else:
    # Something else is wrong
    print("Solver Status: ",  results.solver.status)


import plotly.express as px
fig = px.line(solution)
fig.show()
