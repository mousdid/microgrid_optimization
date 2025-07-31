# SmartGrid: A Resilient RL-Based Controller for Microgrid Optimization

## Overview

This project explores the integration of **Deep Reinforcement Learning (DRL)** for **real-time microgrid optimization**, addressing resilience under disturbances such as grid outages, battery failures, and load spikes. It introduces a **Proximal Policy Optimization (PPO)** controller trained and evaluated on a real-world-inspired hybrid microgrid scenario, benchmarked against a traditional **Mixed-Integer Linear Programming (MILP)** baseline.

## Motivation

Traditional MILP-based optimization is effective for planning but lacks adaptability during dynamic events. With rising renewable penetration and unpredictable disruptions, microgrid controllers must respond **robustly and intelligently**. This work adopts the **DIRE framework** (Disturbance and Impact Resilience Evaluation) and leverages RL to design a more **adaptive, cost-efficient, and fault-tolerant control system**.

---

## Microgrid Components

The simulated microgrid includes:

- Diesel Generator (DG)
- Combined Heat and Power (CHP)
- Battery Energy Storage (ES)
- Wind Turbine (WT)
- Photovoltaic Panels (PV)
- Electric Vehicles (EVs)
- Grid import/export interface

---

## Framework Architecture

### 1. **MILP Benchmark**
- Formulated a short-horizon MILP optimization problem
- Minimized operational cost under physical and logical constraints (load balance, SOC, heat demand, EV scheduling, etc.)
- Implemented using Pyomo and solved with Gurobi

### 2. **DRL Controller**
- PPO agent trained with Stable-Baselines3
- State space includes load, renewable capacity, EV status, SOC, and scenario flags
- Action space: continuous control over 7 key units
- Reward function combines five components:
  - Cost minimization
  - Load balance
  - Battery SOC compliance
  - EV SOC fulfillment
  - Heat demand satisfaction

### 3. **Scenario Simulation**
- Injected failures such as:
  - Grid outages (24–72h)
  - Battery failures (48–72h)
  - Load spikes (1–3h)
- Train/test split: 80% training, 20% testing, plus dedicated 48-hour scenario for MILP evaluation

---

## Dataset

- **KU-HMG1** hybrid microgrid dataset (Payra, Bangladesh)
- Features:
  - Hourly load (8761 hours)
  - PV and WT generation
  - SOC data
  - EV availability (simulated)
  - LMP pricing (from PJM-RTO)
  - TOU-based EV charging pricing (from California’s EV tariff)
  - Simulated residential heat demand

---

## Results

### ✅ PPO vs MILP (48-Hour Horizon)
| Model        | Total Cost ($) |
|--------------|----------------|
| MILP         | -80.73         |
| PPO Model 1  | 884.56         |
| PPO Model 2  | 452.29         |
| PPO Model 3  | 19.94          |

### ✅ PPO vs Rule-Based (1752-Hour Horizon)
| Model          | Total Cost ($) |
|----------------|----------------|
| Rule-Based     | 11,866.36      |
| PPO (Best)     | 1,094.50       |

- PPO agent achieved **>90% cost reduction** vs. rule-based baseline
- Demonstrated adaptive response to outages and failures
- PPO showed stable dispatch, thermal balance, and EV integration
- Reward shaping (penalty tuning) significantly impacted long-term behavior

---

## Technologies Used

- **Python**, **NumPy**, **Pandas**, **Matplotlib**
- **Pyomo** (for MILP modeling)
- **Stable-Baselines3** (PPO)
- **Gurobi** (MILP solver)
- **Gym** (RL environment)
- Custom scenario generator and evaluation pipeline

---

## DIRE Resilience Framework Mapping

| Resilience Phase | DRL Behavior                                              |
|------------------|------------------------------------------------------------|
| Resistance       | Prepares through policy learning and safe operation limits |
| Response         | Reacts to outages with battery dispatch                    |
| Recovery         | Gradually stabilizes SOC and costs after events            |
| Restoration      | Resumes optimal operation in normal conditions             |

---

## Future Work

- Extend to **networked microgrids** and multi-agent PPO
- Integrate **Safe RL** and **barrier certificates**
- Real-time deployment on **embedded hardware**
- Investigate **meta-learning** for dynamic reward tuning

---

## References

1. DOE Microgrid Strategy – [Link](https://www.energy.gov/oe/microgrid-program-strategy)
2. Gautam et al., DRL for Resilient Energy Systems, *Electricity*, 2023
3. Mirbarati et al., MILP for Hybrid Microgrids, *Sustainability*, 2022
4. KU-HMG1 Dataset – [Mendeley Data](https://data.mendeley.com/datasets/x8v796pjsx/3)

---

## Author

**Mohamed Yassir Ousdid**  
Master's in Computational Decision Science and Operations Research  
Illinois Institute of Technology – Chicago, USA  
📧 mousdid@hawk.illinoistech.edu

---

> This work bridges predictive and prescriptive analytics for resilient energy systems using AI-powered control strategies.
