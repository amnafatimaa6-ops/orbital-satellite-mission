# orbital-satellite-mission
# 🛰️ ORBITAL — Reinforcement Learning for Autonomous Earth Observation

**An interactive research prototype for energy-aware satellite observation scheduling using Proximal Policy Optimization (PPO).**

[🚀 Live Demo](https://orbital-satellite-mission.streamlit.app/) · [💻 GitHub Repository](https://github.com/amnafatimaa6-ops/orbital-satellite-mission)

---

## 🌍 Overview

ORBITAL explores how a reinforcement learning agent can make sequential observation decisions in a simulated Earth-observation mission while balancing observation opportunities, cloud conditions, and limited onboard energy.

The project combines a custom Gymnasium environment, a trained PPO agent, baseline policy comparisons, robustness experiments, and an interactive 3D-inspired Streamlit dashboard.

**Research question:** How effectively can a reinforcement learning policy schedule observations under changing cloud conditions and energy constraints?

> **Scope:** ORBITAL is a synthetic research simulator, not a physical orbital mechanics model. The 3D orbit is illustrative, and the target coordinates are visualization anchors.

## ✨ Features

- **PPO agent:** Trained using Stable-Baselines3 for 30,000 timesteps.
- **Custom environment:** Eight-dimensional observation space and four discrete actions.
- **Three policies:** Compare PPO, Greedy, and Random decision-making.
- **Weather scenarios:** Evaluate missions under normal and harsher cloud conditions.
- **Mission playback:** Inspect decisions, rewards, capture outcomes, and energy over time.
- **Interactive dashboard:** Explore mission outcomes and policy benchmarks using Streamlit and Plotly.
- **Saved model:** The trained PPO model is included in `satellite_capture_ppo.zip`.

## 🧠 Environment Design

### Observation space

The agent observes eight values:

- Three target-priority values
- Three cloud-fraction values
- Remaining energy
- Normalized time remaining

### Action space

| Action | Description |
|---|---|
| Observe Target A | Attempt an observation of target A |
| Observe Target B | Attempt an observation of target B |
| Observe Target C | Attempt an observation of target C |
| Recharge | Restore energy, subject to the environment's rules |

### Reward design

The reward function encourages useful observations while accounting for energy use, failed capture attempts, recharging, and time spent. Target priorities and cloud conditions evolve during a mission.

This design provides a controlled setting for investigating sequential decision-making under uncertainty.

## 📊 Experimental Results

The PPO agent was evaluated against Greedy and Random baselines across repeated missions.

### Initial 100-mission comparison

| Policy | Mean reward | Successful captures per mission | Failed attempts per mission |
|---|---:|---:|---:|
| PPO | 1.617 | 22.27 | 8.73 |
| Greedy | 1.725 | 17.97 | 13.03 |
| Random | 0.010 | 14.02 | 11.74 |

*Results are from the initial evaluation run in the synthetic environment.*

### Paired comparison: PPO vs. Greedy

Using 100 matched random seeds:

- PPO mean reward: approximately 1.806
- Greedy mean reward: approximately 1.807
- 95% confidence interval for the paired reward difference: approximately −0.178 to 0.177
- PPO wins: 49 missions; Greedy wins: 51 missions

The paired experiment did **not** demonstrate a statistically significant reward advantage for PPO over Greedy in that evaluation.

### Robustness under harsher clouds

In the recorded harsh-cloud evaluation:

| Policy | Mean reward | Successful captures per mission | Failed attempts per mission |
|---|---:|---:|---:|
| PPO | 0.032 | 13.02 | 17.97 |
| Greedy | 0.197 | 10.99 | 20.01 |
| Random | -2.029 | 7.12 | 18.09 |

Performance degraded under harsher conditions, illustrating the challenge of scheduling observations when cloud cover makes successful captures less likely.

**Interpretation:** PPO substantially outperformed Random on mean reward in the initial evaluation and achieved more successful captures than Greedy. However, its overall reward was similar to Greedy in the paired comparison. These findings apply to this synthetic environment and do not establish superiority in real satellite operations.

## 🖥️ Interactive Demo

The Streamlit application allows users to:

1. Select PPO, Greedy, or Random.
2. Choose normal or harsh cloud conditions.
3. Set a mission seed.
4. Run a simulated mission and inspect its results.
5. Replay mission steps and examine the reward curve and action log.
6. Compare policies across weather scenarios.

**[Launch ORBITAL Mission Control](https://orbital-satellite-mission.streamlit.app/)**

If the demo URL changes, use the app URL provided by your Streamlit Community Cloud deployment.

## 🛠️ Tech Stack

- Python
- Gymnasium
- Stable-Baselines3
- Proximal Policy Optimization (PPO)
- NumPy and Pandas
- Streamlit
- Plotly

## 📁 Repository Structure

```text
orbital-satellite-mission/
├── app.py
├── requirements.txt
├── satellite_capture_ppo.zip
└── README.md
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/amnafatimaa6-ops/orbital-satellite-mission.git
cd orbital-satellite-mission
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the application

```bash
streamlit run app.py
```

The application will provide a local URL in the terminal.

## ⚠️ Limitations

- The environment is synthetic; it does not use live satellite telemetry or real orbital dynamics.
- Cloud conditions, energy, target priorities, and capture outcomes are simulated.
- The illustrative 3D visualization is not a validated orbital trajectory.
- PPO did not demonstrate a statistically significant reward advantage over Greedy in the recorded paired evaluation.
- Results depend on the environment design, reward function, training configuration, and evaluation procedure.

## 🔭 Future Work

- Evaluate across more random seeds and training runs.
- Investigate reward-function design and energy-aware scheduling.
- Explore richer cloud dynamics and observation constraints.
- Compare additional reinforcement learning algorithms.
- Investigate connections to real Earth-observation datasets and operational scheduling requirements.

## 👩‍💻 Author

**Amna Fatima**

ORBITAL is an independent research and engineering project exploring reinforcement learning, autonomous decision-making, and Earth-observation scheduling.

---

*Built as a research prototype for experimentation and learning. Not intended for operational satellite control.*
