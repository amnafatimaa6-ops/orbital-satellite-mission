
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from stable_baselines3 import PPO

st.set_page_config(page_title="ORBITAL | Mission Control",
                   page_icon="🛰️", layout="wide")

st.title("🛰️ ORBITAL — Autonomous Earth Observation")
st.caption("PPO-powered observation scheduling | Synthetic research simulator")

MODEL_PATH = os.path.join(os.path.dirname(__file__),
                          "satellite_capture_ppo.zip")

@st.cache_resource
def load_model():
    return PPO.load(MODEL_PATH, device="cpu")

model = load_model()

TARGETS = [
    {"name": "Target A", "lat": 15, "lon": -45},
    {"name": "Target B", "lat": 5, "lon": 65},
    {"name": "Target C", "lat": -30, "lon": 145},
]

def reset_mission(seed, cloud_min, cloud_max):
    rng = np.random.default_rng(seed)
    return {
        "rng": rng,
        "priority": rng.uniform(0.2, 1.0, 3),
        "cloud": rng.uniform(cloud_min, cloud_max, 3),
        "energy": 1.0,
        "step": 0,
        "reward": 0.0,
        "successes": 0,
        "failures": 0,
        "recharges": 0,
        "history": []
    }

def get_obs(s):
    return np.concatenate([
        s["priority"], s["cloud"],
        [s["energy"], (40-s["step"])/40]
    ]).astype(np.float32)

def take_step(s, action):
    rng = s["rng"]
    reward = -0.02
    success = False
    target_name = "Recharge"

    if action < 3:
        target_name = TARGETS[action]["name"]
        if s["energy"] < 0.12:
            reward -= 0.25
        else:
            s["energy"] -= 0.12
            p_success = 0.95 * (1 - s["cloud"][action])
            success = rng.random() < p_success

            if success:
                reward += float(s["priority"][action])
                s["successes"] += 1
                s["priority"][action] *= 0.60
            else:
                reward -= 0.10
                s["failures"] += 1
    else:
        s["energy"] = min(1.0, s["energy"] + 0.35)
        s["recharges"] += 1
        reward -= 0.08

    s["cloud"] = np.clip(
        s["cloud"] + rng.normal(0, 0.08, 3), 0, 1)
    s["priority"] = np.clip(
        s["priority"] + rng.normal(0, 0.04, 3), 0, 1)

    s["step"] += 1
    s["reward"] += reward

    s["history"].append({
        "step": s["step"],
        "action": target_name,
        "success": success,
        "reward": reward,
        "total_reward": s["reward"],
        "energy": s["energy"],
        "cloud": s["cloud"].copy(),
        "priority": s["priority"].copy(),
        "successes": s["successes"],
        "failures": s["failures"],
        "recharges": s["recharges"]
    })

def choose_action(policy, obs, rng):
    if policy == "PPO":
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    if policy == "Random":
        return int(rng.integers(0, 4))
    if obs[6] < 0.24:
        return 3
    return int(np.argmax(obs[:3] * (1 - obs[3:6])))

def run_mission(policy, seed, cloud_min, cloud_max):
    s = reset_mission(seed, cloud_min, cloud_max)
    for _ in range(40):
        obs = get_obs(s)
        action = choose_action(policy, obs, s["rng"])
        take_step(s, action)
    return s

def xyz(lat, lon, radius=1.0):
    lat, lon = np.radians(lat), np.radians(lon)
    return (radius*np.cos(lat)*np.cos(lon),
            radius*np.cos(lat)*np.sin(lon),
            radius*np.sin(lat))

def globe_figure(record):
    # Textured-looking blue sphere using a clean 3D surface
    u = np.linspace(0, 2*np.pi, 70)
    v = np.linspace(0, np.pi, 45)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "#071a36"], [0.5, "#176b87"],
                    [1, "#5bb8ad"]],
        showscale=False, opacity=0.96,
        lighting=dict(ambient=0.65, diffuse=0.7),
        hoverinfo="skip", name="Earth"
    ))

    # Orbit line, deliberately illustrative rather than physical
    theta = np.linspace(0, 2*np.pi, 220)
    orbit_x = 1.18*np.cos(theta)
    orbit_y = 1.18*np.sin(theta)
    orbit_z = 0.30*np.sin(theta*2)
    fig.add_trace(go.Scatter3d(
        x=orbit_x, y=orbit_y, z=orbit_z,
        mode="lines", line=dict(color="#f4d35e", width=3),
        name="Illustrative orbit"
    ))

    # Observation targets: color represents current cloud fraction
    tx, ty, tz, colors, labels = [], [], [], [], []
    for i, target in enumerate(TARGETS):
        a, b, c = xyz(target["lat"], target["lon"], 1.025)
        tx.append(a); ty.append(b); tz.append(c)
        colors.append(float(record["cloud"][i]))
        labels.append(
            f'{target["name"]}<br>Priority: '
            f'{record["priority"][i]:.2f}<br>Cloud: '
            f'{record["cloud"][i]:.0%}'
        )

    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz, mode="markers+text",
        text=[t["name"] for t in TARGETS],
        textposition="top center",
        marker=dict(size=8, color=colors, cmin=0, cmax=1,
                    colorscale="YlOrRd",
                    line=dict(color="white", width=1),
                    colorbar=dict(title="Cloud", len=0.55)),
        hovertext=labels, hoverinfo="text", name="Targets"
    ))

    # Satellite marker advances around the illustrative orbit by step
    t = record["step"] / 40 * 2*np.pi
    sx, sy, sz = 1.18*np.cos(t), 1.18*np.sin(t), 0.30*np.sin(2*t)
    fig.add_trace(go.Scatter3d(
        x=[sx], y=[sy], z=[sz],
        mode="markers",
        marker=dict(size=7, color="#ffffff",
                    symbol="diamond", line=dict(color="#f4d35e", width=3)),
        name="Satellite"
    ))

    fig.update_layout(
        height=540, margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="#07111f",
        font=dict(color="white"),
        scene=dict(
            bgcolor="#07111f",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.01)
    )
    return fig

with st.sidebar:
    st.header("Mission controls")
    policy = st.selectbox("Control policy", ["PPO", "Greedy", "Random"])
    weather = st.selectbox("Cloud scenario", ["Normal", "Harsh"])
    seed = st.number_input("Mission seed", min_value=0,
                           max_value=999999, value=42, step=1)
    cloud_min, cloud_max = (
        (0.0, 0.8) if weather == "Normal" else (0.6, 1.0)
    )
    run = st.button("▶ Run 40-step mission", type="primary",
                    use_container_width=True)
    benchmark = st.button("Compare all policies (30 missions each)",
                          use_container_width=True)

if run:
    st.session_state.mission = run_mission(
        policy, int(seed), cloud_min, cloud_max)
    st.session_state.policy = policy
    st.session_state.weather = weather

if "mission" not in st.session_state:
    st.info("Choose a policy and click **Run 40-step mission** to begin.")
    initial = reset_mission(int(seed), cloud_min, cloud_max)
    st.plotly_chart(globe_figure({
        "step": 0, "cloud": initial["cloud"],
        "priority": initial["priority"]
    }), use_container_width=True)
else:
    s = st.session_state.mission
    st.subheader(
        f'{st.session_state.policy} policy · '
        f'{st.session_state.weather} clouds'
    )
    r = s["history"][-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mission reward", f'{s["reward"]:.2f}')
    c2.metric("Successful captures", s["successes"])
    c3.metric("Failed captures", s["failures"])
    c4.metric("Final energy", f'{s["energy"]:.0%}')

    step = st.slider("Mission playback — step", 1, 40, 40)
    rec = s["history"][step-1]
    left, right = st.columns([1.5, 1])

    with left:
        st.plotly_chart(globe_figure(rec), use_container_width=True)
    with right:
        st.markdown("### Current step")
        st.metric("Action", rec["action"])
        st.metric("Step reward", f'{rec["reward"]:.3f}')
        st.metric("Energy", f'{rec["energy"]:.0%}')
        st.metric("Successful captures so far", rec["successes"])
        st.metric("Failed attempts so far", rec["failures"])
        st.metric("Recharge actions so far", rec["recharges"])

    df = pd.DataFrame(s["history"])
    st.markdown("### Mission reward over time")
    st.line_chart(df.set_index("step")["total_reward"])
    st.markdown("### Action log")
    st.dataframe(df[["step", "action", "success", "reward",
                     "total_reward", "energy"]],
                 use_container_width=True, hide_index=True)

if benchmark:
    rows = []
    with st.spinner("Running 90 simulated missions..."):
        for scenario, lo, hi in [
            ("Normal", 0.0, 0.8), ("Harsh", 0.6, 1.0)
        ]:
            for pol in ["PPO", "Greedy", "Random"]:
                rewards, captures = [], []
                for sd in range(7000, 7030):
                    result = run_mission(pol, sd, lo, hi)
                    rewards.append(result["reward"])
                    captures.append(result["successes"])
                rows.append({
                    "Weather": scenario, "Policy": pol,
                    "Mean reward": np.mean(rewards),
                    "Reward std": np.std(rewards),
                    "Mean successful captures": np.mean(captures)
                })
    dfb = pd.DataFrame(rows)
    st.markdown("### Policy benchmark")
    st.dataframe(dfb.round(3), use_container_width=True, hide_index=True)
    st.bar_chart(dfb, x="Policy", y="Mean reward", color="Weather")

st.divider()
st.caption(
    "Research prototype: stochastic synthetic environment, not a physical "
    "orbital simulator. The 3D orbit is illustrative; target coordinates "
    "are visualization anchors. PPO was trained for 30,000 timesteps."
)
