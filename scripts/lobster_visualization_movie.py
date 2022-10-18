import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from itertools import product
import io 
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

from unc.utils import load_info
from unc.utils.data import save_video
from unc.agents import DQNAgent
from definitions import ROOT_DIR
from unc.envs.wrappers.lobster.belief import get_lobster_state_map


def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


if __name__ == "__main__":
    data_path = Path(ROOT_DIR, 'results', 'lobster_data.npy')
    fa = 'linear'
    lobster_data = np.load(data_path, allow_pickle=True).item()
    pb_data = lobster_data['2pb']

    # Get optimal lobster states
    optimal_lobster_fpath = Path(ROOT_DIR, 'results', 'optimal_lobster_results.npy')

    optimal_lobster_res = load_info(optimal_lobster_fpath)
    optimal_q = optimal_lobster_res['qs']
    state_to_idx = optimal_lobster_res['state_to_idx']
    zero_states = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    optimal_zero_qs = []
    for zero_state in zero_states:
        idx = state_to_idx[str(zero_state)]
        optimal_zero_qs.append(optimal_q[idx])
    optimal_zero_qs = np.stack(optimal_zero_qs)

    # get pf lobster states
    tol = 0.001
    pb_state_map = get_lobster_state_map()

    pb_obs = pb_data['obs'].reshape(-1, pb_data['obs'].shape[-1])
    # pb_unique_obs = np.unique(np.floor(pb_obs / tol).astype(int), axis=0) * tol
    pb_zero_obs_mask = (pb_obs[:, :4].sum(axis=-1) > 0) & (pb_obs[:, 4:].sum(axis=-1) == 0)
    pb_zero_obs = pb_obs[pb_zero_obs_mask]
    # pb_0_obs = np.unique(pb_unique_obs[:, :4], axis=0)
    r1_pb_states = [pb_state_map[0, 1, 0], pb_state_map[0, 1, 1]]
    r2_pb_states = [pb_state_map[0, 0, 1], pb_state_map[0, 1, 1]]


    # Likelihood Predictions
    # all possible states for 2e
    counts = np.arange(201)
    rate = 1 / 10
    likelihoods = np.exp(-counts * rate)
    all_possible_likelihoods = np.array(list(product(likelihoods, likelihoods)))
    all_zero_obs_2e = np.zeros((all_possible_likelihoods.shape[0], 9))
    all_zero_obs_2e[:, [4, 7]] = all_possible_likelihoods


    # so for unc encoding, 1 == observable and collected.
    ot_obs_2g = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    # For particle filter belief state, we simply find distribution over first four states
    r1r2 = np.mgrid[0:1.:0.05, 0:1.:0.05].reshape(2,-1).T
    r1_and_r2 = r1r2[:, 0] * r1r2[:, 1]
    not_r1_and_r2 = np.expand_dims(1 - r1_and_r2, -1)
    r1_and_r2 = np.expand_dims(r1_and_r2, -1)

    zero_obs_2pb = np.concatenate([not_r1_and_r2, r1r2, r1_and_r2, np.zeros((r1r2.shape[0], 12 - 4))], axis=-1)

    # here we get all possible observations at node 0 for 2
    zero_obs_2 = np.array([[1., 0., 0., 0., 0., 1., 0., 0., 1.]])

    # all possible observations at node 0 for 2o
    discount = 0.95

    obs_2o_range_single = discount ** (np.arange(300) + 1)
    obs_2o_range_x, obs_2o_range_y = np.meshgrid(obs_2o_range_single, obs_2o_range_single)
    obs_2o_range = np.stack((obs_2o_range_x, obs_2o_range_y), axis=-1)
    ot_obs_2o = obs_2o_range.reshape(-1, 2)

    zero_obs_2o = np.repeat(zero_obs_2, ot_obs_2o.shape[0], axis=0)
    zero_obs_2o[:, [3, 6]] = ot_obs_2o

    # get all agents
    obs_agent_fname = Path(ROOT_DIR, 'results', f'2_{fa}_agent.pth')
    unc_agent_fname = Path(ROOT_DIR, 'results', f'2o_{fa}_agent.pth')
    pb_agent_fname = Path(ROOT_DIR, 'results', f'2pb_{fa}_agent.pth')
    pred_agent_fname = Path(ROOT_DIR, 'results', f'2e_{fa}_agent.pth')

    obs_agent = DQNAgent.load(obs_agent_fname, DQNAgent)
    unc_agent = DQNAgent.load(unc_agent_fname, DQNAgent)
    pb_agent = DQNAgent.load(pb_agent_fname, DQNAgent)
    pred_agent = DQNAgent.load(pred_agent_fname, DQNAgent)

    # get all q values
    all_zero_2_qs = obs_agent.Qs(zero_obs_2, obs_agent.network_params)
    all_zero_2o_qs = unc_agent.Qs(zero_obs_2o, unc_agent.network_params)
    all_zero_2pb_qs = pb_agent.Qs(pb_zero_obs, pb_agent.network_params)
    all_zero_2e_qs = pred_agent.Qs(all_zero_obs_2e, pred_agent.network_params)[:, :2]

    # Normalize
    range_2 = all_zero_2_qs.max() - all_zero_2_qs.min()
    range_2o = all_zero_2o_qs.max() - all_zero_2o_qs.min()
    range_optimal = optimal_zero_qs.max() - optimal_zero_qs.min()
    range_2pb = all_zero_2pb_qs.max() - all_zero_2pb_qs.min()
    range_2e = all_zero_2e_qs.max() - all_zero_2e_qs.min()

    normalized_2_qs = (all_zero_2_qs - all_zero_2_qs.min()) / range_optimal
    normalized_2o_qs = (all_zero_2o_qs - all_zero_2o_qs.min()) / range_2o
    normalized_optimal_qs = (optimal_zero_qs - optimal_zero_qs.min()) / range_optimal
    normalized_2pb_qs = (all_zero_2pb_qs - all_zero_2pb_qs.min()) / range_2pb
    normalized_2e_qs = (all_zero_2e_qs - all_zero_2e_qs.min()) / range_2e


    # now we get our video
    fps = 36
    video_duration = 10

    # actions_to_plot = [0]
    # action_sets = [[0], [0, 1]]
    actions_to_plot = [0, 1]
    # actions_to_plot = [0]

    algs = ['2o', '2pb', '2e']
    # algs = ['2o']

    frames = fps * video_duration
    twopi = 2 * np.pi
    rads_per_frame = np.pi / 180


    x_pos = -1.7
    y_pos = -0.85

    og_camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.2),
        eye=dict(x=x_pos, y=y_pos, z=0.6)
    )

    # Make our movie here!

    show_legend = False

    for alg in algs:
        all_frames = []
        camera = deepcopy(og_camera)

        video_path = Path(ROOT_DIR, 'results', f'{alg}_3d_scatter_{actions_to_plot}.mp4')
        print(f"Making movie for {alg} to {video_path}")

        for frame in tqdm(range(frames)):
            action_mapping = ['Left', 'Right', 'Collect']
            actions_to_color = ['rgb(241, 196, 15)', 'rgb(52, 152, 219)']
            if show_legend:
                fig_path = Path(ROOT_DIR, 'results', f'lobster_interpolation_{alg}_{actions_to_plot}_legend.pdf')
            else:
                fig_path = Path(ROOT_DIR, 'results', f'lobster_interpolation_{alg}_{actions_to_plot}.pdf')

            fig = go.Figure(layout=go.Layout(
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=show_legend,
                font=dict(size=18),
                scene = dict(
                    xaxis = dict(
                        backgroundcolor="rgb(255, 255, 255)",
                        gridcolor="rgb(189, 195, 199)",
                        title=r'r(L1) feature',
                        range=[-0.1, 1.1],
                        tickvals=[0, 1],
                        tickangle=0
                    ),
                    yaxis = dict(
                        backgroundcolor="rgb(255, 255, 255)",
                        gridcolor="rgb(189, 195, 199)",
                        title=r'r(L2) feature',
                        range=[-0.1, 1.1],
                        tickvals=[0, 1],
                        tickangle=0
                    ),
                    zaxis = dict(
                        backgroundcolor="rgb(255, 255, 255)",
                        gridcolor="rgb(189, 195, 199)",
                        title="Normalized Q",
                        range=[-0.05, 1.05],
                        tickvals=[0, 1],
                        tickangle=0
                    ),
                ),

            ))

            for action in actions_to_plot:
                if alg == '2o':
                    z_2o = normalized_2o_qs[:, action]
                    trace_2o = go.Scatter3d(
                        x=1 - ot_obs_2o[:, 0],
                        y=1 - ot_obs_2o[:, 1],
                        z=z_2o,
                        name=f"Exp Trace",
                    #         name=f"{action_mapping[action]}",
                        mode='markers',
                        marker={
                            'size': 2,
                            'color': actions_to_color[action],
                            'symbol': 'circle'
                        }
                    )
                    fig.add_trace(trace_2o)
                elif alg == '2pb':
                    z_2pb = normalized_2pb_qs[:, action]
                    trace_2pb = go.Scatter3d(
                        x=pb_zero_obs[:, r1_pb_states].sum(axis=-1),
                        y=pb_zero_obs[:, r2_pb_states].sum(axis=-1),
                        z=z_2pb,
                        name=f"PF",
                        mode='markers',
                        marker={
                            'size': 2,
                            'color': actions_to_color[action],
                            'symbol': 'circle'
                        }
                    )
                    fig.add_trace(trace_2pb)
                # elif alg == '2f':
                #     z_2f = normalized_2f_qs[:, action]
                #     trace_2f = go.Scatter3d(
                #         x=gvf_zero_predictions_normalized[:, 0],
                #         y=gvf_zero_predictions_normalized[:, 1],
                #         z=z_2f,
                #         name=f"GVF",
                # #         name=f"{action_mapping[action]}",
                #         mode='markers',
                #         marker={
                #             'size': 2,
                #             'color': actions_to_color[action],
                #             'symbol': 'circle'
                #         }
                #     )
                #     fig.add_trace(trace_2f)
                elif alg == '2e':
                    z_2e = normalized_2e_qs[:, action]
                    trace_2e = go.Scatter3d(
                        x=all_possible_likelihoods[:, 0],
                        y=all_possible_likelihoods[:, 1],
                        z=z_2e,
                        name=f"Likelihood",
                #         name=f"{action_mapping[action]}",
                        mode='markers',
                        marker={
                            'size': 2,
                            'color': actions_to_color[action],
                            'symbol': 'circle'
                        }
                    )
                    fig.add_trace(trace_2e)

                z_optimal = normalized_optimal_qs[:, action]
                trace_optimal = go.Scatter3d(
                    x=zero_states[:, 1],
                    y=zero_states[:, 2],
                    z=z_optimal,
                    name=f"Ground-truth state",
                    mode='markers',
                    marker={
                        'size': 10,
                        'color': actions_to_color[action],
                        'symbol': 'cross',
                        'line': dict(width=0.5, color="black")
                    }
                )
                fig.add_trace(trace_optimal)

                z_2 = normalized_2_qs[:, action]
                trace_2 = go.Scatter3d(
                    x=np.array([0]),
                    y=np.array([0]),
                    z=z_2,
                    name=f"Observations",
                    mode='markers',
                    marker={
                        'size': 5,
                        'color': actions_to_color[action],
                        'symbol': 'diamond',
                        'line': dict(width=0.5, color="black")

                    }
                )
                fig.add_trace(trace_2)

            fig.update_layout(scene_camera=camera)

            all_frames.append(plotly_fig2array(fig))

            # Rotate our camera
            prev_x, prev_y = camera['eye']['x'], camera['eye']['y']
            rho, prev_angle = cart2pol(prev_x, prev_y)
            new_angle = (prev_angle + rads_per_frame) % twopi
            new_x, new_y = pol2cart(rho, new_angle)
            camera['eye']['x'] = new_x
            camera['eye']['y'] = new_y

        all_frames = np.stack(all_frames)
        save_video(all_frames, video_path, fps=fps)

