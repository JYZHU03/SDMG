import torch
import dgl.function as fn

def mean_aggregation(graph, node_features):
    with graph.local_scope():
        graph.ndata['h'] = node_features
        graph.update_all(message_func=fn.copy_u('h', 'm'),
                         reduce_func=fn.mean('m', 'h'))
        return graph.ndata['h']

def compute_k_hop_aggregated_features(graph, node_features, k):
    h = node_features
    for _ in range(k):
        h = mean_aggregation(graph, h)
    return h

def compute_ms_ssim_loss(graph, original_features, reconstructed_features, scales, weights):
    N = original_features.shape[0]
    sim_list = []
    alpha = 2

    sim_feature = (1 - (reconstructed_features * original_features).sum(dim=-1)).pow_(alpha)
    sim_feature = (sim_feature + 1) / 2
    sim_list.append(sim_feature ** weights[0])

    for i, k in enumerate(scales):
        orig_agg_feats = compute_k_hop_aggregated_features(graph, original_features, k)
        recon_agg_feats = compute_k_hop_aggregated_features(graph, reconstructed_features, k)
        sim_k = (1 - (recon_agg_feats * orig_agg_feats).sum(dim=-1)).pow_(alpha)
        sim_k = (sim_k + 1) / 2
        sim_list.append(sim_k ** weights[i + 1])

    ms_sim = torch.ones(N).to(original_features.device)
    for sim in sim_list:
        ms_sim = ms_sim * sim

    loss = ms_sim.mean()

    return loss