
from functools import partial
import networkx as nx
from itertools import chain, combinations, permutations
import numpy as np
import torch
torch.set_printoptions(precision=16)
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import edmonds_cpp

def clip_range(x, max_range=np.inf):
    m = torch.max(x, axis=-1, keepdim=True)[0]
    return torch.max(x, -1.0 * torch.tensor(max_range) * torch.ones_like(x) + m)

def submatrix_index(n, i):
    I = torch.ones((n, n), dtype=bool)
    I[i, :] = False
    I[:, i] = False
    return I

def get_spanning_tree_marginals(logits, n):
    (i, j) = torch.triu_indices(n, n, offset=1)
    c = torch.max(logits, axis=-1, keepdims=True)[0]
    k = torch.argmax(logits, axis=-1)
    
    removei = i[k]

    weights = torch.exp(logits - c)

    W = torch.zeros(n, n)
    W = W.cuda() if logits.is_cuda else W
    W[i, j] = weights
    W[j, i] = weights

    L = torch.diag_embed(W.sum(axis=-1)) - W
    subL = L[submatrix_index(n, removei)].view(n - 1, n - 1)
    logzs = torch.slogdet(subL)[1]
    #logzs = torch.logdet(subL)
    logzs = torch.sum(logzs + (n - 1) * c.flatten())
    sample = torch.autograd.grad(logzs, logits, create_graph=True)[0]
    return sample

def edmonds_cpp_pytorch(adjs, n):
    """
    Gets the maximum spanning arborescence given weights of edges.
    We assume the root is node (idx) 0.
    Args:
        adjs: shape (batch_size, n, n), where 
            adjs[.][i][j] is the weight for edge j -> i.
        n: number of vertices.
    Returns:
        heads: Size (batch_size, n). 
            heads[i] = parent node of i; heads[0] = 0 always.
    """
    heads = edmonds_cpp.get_maximum_spanning_arborescence(adjs.unsqueeze(0), n)
    return heads

def edmonds_python(adjs, n):
    """
    Gets the maximum spanning arborescence given weights of edges.
    We assume the root is node (idx) 0.
    Args:
        adjs: shape (batch_size, n, n), where 
            adjs[.][i][j] is the weight for edge j -> i.
        n: number of vertices.
    Returns:
        heads. Size (batch_size, n). heads[0] = 0 always.
    """
    # Convert roots and weights_and_edges to numpy arrays on the cpu.
    if torch.is_tensor(adjs):
        adjs = adjs.detach().to("cpu").numpy()

    # Loop over batch dimension to get the maximum spanning arborescence for
    # each graph.
    batch_size = adjs.shape[0]
    heads = np.zeros((batch_size, n))
    for sample_idx in range(batch_size):
        # We transpose adj because networkx accepts adjacency matrix
        # where adj[i][j] corresponds to edge i -> j.
        np.fill_diagonal(adjs[sample_idx], 0.0)
        # We multiply by -1.0 since networkx obtains the
        # minimum spanning arborescence. We want the maximum.
        G = nx.from_numpy_matrix(-1.0 * adjs[sample_idx].T, create_using=nx.DiGraph())
        
        Gcopy = G.copy()
        # Remove all incoming edges for the root such that
        # the given "root" is forced to be selected as the root.
        Gcopy.remove_edges_from(G.in_edges(nbunch=[0]))
        msa = nx.minimum_spanning_arborescence(Gcopy)
        
        # Convert msa nx graph to heads list.
        for i, j in msa.edges:
            i, j = int(i), int(j)
            heads[sample_idx][j] = i
     
    heads = torch.from_numpy(heads)
    return heads

def sample_tree_from_logits(logits, tau=1.0, hard=False, max_range=np.inf, device="cpu"):

    """Samples a maximum spanning tree given logits.
    Args:
        logits: Logits of shape (n * (n - 1), 1).
            They correspond to a flattened and transposed adjacency matrix
            with the diagonals removed.
            We assume the logits are edge-symmetric.
        tau: Float representing temperature.
        hard: Whether or not to sample hard edges.
        hard_with_grad: Whether or not to allow sample hard, but have gradients
            for backprop.
        relaxation: Relaxation type.
        max_range: Maxiumum range between maximum edge weight and any other
            edge weights. Used for relaxation == "exp_family_entropy" only.
    Returns:
        Sampled edges with the same shape as logits, and
        sampled edge weights of same shape as logits.
    """

    # n * (n - 1) = len(logits), where n is the number of vertices.
    n =  int(0.5 * (1 + np.sqrt(8 * logits.size(0) + 1)))

    # Reshape to adjacency matrix (with the diagonals removed).
    #reshaped_logits = logits.view(n, n - 1)
    #reshaped_logits = reshaped_logits.transpose(0, 1)  # (n-1, n)

    #vertices = torch.triu_indices(n-1, n, offset=1)
    #edge_logits = reshaped_logits[vertices[0], vertices[1]]

    """
    uniforms = torch.empty_like(edge_logits).float().uniform_().clamp_(EPS, 1 - EPS)
    gumbels = uniforms.log().neg().log().neg()
    gumbels = gumbels.cuda() if logits.is_cuda else gumbels
    """
    edge_weights = logits
    
    vertices = torch.triu_indices(n, n, offset=1)

    #hard = True if hard_with_grad else hard
    if hard:
        adj = torch.zeros((n,n)).to(device)
        #adj[vertices[0], vertices[1]] = torch.exp(edge_weights)
        #adj[vertices[1], vertices[0]] = torch.exp(edge_weights)
        #adj[vertices[0], vertices[1]] = edge_weights
        adj[vertices[1], vertices[0]] = edge_weights
        
        #samples = edmonds_cpp_pytorch(adj, n)
        heads = edmonds_python(adj.unsqueeze(0), n).flatten()
        
        samples = torch.zeros((n,n)).to(device)
        for col in range(n):
            if col > 0:
                samples[int(heads[col]), col] = 1

    if not hard:
        weights = edge_weights / tau
        weights = clip_range(weights, max_range)
        X = get_spanning_tree_marginals(weights, n)

        #samples = torch.zeros_like(reshaped_logits)
        samples = torch.zeros((n,n)).to(device)
        samples[vertices[0], vertices[1]] = X
        #samples[vertices[1], vertices[0]] = X

    # Return the flattened sample in the same format as the input logits.
    #samples = samples.transpose(0,1).contiguous().view(n * (n - 1))

    # Make sampled edge weights into adj matrix format.
    #edge_weights_reshaped = torch.zeros_like(reshaped_logits)
    #edge_weights_reshaped[vertices[0], vertices[1]] = edge_weights
    #edge_weights_reshaped[vertices[1] - 1, vertices[0]] = edge_weights
    #edge_weights = edge_weights_reshaped.transpose(0, 1).contiguous().view(logits.shape)

    #return samples, edge_weights
    return samples