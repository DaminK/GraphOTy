import numpy as np
import networkx as nx

def indicator_matrix_from_colors(colors, normalized=False):
    if not normalized:
        return np.array([[1 if i == colors[j] else 0 for j in range(len(colors))] for i in np.unique(colors)]).transpose()
    else:
        c, counts = np.unique(colors, return_counts=True)
        return np.array([[1/np.sqrt(counts[j]) if colors[i] == c[j] else 0 for i in range(len(colors))] for j in range(len(c))]).transpose()
    

def sample_directed_csEP(nodes_per_group, links_to_other_group):
    group_offset = []
    num_nodes = 0
    links_needed = np.zeros((np.sum(nodes_per_group), len(nodes_per_group)), dtype=int)
    for i in range(len(nodes_per_group)):
        for node in range(nodes_per_group[i]):
            links_needed[node + num_nodes, :] = links_to_other_group[i]
        group_offset.append(num_nodes)
        num_nodes += nodes_per_group[i]

    adjacency_matrix = np.zeros((np.sum(nodes_per_group), np.sum(nodes_per_group)), dtype=int)

    for group in range(len(nodes_per_group)):
        for node in range(group_offset[group], group_offset[group] + nodes_per_group[group]):
            for other_group in range(len(links_to_other_group[group])):
                chosen_links = np.random.permutation([(node, i) for i in range(group_offset[other_group], group_offset[other_group] + nodes_per_group[other_group]) if node != i])
                for link in list(chosen_links[:links_needed[node, other_group]]):
                    adjacency_matrix[link[0], link[1]] = 1
                    links_needed[node, other_group] -= 1
    
    assert np.all(links_needed == 0)


    return adjacency_matrix


def sample_csEP(nodes_per_group, links_to_other_group, multi_graph_okay=False):
    group_offset = []
    num_nodes = 0
    links_needed = np.zeros((np.sum(nodes_per_group), len(nodes_per_group)), dtype=int)
    for i in range(len(nodes_per_group)):
        for node in range(nodes_per_group[i]):
            links_needed[node + num_nodes, :] = links_to_other_group[i]
        group_offset.append(num_nodes)
        num_nodes += nodes_per_group[i]

    adjacency_matrix = np.zeros((np.sum(nodes_per_group), np.sum(nodes_per_group)), dtype=int)

    for group in range(len(nodes_per_group)):
        for node in range(group_offset[group], group_offset[group] + nodes_per_group[group]):
            for other_group in range(len(links_to_other_group[group])):
                chosen_links = np.random.permutation([(node, i) for i in range(group_offset[other_group], group_offset[other_group] + nodes_per_group[other_group]) if links_needed[i, group] > 0 and node != i])
                for link in list(chosen_links[:links_needed[node, other_group]]):
                    adjacency_matrix[link[0], link[1]] = 1
                    adjacency_matrix[link[1], link[0]] = 1
                    links_needed[node, other_group] -= 1
                    links_needed[link[1], group] -= 1
                while links_needed[node, other_group] > 0:
                    chosen_links = np.random.permutation([(node, i) for i in range(group_offset[other_group], group_offset[other_group] + nodes_per_group[other_group]) if links_needed[i, group] > 0])
                    link = chosen_links[0]
                    adjacency_matrix[link[0], link[1]] += 1
                    adjacency_matrix[link[1], link[0]] += 1 if link[0] != link[1] else 0
                    links_needed[node, other_group] -= 1
                    links_needed[link[1], group] -= 1 if node != link[1] else 0
                    
    
    group_offset.append(np.sum(nodes_per_group))
    #print(links_needed)
    #print([[np.sum([adjacency_matrix[i,j] for j in range(group_offset[group], group_offset[group+1])]) for group in range(len(nodes_per_group))] for i in range(len(adjacency_matrix))])
    links_to_other_group_per_node = []
    colors = []
    for i,g in enumerate(nodes_per_group):
        links_to_other_group_per_node.extend([links_to_other_group[i]] * g)
        colors.extend([i]*g)
    #print(links_to_other_group_per_node)
    assert np.all(adjacency_matrix @ indicator_matrix_from_colors(colors) == links_to_other_group_per_node)
    adjacency_matrix = make_simple_graph(adjacency_matrix, nodes_per_group, links_to_other_group, group_offset)
    #print([[np.sum([adjacency_matrix[i,j] for j in range(group_offset[group], group_offset[group+1])]) for group in range(len(nodes_per_group))] for i in range(len(adjacency_matrix))])
    #print(links_to_other_group_per_node)
    assert np.all(adjacency_matrix @ indicator_matrix_from_colors(colors) == links_to_other_group_per_node)
    assert len([(i,j) for i in range(len(adjacency_matrix)) for j in range(len(adjacency_matrix)) if adjacency_matrix[i,j] > 1 or (i==j and adjacency_matrix[i,j] > 0)]) == 0
    
    return adjacency_matrix.copy()

def make_simple_graph(adjacency_matrix, nodes_per_group, links_to_other_group, group_offset):
    #print(adjacency_matrix)
    diagonal_entries_to_fix = [(i,i) for i in range(len(adjacency_matrix)) if adjacency_matrix[i,i] > 0]
    for edge in diagonal_entries_to_fix:
        for i in range(adjacency_matrix[edge[0],edge[1]]):
            if adjacency_matrix[edge[0],edge[1]] >= 2:
                group_x = [g for g in range(len(nodes_per_group)) if edge[0] >= group_offset[g] and edge[0] < group_offset[g+1]][0]
                group_y = [g for g in range(len(nodes_per_group)) if edge[1] >= group_offset[g] and edge[1] < group_offset[g+1]][0]

                candidates_x = [n for n in range(group_offset[group_x], group_offset[group_x + 1]) if n != edge[1] and adjacency_matrix[edge[1], n] < 1]
                candidates_y = [n for n in range(group_offset[group_y], group_offset[group_y + 1]) if n != edge[0] and adjacency_matrix[n, edge[0]] < 1]

                choice = np.random.permutation([(x,y) for x in candidates_x for y in candidates_y if adjacency_matrix[x,y] >= 1 and x != y])[0]
                #print(edge, choice)
                adjacency_matrix[edge[0], choice[1]] = 1
                adjacency_matrix[choice[1], edge[0]] = 1
                adjacency_matrix[choice[0], edge[1]] = 1
                adjacency_matrix[edge[1], choice[0]] = 1
                
                adjacency_matrix[edge[0], edge[1]] -= 1
                adjacency_matrix[edge[1], edge[0]] -= 1 
                adjacency_matrix[choice[0], choice[1]] -= 1
                adjacency_matrix[choice[1], choice[0]] -= 1
                #print(adjacency_matrix)
            elif adjacency_matrix[edge[0],edge[1]] == 1:
                group = [g for g in range(len(nodes_per_group)) if edge[0] >= group_offset[g] and edge[0] < group_offset[g+1]][0]
                candidates = [n for n in range(group_offset[group], group_offset[group + 1]) if n != edge[0]]
                choice = np.random.permutation([(x,x) for x in candidates if adjacency_matrix[x,x] >= 1])[0]
                #print(edge, choice)
                adjacency_matrix[edge[0], choice[1]] += 1
                adjacency_matrix[choice[1], edge[0]] += 1
                adjacency_matrix[edge[1], edge[0]] -= 1 
                adjacency_matrix[choice[0], choice[1]] -= 1
                #print(adjacency_matrix)
            else:
                continue


            

    edges_to_fix =[(i,j) for i in range(len(adjacency_matrix)) for j in range(i+1,len(adjacency_matrix)) if adjacency_matrix[i,j] > 1 and i != j]
   

    for edge in edges_to_fix:
        for i in range(adjacency_matrix[edge[0],edge[1]]-1):
                group_x = [g for g in range(len(nodes_per_group)) if edge[0] >= group_offset[g] and edge[0] < group_offset[g+1]][0]
                group_y = [g for g in range(len(nodes_per_group)) if edge[1] >= group_offset[g] and edge[1] < group_offset[g+1]][0]

                candidates_x = [n for n in range(group_offset[group_x], group_offset[group_x + 1]) if n != edge[1] and adjacency_matrix[edge[1], n] < 1]
                candidates_y = [n for n in range(group_offset[group_y], group_offset[group_y + 1]) if n != edge[0] and adjacency_matrix[n, edge[0]] < 1]

                choice = np.random.permutation([(x,y) for x in candidates_x for y in candidates_y if adjacency_matrix[x,y] >= 1 and x != y])[0]
                #print(edge, choice)
                adjacency_matrix[edge[0], choice[1]] = 1
                adjacency_matrix[choice[1], edge[0]] = 1
                adjacency_matrix[choice[0], edge[1]] = 1
                adjacency_matrix[edge[1], choice[0]] = 1
                
                adjacency_matrix[edge[0], edge[1]] -= 1
                adjacency_matrix[edge[1], edge[0]] -= 1 
                adjacency_matrix[choice[0], choice[1]] -= 1
                adjacency_matrix[choice[1], choice[0]] -= 1
                #print(adjacency_matrix)
                if adjacency_matrix[choice[0], choice[1]] > 0:
                    return make_simple_graph(adjacency_matrix, nodes_per_group, links_to_other_group)

    

    return adjacency_matrix


def sample_uniform_csEP(size_classes, num_classes, max_edges=0):
    if max_edges == 0:
        max_edges = size_classes
    A_pi = np.zeros(shape=(num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            A_pi[i,j] = np.random.randint(0, min(size_classes - 1, max_edges))
            A_pi[j,i] = A_pi[i,j]


    return sample_csEP([size_classes] * num_classes, A_pi)

def make_symmetric(A):
    for i in range(A.shape[0]):
        for j in range(i+1,A.shape[1]):
            A[j,i] = A[i,j]
    return A

def sample_planted_role_model(c,n=[40]*3, omega_1=[], return_omega=False, verbose=0, directed=False, p_out=0.05):
    if len(omega_1) < 2:
        omega_1 = np.random.rand(len(n),len(n))
    omega = p_out * np.ones((c * len(n), c *len(n)))
    for i in range(c):
        lower = i*len(n)
        upper = (i+1) * len(n)
        omega[lower:upper, lower:upper] = omega_1
    # tmp1 = np.concatenate([omega_1] + [0.05 * np.ones_like(omega_1)]*2, axis=1)
    # tmp2 = np.concatenate([0.05*np.ones_like(omega_1)] + [omega_1] + [0.05* np.ones_like(omega_1)]*1, axis=1)
    # tmp3 = np.concatenate([0.05*np.ones_like(omega_1)] *2 + [omega_1], axis=1)
    # omega = np.concatenate([tmp1, tmp2, tmp3], axis=0)
    if verbose >= 1:
        print(omega)
    #A = np.mean([nx.to_numpy_array(nx.stochastic_block_model([40]*9, omega))for i in range(2)], axis=0)
    if not directed:
        A = nx.to_numpy_array(nx.stochastic_block_model(n*c, make_symmetric(omega), seed=np.random.seed()))
    else:
        raise Exception("not implemented. Must be undirected.")#A = directedSBM(n*c, omega)#nx.to_numpy_array(nx.stochastic_block_model([40]*9, omega))
    labels = np.array([[i%3]*40 for i in range(9)]).flatten()
    if return_omega:
        return A, omega
    return A

