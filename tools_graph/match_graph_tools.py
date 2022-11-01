import numpy as np
from scipy.signal import convolve2d
from scipy.spatial.distance import cdist
import math
from numpy.linalg import norm
import random
from tools_graph.plot_functs import plot_graph_matches_color,plot_graph_matches_color2
import pandas as pd


#def get_pattern_descriptor(adj_matrix, stacked_adj_features, node_pos,node_deg, weights=None):
def get_pattern_descriptor(nx_graph, node_pos, weights = None):
    def transform_edge_matrix_2_node_descriptor(matrix):
        max = np.max(matrix, axis=0)
        # mean = [np.mean(row[np.nonzero(row)]) for row in matrix]
        # min= [np.min(row[np.nonzero(row)]) for row in matrix]
        mean = np.zeros(np.shape(max))
        min = np.zeros(np.shape(max))
        for i, row in enumerate(matrix):
            idices_non_zero = np.nonzero(row)
            if np.shape(idices_non_zero)[1] == 0:
                mean[i] = 0
                min[i] = 0
            else:
                mean[i] = np.mean(row[idices_non_zero])
                min[i] = np.min(row[idices_non_zero])
        return np.vstack((mean, max, min))

    # function that calculates the angel between the edges
    def calculate_angles_between_edges(node_pos, node_dim, adj_matrix):
        matrix = np.zeros((node_dim.shape[0], 3))
        l = np.zeros((node_dim.shape[0], 3))
        neighbour = np.zeros((node_dim.shape[0], 3))
        for i in range(0, node_dim.shape[0], 1):
            # find nodes with more than one edge
            # define length for node of deg. = 1
            if node_dim[i] == 1:
                n_z = np.nonzero(adj_matrix[i])
                vec1 = node_pos[n_z[0][0]][0] - node_pos[i][0]
                vec2 = node_pos[n_z[0][0]][1] - node_pos[i][1]
                len = np.sqrt(np.power(vec1, 2) + np.power(vec2, 2))
                l[i] = [len, 0, 0]
            # define lenght,angle,neighbour
            if node_dim[i] > 1:
                # find nodes that share egdes with node[i]
                vec = np.zeros((int(node_dim[i]), 2))
                len = np.zeros((int(node_dim[i]), 1))
                n_z = np.nonzero(adj_matrix[i])
                # calculate Vector beween node i and nodes n_z
                ind = 0
                for j in n_z[0]:
                    vec[ind][0] = node_pos[j][0] - node_pos[i][0]
                    vec[ind][1] = node_pos[j][1] - node_pos[i][1]
                    len[ind] = np.sqrt(np.power(vec[ind][0], 2) + np.power(vec[ind][1], 2))
                    ind += 1
                for z in range(1, int(node_dim[i]), 1):
                    if (len[z] == 0) or (len[z - 1] == 0):
                        continue
                    erg = (vec[z - 1][0] * vec[z][0] + vec[z - 1][1] * vec[z][1]) / (len[z - 1] * len[z])
                    matrix[i][z - 1] = math.acos(round(erg[0], 5))
                    l[i][z - 1] = len[z - 1]
                # calculate angle between first and last node
                num = int(node_dim[i]) - 1
                if (len[0] == 0) or (len[num] == 0):
                    continue
                erg = (vec[0][0] * vec[num][0] + vec[0][1] * vec[num][1]) / (len[0] * len[num])
                matrix[i][num] = math.acos(round(erg[0], 5))
                l[i][num] = len[num]
        return matrix, l, neighbour

    def find_neighbour(node_dim, adj_matrix):
        direct_n = np.zeros((node_dim.shape[0], 3))
        indirect_n = np.zeros((node_dim.shape[0], 3))
        for i in range(0, node_dim.shape[0], 1):
            n_z = np.nonzero(adj_matrix[i])
            ind = 0
            #Knoten 1, direkte Nachbarn
            for j in n_z[0]:
                direct_n[i][ind] = node_dim[j]
                n_z_j = np.nonzero(adj_matrix[j])
                sum = 0
                #Knoten2, indirekte Nachbarn
                for z in n_z_j[0]:
                    sum += node_dim[z]
                    # #Knoten 3
                    # n_z_3 = np.nonzero(adj_matrix[z])
                    # sum3 = 0
                    # for t in n_z_3[0]:
                    #     sum+= node_dim[t]
                indirect_n[i][ind] = sum
                ind += 1
        neighbour = np.hstack((np.sort(direct_n, axis=1), np.sort(indirect_n, axis=1)))
        return neighbour

    # 3 28 49 57 58 63 66 67
    node_dim = np.sum(nx_graph[0], axis=0)
    print(node_dim)
    m, l,n = calculate_angles_between_edges(node_pos, node_dim, nx_graph[0])
    m = np.sort(m)
    l = np.sort(l)
    neighbour = find_neighbour(node_dim, nx_graph[0])
    len_descriptor = transform_edge_matrix_2_node_descriptor(nx_graph[1])
    c2_descriptor = transform_edge_matrix_2_node_descriptor(nx_graph[2])
    c3_descriptor = transform_edge_matrix_2_node_descriptor(nx_graph[3])
    descriptor = np.vstack((node_dim,m.transpose(),len_descriptor,l.transpose() ,neighbour.transpose())).transpose()

    if weights is not None:
        return descriptor * weights
    else:
        return descriptor

def match_pattern(descriptors1, descriptors2, max_distance=np.inf, cross_check=True, distance_ratio=None):
    distances = cdist(descriptors1, descriptors2, metric='euclidean')  # distances.shape: [len(d1), len(d2)]
    #print(distances)
    indices1 = np.arange(descriptors1.shape[0])  # [0, 1, 2, 3, 4, 5, 6, 7, ..., len(d1)] "indices of d1"
    indices2 = np.argmin(distances,
                         axis=1)  # [12, 465, 23, 111, 123, 45, 67, 2, 265, ..., len(d1)] "list of the indices of d2 points that are closest to d1 points"

    # Each d1 point has a d2 point that is the most close to it.
    if cross_check:
        '''
        Cross check idea:
        what d1 matches with in d2 [indices2], should be equal to 
        what that point in d2 matches with in d1 [matches1]
        '''
        matches1 = np.argmin(distances, axis=0)  # [15, 37, 283, ..., len(d2)] "list of d1 points closest to d2 points"
        # Each d2 point has a d1 point that is closest to it.
        # indices2 is the forward matches [d1 -> d2], while matches1 is the backward matches [d2 -> d1].
        mask = indices1 == matches1[indices2]  # len(mask) = len(d1)
        # we are basically asking does this point in d1 matches with a point in d2 that is also matching to the same point in d1 ?
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if distance_ratio is not None:
        '''
        the idea of distance_ratio is to use this ratio to remove ambigous matches.
        ambigous matches: matches where the closest match distance is similar to the second closest match distance
                          basically, the algorithm is confused about 2 points, and is not sure enough with the closest match.
        solution: if the ratio between the distance of the closest match and
                  that of the second closest match is more than the defined "distance_ratio",
                  we remove this match entirly. if not, we leave it as is.
        '''
        modified_dist = distances
        fc = np.min(modified_dist[indices1, :], axis=1)
        modified_dist[indices1, indices2] = np.inf
        fs = np.min(modified_dist[indices1, :], axis=1)
        mask = fc / fs <= 0.5
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    # sort matches using distances
    dist = distances[indices1, indices2]
    sorted_indices = dist.argsort()

    matches = np.column_stack((indices1[sorted_indices], indices2[sorted_indices]))
    return matches

# function that calculates the new positions after a rotation and translation
#th function uses to pairs of matches
def calculate_new_position(match1, match2, posi1, posi2):
    pos1 = posi1.copy()
    pos2 = posi2.copy()
    new_pos1 = np.zeros(np.shape(pos1))
    new_pos2 = np.zeros(np.shape(pos2))
    # Definiere Koorinatensystem mit Ursprung pos1[match1[0]]
    x11, y11 = pos1[match1[0]]-pos1[match1[0]]
    x12, y12 = pos1[match2[0]]-pos1[match1[0]]
    p21 = pos2[match1[1]]-pos1[match1[0]]
    p22 = pos2[match2[1]]-pos1[match1[0]]
    # ziehe alle Korrdinaten übereinander
    x21, y21 = p21-p21
    x22, y22 = p22-p21
    #bestimme Winkel
    l1 = np.sqrt(np.power(x12, 2) + np.power(y12, 2))
    l2 = np.sqrt(np.power(x22, 2) + np.power(y22, 2))
    r1,r2 = pos1[match2[0]]-pos1[match1[0]]/l1
    winkel = math.acos(round((x12*x22+y12*y22)/(l1*l2),5))
    ref_direction = np.array([x12,y12])/l1
    dir1 = rotate(np.array([x22,y22]),-winkel)
    dir2 = rotate(np.array([x22, y22]), winkel)
    if norm(dir1-ref_direction) < norm(dir2-ref_direction):
        winkel = -winkel

    c =0
    for new_pos in pos2:
        new_pos2[c] = new_pos - pos2[match1[1]]
        new_pos2[c] = rotate(new_pos2[c],winkel)
        c+=1

    c=0
    for new_pos in pos1:
        new_pos1[c] = new_pos - pos1[match1[0]]
        c+=1
    return new_pos1,new_pos2

#function that applies rotation to coordinates
def rotate(point,angle):
    px,py = point
    qx = math.cos(angle) * px - math.sin(angle) * py
    qy = math.sin(angle) * px + math.cos(angle) * py
    return qx, qy

def calculate_angle_for_nodes(adj_matrix, pos):
    #create matrix of size(adj_matrix)
    angle_matrix = np.zeros(np.shape(adj_matrix))
    #for every node in graph:
    for i in range(0,np.shape(pos)[0],1):
        #find neighbours of node[i]
        n_z =np.nonzero(adj_matrix[i])
        #for every neighbour node of node[i]:
        for j in n_z[0]:
            #creade coordinates system with node[i] as center
            p1, p2 = pos[j]-pos[i]
            l = np.sqrt(np.power(p1, 2) + np.power(p2, 2))
            s = np.sign(p1)
            if l ==0:
                continue
            angle_matrix[i][j] = s*math.degrees(math.acos((0*p1+1*p2)/(l*1)))
            if angle_matrix[i][j] ==0:
                angle_matrix[i][j]= 0.001
    return angle_matrix

    #def find_more_matches(self, pos1, pos2, angle_matrix, adj_matrix1, adj_matirx2):


def find_match_for_neighbour(match, angle_matrix1, angle_matrix2, already_matched_nodes1=[], already_matched_nodes2=[]):
    new_match = []
    # index for neighbours
    ind_node1 = np.nonzero(angle_matrix1[match[0]])
    for index in ind_node1[0]:
        if index in already_matched_nodes1:
            angle_matrix1[match[0]][index] = 0
    ind_node1 = np.nonzero(angle_matrix1[match[0]])
    print("Für", match[0], " ", ind_node1)
    ind_node2 = np.nonzero(angle_matrix2[match[1]])
    for index in ind_node2[0]:
        if index in already_matched_nodes2:
            angle_matrix2[match[1]][index] = 0
    ind_node2 = np.nonzero(angle_matrix2[match[1]])
    print("Für", match[1], " ", ind_node2)
    # values for neighbours
    n_z1 = angle_matrix1[match[0]][angle_matrix1[match[0]] != 0]
    n_z2 = angle_matrix2[match[1]][angle_matrix2[match[1]] != 0]
    print("G1 Knoten:", match[0], "Winkel:", n_z1)
    print("G2 Knoten:", match[1], "Winkel:", n_z2)

    erg = np.zeros((np.shape(n_z1)[0], np.shape(n_z2)[0]))
    ind1 = 0
    for i in n_z1:
        ind2 = 0
        for j in n_z2:
            erg[ind1][ind2] = np.sqrt(np.power(j - i, 2))
            ind2 += 1
        ind1 += 1
    if erg.any():
        indices2 = np.argmin(erg, axis=1)
        print("erg", erg)
        for i in range(0, np.shape(indices2)[0], 1):
            if erg[i][indices2[i]] < 10:
                new_match.append([ind_node1[0][i], ind_node2[0][indices2[i]]])
    return new_match


def spread_matches(start_match, ang1, ang2, node_dim, liste_mit_matches=[], list_with_aready_matched_nodes1=[],
                   list_with_aready_matched_nodes2=[]):
    print('observed node:', start_match[0])
    ma = find_match_for_neighbour(start_match, angle_matrix1=ang1, angle_matrix2=ang2,
                                  already_matched_nodes1=list_with_aready_matched_nodes1,
                                  already_matched_nodes2=list_with_aready_matched_nodes2)
    print('zurück gegebenes match', ma)
    list_with_aready_matched_nodes1.append(start_match[0])
    list_with_aready_matched_nodes2.append(start_match[1])
    # print('already matched:',list_with_aready_matched_nodes)
    # liste_mit_matches.append(ma)
    # print('list with matches:',liste_mit_matches)
    for i in ma:
        if i[0] in list_with_aready_matched_nodes1:
            print('Bereits gematched:', i[0])
            continue
        else:
            if node_dim[i[0]] == 1:
                liste_mit_matches.append(i)
                # continue
            else:
                liste_mit_matches.append(i)
                liste_mit_matches, list_with_aready_matched_nodes1, list_with_aready_matched_nodes2 = spread_matches(i,
                                                                                                                     ang1,
                                                                                                                     ang2,
                                                                                                                     node_dim,
                                                                                                                     liste_mit_matches,
                                                                                                                     list_with_aready_matched_nodes1,
                                                                                                                     list_with_aready_matched_nodes2)

    return liste_mit_matches, list_with_aready_matched_nodes1, list_with_aready_matched_nodes2


def test_matches(pos1, adj1,img1,pos2, adj2,img2,k):
    fish1 = fisheye_distortion1(pos2,np.shape(img2),k)
    d1 = get_pattern_descriptor(adj1,pos1)
    d2 = get_pattern_descriptor(adj2, fish1)
    matches = match_pattern(d1, d2)
    plot_graph_matches_color(img1, img2, pos1, adj1[0], fish1, adj2[0], matches)

    angle_matrix1 = calculate_angle_for_nodes(adj1[0], pos1)
    angle_matrix2 = calculate_angle_for_nodes(adj2[0], fish1)
    node_dim = np.sum(adj1[0], axis=0)
    final_matches_after_ransac = []

    for i in range(1, 200, 1):
        a = random.randrange(1, np.shape(matches)[0], 1)
        b = random.randrange(1, np.shape(matches)[0], 1)

        # spread matches
        new_matches = []
        new_matches, ng1, ng2 = spread_matches(matches[a], angle_matrix1, angle_matrix2, node_dim, liste_mit_matches=[],
                                               list_with_aready_matched_nodes1=[],
                                               list_with_aready_matched_nodes2=[])

        diff = compare_descriptors_of_matches(new_matches, d1, d2)
        erg1 = np.sum(diff < 10)
        if erg1 > 5:
            final_matches_after_ransac.append(matches[a])
            final_matches_after_ransac.append(matches[b])

    plot_graph_matches_color(img1, img2, pos1,adj1[0], fish1,adj2[0], final_matches_after_ransac)


def fisheye_distortion1(pos, img_size, k):
    # pull coordinates in a centered coordianates system
    new_p = np.zeros(np.shape(pos))
    for ind in range(0, len(pos), 1):
        new_p[ind] = pos[ind] - np.array([[img_size[0] / 2, img_size[1] / 2]])
        x, y = new_p[ind]
        du = np.sign(x) * math.sqrt(math.pow(x, 2) + math.pow(y, 2))
        if x == 0:
            angle = 0
        else:
            angle = math.atan(y / x)
        dd = du + k * math.pow(du, 3)
        yneu = dd * math.sin(angle)
        xneu = dd * math.cos(angle)
        new_p[ind] = np.array([[xneu, yneu]]) + np.array([[img_size[0] / 2, img_size[1] / 2]])
    return new_p


def compare_descriptors_of_matches(matches, d1, d2):
    c = 0
    diff = np.zeros((np.shape(matches)[0], 1))
    for ma in matches:
        a = d1[ma[0]]
        b = d2[ma[1]]
        diff[c] = norm(a - b)
        c += 1
    return diff
def get_matched_nodes1():
    a = np.array([[66,81],
             [71,86],
             [70,85],
             [68,84],
             [72,87],
             [51,69],
             [48,62],
             [46,56],[56,72],[47,57],[49,65],[58,47],[45,51],[60,75],[62,81],[37,37],[69,83],[76,90]])
    return a


def match_descriptors_in_clusters(pos1, pos2, d1, d2, img1, img2,adj1, adj2):
    img1_xsize, img1_ysize,z = np.shape(img1)
    img2_xsize, img2_ysize, z = np.shape(img2)
    finale_matches = []
    # define  nodes to match in image 1
    for lim1 in range(0, img1_xsize, round(img1_xsize / 6)):
        erg = 900
        x_untere_Grenze = lim1
        x_obere_Grenze = lim1 + round(img1_xsize / 3)
        for lim2 in range(0, img1_ysize, round(img1_ysize / 6)):
            y_untere_Grenze = lim2
            y_obere_Grenze = lim2 + round(img1_ysize / 3)
            # find nodes in this range
            node_numbers1 = []
            for p in range(0, np.shape(pos1)[0], 1):
                # print("Position:",pos1[p],pos1[p][0],pos1[p][1])
                if (x_untere_Grenze < pos1[p][0] < x_obere_Grenze):
                    if (y_untere_Grenze < pos1[p][1] < y_obere_Grenze):
                        node_numbers1.append(p)
            descriptors1 = []
            for ind in node_numbers1:
                descriptors1.append(d1[ind])
            if any(node_numbers1):
                print("Node numbers 1:", node_numbers1)
                for lim3 in range(0, img2_xsize, round(img2_xsize / 6)):
                    x_untere_Grenze2 = lim3
                    x_obere_Grenze2 = lim3 + round(img2_xsize / 3)
                    for lim4 in range(0, img2_ysize, round(img2_ysize / 6)):
                        y_untere_Grenze2 = lim4
                        y_obere_Grenze2 = lim4 + round(img1_ysize / 3)
                        # find nodes in this range
                        node_numbers2 = []
                        for p in range(0, np.shape(pos2)[0], 1):
                            if (x_untere_Grenze2 < pos2[p][0] < x_obere_Grenze2):
                                if (y_untere_Grenze2 < pos2[p][1] < y_obere_Grenze2):
                                    node_numbers2.append(p)
                        descriptors2 = []
                        for ind in node_numbers2:
                            descriptors2.append(d2[ind])
                        if any(node_numbers2):
                            print("\t 2:", node_numbers2)
                            distance = cdist(descriptors1, descriptors2, metric="euclidean")
                            indices = np.argmin(distance, axis=1)
                            m1 = []
                            for i in indices:
                                m1.append(node_numbers2[i])
                            matches = np.vstack((node_numbers1,np.transpose(m1))).transpose()
                            fig = plot_graph_matches_color2(img1, img2, pos1,adj1,pos2,adj2, matches,x_untere_Grenze,y_obere_Grenze,round(img1_xsize / 3),round(img1_ysize / 3),
                                                            x_untere_Grenze2,y_obere_Grenze2,round(img2_xsize / 3),round(img2_ysize / 3))
                            print("stop")
                            #fig.line([(lim1,lim2),(lim3,lim4)])

def calculate_lengh_between_nodes(adj, pos):
    erg = np.zeros((np.shape(adj)))
    for i in range(0, len(pos),1):
        n_z = np.nonzero(adj[i])
        for j in n_z[0]:
            erg[i][j] = norm(pos[i]-pos[j])
            if erg[i][j] ==0:
                erg[i][j] = 0.0001
    return erg


def select_nodes(length_m, adj, pos):
    number_to_delete = []
    for i in range(0, np.shape(length_m)[0], 1):
        n_z = length_m[i][length_m[i] != 0]
        if max(n_z) < 10:
            number_to_delete.append(-1 * i)
    number_to_delete = np.sort(number_to_delete)
    for k in number_to_delete:
        adj = np.delete(adj, -1 * k, 1)
        adj = np.delete(adj, -1 * k, 0)
        pos = np.delete(pos, -1 * k, 0)

    return adj, pos









if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    from time import time
    from skimage.feature import plot_matches
    from skimage.transform import pyramid_gaussian

    # Trying multi-scale
    N_LAYERS = 4
    DOWNSCALE = 2
    img1 = cv2.imread('sandbox/matching_test/sofa2.jpg', cv2.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv2.imread('sandbox/matching_test/sofa3.jpg', cv2.IMREAD_GRAYSCALE)  # trainImage
    #img1 = cv2.imread('images/chess3.jpg')
    original_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grays1 = list(pyramid_gaussian(gray1, downscale=DOWNSCALE, max_layer=N_LAYERS, multichannel=False))

   # img2 = cv2.imread('images/chess.jpg')
    original_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    grays2 = list(pyramid_gaussian(gray2, downscale=2, max_layer=4, multichannel=False))

    scales = [(i * DOWNSCALE if i > 0 else 1) for i in range(N_LAYERS)]
    features_img1 = np.copy(img1)
    features_img2 = np.copy(img2)

    kps1 = []
    kps2 = []
    ds1 = []
    ds2 = []
    ms = []
    for i in range(len(scales)):
        scale_kp1 = FAST(grays1[i], N=9, threshold=0.15, nms_window=3)
        kps1.append(scale_kp1 * scales[i])
        scale_kp2 = FAST(grays2[i], N=9, threshold=0.15, nms_window=3)
        kps2.append(scale_kp2 * scales[i])
        for keypoint in scale_kp1:
            features_img1 = cv2.circle(features_img1, tuple(keypoint * scales[i]), 3 * scales[i], (0, 255, 0), 1)
        for keypoint in scale_kp2:
            features_img2 = cv2.circle(features_img2, tuple(keypoint * scales[i]), 3 * scales[i], (0, 255, 0), 1)
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(grays1[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(features_img1)

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(grays2[i], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(features_img2)

        d1 = BRIEF(grays1[i], scale_kp1, mode='uniform', patch_size=8, n=100)
        ds1.append(d1)
        d2 = BRIEF(grays2[i], scale_kp2, mode='uniform', patch_size=8, n=100)
        ds2.append(d2)

        matches = match(d1, d2, cross_check=True)
        ms.append(matches)
        print('no. of matches: ', matches.shape[0])

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)

        plot_matches(ax, grays1[i], grays2[i], np.flip(scale_kp1, 1), np.flip(scale_kp2, 1), matches)
        plt.show()

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(features_img1)
    plt.subplot(1, 2, 2)
    plt.imshow(features_img2)
    plt.show()