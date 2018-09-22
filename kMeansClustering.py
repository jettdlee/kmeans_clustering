'''
########################################################################################################################
Assignment 2 - K-means clustering algorithm
Created by Jet-Tsyn Lee in PyCharm Community Edition 4.5.2
Created 13/03/18, last updated 06/04/18

Required:
	Python 3.6.1
	Numpy 1.13.1
	Matplotlib 2.1.2

k-Means program to identify the clustering of a number of datasets
########################################################################################################################
'''


import numpy as np
import matplotlib.pyplot as plt
import math

# #######  K-MEANS CLUSTERING OBJECT  #######
class k_mean_cluster(object):


    # #######  CONSTRUCTOR  #######
    def __init__(self, k_clusters, data_set, type, classes=[], l2_norm=False):

        # Store variables
        self.k = k_clusters         # k #. of clusters
        self.centres = []           # centre array
        # selected distance type to use, euclidean 'E', manhattan 'M', or cosine similarity 'C'
        self.type = type
        self.classes = classes      # Store the number of different classes


        # ===== STORE DATASET =====
        # Format the imported data set to store into object, and combine data files to one array
        data_temp = []

        # Loop all data sets to combine
        for iArr in range(len(data_set)):
            # loop each data point
            for jInput in range(len(data_set[iArr])):

                data_input = data_set[iArr][jInput]    # Get input

                # Normalise the data to L2 form if selected
                if l2_norm == True:
                    # from the input, compute the unit (L2) norm of the vector, x/||x||2
                    data_input = data_input / np.sqrt(np.sum(data_input ** 2))

                # Include the class label to the data input
                if classes:
                    data_input = np.append(data_input, classes[iArr])

                # Append data input into temp array
                data_temp.append(data_input)

        # Convert to numpy array, and shuffle to ensure randomness every run
        data_temp = np.asarray(data_temp)
        np.random.shuffle(data_temp)


        self.dataset = data_temp        # Store data set in object
        self.n = len(self.dataset)      # Store number of data inputs
        self.pairs = (self.n * (self.n-1))/2    # Number of data set pairs

        # =====  INITIALISE CENTRES  =====
        self.centres = self.initialize_centres(self.k, self.dataset)

        # Copy centres for contingency check
        self.cen_old = np.zeros_like(self.centres)

        # Create array to store cluster lables, excluding actual label column
        self.cluster = np.zeros(len(self.dataset[:][:-1]))

    def initialize_centres(self, k, data_set):
        centres = []

        # Loop for the number of clusters
        for _ in range(0, k):
            centre_feature = []     # Array to store features
            # Loop for the number of features in data, excluding label
            for iFeature in range(len(data_set[0][:-1])):

                # get all values for the set feature
                data_col = data_set[:, iFeature]

                # randomly select value between the min/max of the feature column
                c_feature = np.random.uniform(np.min(data_col), np.max(data_col))

                # Store in tempoary array
                centre_feature.append(c_feature)

            # Append array to return
            centres.append(centre_feature)

        # Convert to Numpy array and return centres
        return np.array(centres)


    # #######  RUN ALGORITHM  #######
    # Run the k-means clustering algorithm
    def initialize_k_means(self, iterations):

        # =====  K-MEANS ALGORITHM  =====
        # Loop for the number of iterations, or until centres converges
        for _ in range(iterations):

            # ~~~~~  ASSIGNEMENT  ~~~~~
            # Assign all plots to the nearest cluster
            self.cluster = self.assignment(self.dataset, self.centres, self.type)

            # ~~~~~  UPDATE CENTRES  ~~~~~
            # Update centre positions to the means of the cluster
            self.cen_old = np.copy(self.centres)    # Save current centres as old
            self.centres = self.update_centres(self.dataset, self.cluster)

            # ~~~~~  CHECK CONVERGENCE  ~~~~~
            # Compare the new and old centres to check if the points has converged
            if np.all(self.cen_old == self.centres):
                break   # Exif for loop


        # =====  SET CLUSTER  =====
        # set the cluster label based on the label corresponding to the most points in the cluster
        #self.set_cluster_labels()


        # =====  CONFUSION MATRIX  =====
        # Get confusion matrix values to apply to the P/R/F formulas
        tp, fp, tn, fn = self.confusion_matrix()


        # =====  EVALUATION  =====
        precision = compute_precision(tp, fp)
        recall = compute_recall(tp, fn)
        f_score = compute_f_score(precision, recall)


        # =====  PRINT RESULTS  =====
        print("\n =====  K = %d =====" % self.k)
        print("TP:", tp,"FN:", fn,"FP:", fp,"TN:", tn)
        print("Precision =", precision, "\nRecall =", recall, "\nF-Score =", f_score)

        # Return macro average values to be plotted in graph
        return [self.k, precision, recall, f_score]


    # #######  OBJECT FUNCTIONS  #######

    # Update centre values by calculating mean of each point in the cluster
    def update_centres(self, data_set, clusters):
        new_centres = []        # Store new centres
        k = len(self.centres)   # No of clusters

        # Loop each cluster
        for iCluster in range(0, k):

            # Get data only if assigned to the specific cluster
            labeled_data = data_set[clusters == iCluster]

            # Check if data is avalible in cluster
            if len(labeled_data) == 0:
                # if no assignments, set as zero array
                mean = np.zeros(len(self.centres[iCluster])+1)
            else:
                # calculate the means of each feature
                mean = np.mean(labeled_data, axis=0)

            # store mean in new centres, excluding label column
            new_centres.append(mean[0:-1])

        # Return new centres
        return np.array(new_centres)


    # Assign data inputs to nearest cluster
    def assignment(self, dataset, centres, type = "E"):

        # Array to store the cluster number of the data input
        clusters = np.zeros(len(dataset))

        # Loop all data inputs
        for i in range(len(dataset)):

            data = dataset[i][:-1]      # Get data input value
            distances = np.zeros(len(centres))  # Create array to store distances from data input to all clusters

            # Loop all clusters
            for jCentre in range(len(centres)):
                centre = centres[jCentre]   # Get centre value
                #dist = np.zeros_like(centre)

                # Depending on the distance type to run, two different formulas will run to calcuate distance
                if type == "E":     # Euclidean
                    dist = compute_euclidean_distance(data, centre)
                elif type == "M":   # Manhattan
                    dist = compute_manhattan_distance(data, centre)
                elif type == "C":   # Cosine Similarity Distance
                    dist = compute_cosine_similarity_distance(data, centre)

                # Store distance in array
                distances[jCentre] = dist

            # Get the cluster that reults in the minimum distance between the data input and centre
            # Store cluster in array
            clusters[i] = np.argmin(distances)

        # Return array containing the assigned clusters
        return clusters


    # Set cluster label basen on most data points assigned in cluster
    def set_cluster_labels(self):

        # Reset cluster labels
        self.cluster_lbl = np.zeros(self.k)

        # Loop for each cluster
        for iCluster in range(self.k):

            # get data inputs for the specific cluster
            clust_label = self.dataset[self.cluster == iCluster,-1]
            # Create array to store count
            lbl_count = np.zeros(len(self.classes))

            # for each input in cluster, count the actual labels by incrementing the array index
            for lbl in clust_label:
                lbl_count[int(lbl)] += 1

            # Assign cluster label based on the most points counted in cluster
            self.cluster_lbl[iCluster] = np.argmax(lbl_count)




    def confusion_matrix(self):

        # Binomial Theorem formula, nCr
        def nCr(n, r):
            return math.factorial(n)/(math.factorial(r)*math.factorial(n-r))

        #for each cluster count the number of class points in the cluster
        cluster_class_count = []
        for i in range(0, self.k):
            clustering = self.dataset[self.cluster == i, -1]   # Filter for labels in the set cluster
            class_label = np.zeros(len(self.classes))     # array to store count

            # loop each label in cluster, and increment count
            for j in clustering:
                class_label[int(j)] += 1

            # add to data array
            cluster_class_count.append(class_label)
        cluster_class_count = np.array(cluster_class_count)


        # =====  True Positive + False Positive  =====
        # Identify all positive pairs, i.e. calculate all points in cluster
        tp_fp = 0.0

        # for each cluster
        for iCluster in range(len(cluster_class_count)):

            # Count all points in the current cluster
            cluster_sum = np.sum(cluster_class_count[iCluster])

            if cluster_sum <= 1:
                tp_fp += 0
            else:
                # if pairs identified, i.e. 2+ points in cluster, calculate nCr and sum
                tp_fp += nCr(cluster_sum, 2)


        # =====  True Positive  =====
        tp = 0.0
        # For each cluster
        for iCluster in range(len(cluster_class_count)):
            # For each class in the cluster
            for jClass in range(len(cluster_class_count[iCluster])):

                # if there are a matching pair of points in the cluster ,2+
                if cluster_class_count[iCluster, jClass] >= 2:
                    cluster_sum = cluster_class_count[iCluster, jClass]
                    if cluster_sum <= 1:
                        tp += 0
                    else:
                        # sum pair count
                        tp += nCr(cluster_sum, 2)


        # =====  False Positive  =====
        # Calculated by the difference between the true positives and all positives
        fp = tp_fp - tp


        # =====  True Negative + False Negative  =====
        # All pairs - all positives
        tn_fn = self.pairs - tp_fp


        # =====  False Negatives  =====
        fn = 0.0
        # For all clusters and classes
        for iCluster in range(len(cluster_class_count)):
            for jClass in range(len(cluster_class_count[i])):
                # if a point has been clustered
                if cluster_class_count[iCluster, jClass] > 0:

                    # calculate the sum of all point in the same class but in different clusters
                    mismatched = cluster_class_count[iCluster + 1:, jClass]

                    # Sum the mismatched values
                    fn += cluster_class_count[iCluster, jClass] * np.sum(mismatched)


        # =====  True Negatives  =====
        # Calculated by the difference between the False Negatives and all negatives
        tn = tn_fn - fn

        # Return results
        return tp, fp, tn, fn




# #######  PUBLIC FUNCTIONS  #######

def compute_euclidean_distance(data_x, data_y):
    # sqrt(sum((x-y)^2))
    return np.sqrt(np.sum((data_x - data_y)**2))


def compute_manhattan_distance(data_x, data_y):
    # sum(|x-y|)
    return np.sum(np.abs(data_x - data_y))


def compute_cosine_similarity_distance(data_x, data_y):
    # x^T*Y/||x||||y|| = sum(x * y)/sqrt(x^2)*sqrt(y^2)
    numerator = np.sum(data_x * data_y)
    denominator = np.sqrt(np.sum(data_x ** 2)) * np.sqrt(np.sum(data_y ** 2))
    similarity = numerator/denominator

    distance = np.arccos(similarity)/np.pi  # Angular Cosine difference
    #distance = 1-similarity     # Cosine difference

    return distance


def compute_precision(tp, fp):
    if (tp + fp) == 0:
        return 0
    else:
        pres = tp / (tp + fp)
        return pres


def compute_recall(tp, fn):
    if (tp + fn) == 0:
        return 0
    else:
        return tp / (tp + fn)


def compute_f_score(p, r):
    if (p + r) == 0:
        return 0
    else:
        return (2 * p * r)/(p + r)


# Plot Data results into a single graph
def plot_graph(data, title=""):

    data = np.array(data)   # convert to NP to allow array slice

    # Slice Data, array index depends on results returned from run function
    x_cluster = data[:,0]       # #. k clusters
    y_precision = data[:,1]     # macro-average precision
    y_recall = data[:,2]        # macro-average recall
    y_f_score = data[:,3]       # macro-average f-score

    fig, ax = plt.subplots()

    # Plot Macro average, precision, recall, f-score
    line1, = ax.plot(x_cluster, y_precision, '-o', linewidth=2, label='Precision')
    line2, = ax.plot(x_cluster, y_recall, '-D',label='Recall')
    line3, = ax.plot(x_cluster, y_f_score, '-x', label='F-Score')

    # Graph properties
    plt.title(title)
    plt.xlabel('k-clusters')
    plt.ylabel('Value')
    ax.legend(loc='lower right')
    plt.xticks(np.arange(1, 11, step=1))

    # Show graph
    plt.show()


# ##############   MAIN   ##############
if __name__ == '__main__':

    # =======  INITIALIZE  =======

    # File Names
    path_animal = 'animals'
    path_countries = 'countries'
    path_fruits = 'fruits'
    path_veggies = 'veggies'

    # Import data to numpy array, ignore first value from array
    animal_file = np.genfromtxt(path_animal, delimiter=" ")[:, 1:]
    countries_file = np.genfromtxt(path_countries, delimiter=" ")[:,1:]
    fruits_file = np.genfromtxt(path_fruits, delimiter=" ")[:,1:]
    veggies_file = np.genfromtxt(path_veggies, delimiter=" ")[:,1:]

    # Variables
    classes = [0, 1, 2, 3]  # Number labels to be included in data to determine data labels
    data_files = [animal_file, countries_file, fruits_file, veggies_file]   # Array of all data

    loop_limit = 10         # Max number of k clusters
    max_iterations = 500    # set number of iterations if k-means loop doesn't converge to avoid inf loop


    # =====  Q2) EUCLIDEAN Distance  =====

    print("\n\n#######    EUCLIDEAN DISTANCE    #######")
    results_euc = []    # Array to store the macro-average results from run

    # Loop for the number of k Clusters to check, +1 to include last number
    for k in range(1, loop_limit+1):
        # Create object storing the variables
        kmeans = k_mean_cluster(k, data_files, "E", classes)

        # Run k-Means clustering, returning the macro average results from test and store in array
        results_euc.append(kmeans.initialize_k_means(max_iterations))


    # =====  Q3) EUCLIDEAN Distance, with L2 normalized data  =====

    print("\n\n#######    EUCLIDEAN DISTANCE, L2 Normalized data    #######")
    results_euc_l2 = []

    for k in range(1, loop_limit+1):
        kmeans = k_mean_cluster(k, data_files, "E", classes, True)
        results_euc_l2.append(kmeans.initialize_k_means(max_iterations))


    # =====  Q4) MANHATTAN Distance  =====

    print("\n\n#######    MANHATTAN DISTANCE    #######")
    results_man = []

    for k in range(1, loop_limit+1):
        kmeans = k_mean_cluster(k, data_files, "M", classes)
        results_man.append(kmeans.initialize_k_means(max_iterations))


    # =====  Q5) MANHATTAN Distance, with L2 normalized data  =====

    print("\n\n#######    MANHATTAN DISTANCE, L2 Normalized data    #######")
    results_man_l2 = []

    for k in range(1, loop_limit+1):
        kmeans = k_mean_cluster(k, data_files, "M", classes, True)
        results_man_l2.append(kmeans.initialize_k_means(max_iterations))


    # =====  Q6) COSINE SIMILARITY  =====

    print("\n\n#######    COSINE SIMILARITY DISTANCE    #######")
    results_cos = []

    for k in range(1, loop_limit+1):
        kmeans = k_mean_cluster(k, data_files, "C", classes)
        results_cos.append(kmeans.initialize_k_means(max_iterations))


    # =====  GRAPHS  =====
    # Plot graphs from the results from the test

    plot_graph(results_euc, "EUCLIDEAN DISTANCE")
    plot_graph(results_euc_l2, "EUCLIDEAN DISTANCE, L2 Normalized data")
    plot_graph(results_man, "MANHATTAN DISTANCE")
    plot_graph(results_man_l2, "MANHATTAN DISTANCE, L2 Normalized data")
    plot_graph(results_cos, "COSINE SIMILARITY DISTANCE")
