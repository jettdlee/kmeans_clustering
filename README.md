# K-Means Clustering
##### For Assignments COMP527 Data Mining, MSc Computer Science, University of Liverpool

Implementation of the k-Mean clustering algorithm to cluster words belonging to four categories: animals, countries,
fruits and veggies, arranged into four different files. 
The first entry in each line is a word followed by 300 features (word embedding) describing the meaning of that word.

Implementation of the k-means is tested with k values between 1-10 and the precision, recall, and F-score is computed for each set of clusters, which is then plotted in a graph.

Several tests are computed with varying distence measures:
* Euclidean distance
* Euclidean distance with a unit L2 normilization
* Manhattan distance
* Manhattan distance with a unit L2 normilization
* Cosine Similarity distance

Created and tested in PyCharm Community Edition 4.5.2

Ensure following software applications is avalible:
	Python 3.6.4
	Numpy 1.13.1
	Matplotlib 2.1.2
	(Optional) Python IDE, i.e. PyCharm, IDLE, Visual studio etc

Required files:
	kMeansClustering.py
	animals
	countries
	veggies
	fruits

Please save k-Means clustering python file 'kMeansClustering.py' and required data files (animals, countries, veggies, fruits) in the same selected folder.
If path or data file name requires adjustment, please adjust variables (path_animal, path_countries, path_fruits, path_veggies) respectivly within the 
'kMeansClustering.py' script.

Depending on your running tool, please follow methods accordingly:


Command Prompt (cmd.exe):
1) Open command prompt, and change directory to the saved file path of the python script using 'cd'. 
e.g.	M:>cd 
	M:\kMeansClustering>

2) Input 'python kMeansClustering.py' in the window to run the script.
e.g.	M:\kMeansClustering>python kMeansClustering.py

IDE (PyCharm, IDLE etc.):
1) In selected IDE, open 'kMeansClustering.py' file in saved directory.

2) Run code from IDE, depending on program:
e.g.	IDLE - Run > Run Module (F5)
	PyCharm - Run > Run... (Alt+Shift+F10)
				Select 'kMeansClustering'
