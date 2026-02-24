import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib.widgets import Slider
# for creating a responsive plot
from mpl_toolkits.mplot3d import Axes3D

class GMM():
    def __init__(self):
        pass
    # Function to calculate BIC for different numbers of clusters
    def calculate_bic_for_gmm(self,X, max_clusters:int):
        bic_values = []
        for n in range(1, max_clusters + 1):
            gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42).fit(X)
            bic_values.append(gmm.bic(X))
        return bic_values


    # Calculate BIC values for 1 to 10 clusters
    def GMM_calc(self,X):
        bic_values = self.calculate_bic_for_gmm(X, max_clusters=10)
        optimal_clusters = np.argmin(bic_values) + 1
        gmm= GaussianMixture(n_components=optimal_clusters, covariance_type='full', random_state=42).fit(X)
        labels = gmm.predict(X)
        return labels,gmm

    def visual(self,X,labels,gmm):
        xs = X[:, 0]
        ys = X[:, 1]
        zs = X[:, 2]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Set initial view angle
        init_azim = 0
        init_elev = 30
        
        #Scatter plot
        scatter = ax.scatter(xs, ys, zs, c=labels, cmap='viridis')
        centroids = ax.scatter(gmm.means_[:, 0], gmm.means_[:, 1], gmm.means_[:, 2],
                   c='red', marker='X', s=300, label='Centroids')

        # Set labels and title
        ax.set_title("3D plot")
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
        ax.legend()
        # Set the initial view angle
        ax.view_init(elev=init_elev, azim=init_azim)
        # Create an animation by rotating the view
        plt.subplots_adjust(bottom=0.25)
        ax_azim = plt.axes([0.2, 0.1, 0.6, 0.03])
        ax_elev = plt.axes([0.2, 0.05, 0.6, 0.03])

        # Create sliders for azimuth and elevation
        slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=init_azim, valstep=1)
        slider_elev = Slider(ax_elev, 'Elevation', 0, 360, valinit=init_elev, valstep=1)

        def update(val):
            ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
            fig.canvas.draw_idle()

        slider_azim.on_changed(update)
        slider_elev.on_changed(update)
        plt.show()
