from qcomp_nlce.simulation.vqe.plot_dispersion import plot_single_cluster, plot_nlce

plot_single_cluster(N=5, J=1.0, hl=0.0)
plot_single_cluster(N=4, J=1.0, hl=0.0)
plot_nlce(cluster_pairs=[(5, 4)], J=1.0, hl=0.0)
