import pickle, os, json
from collections import defaultdict
from scipy.spatial import ConvexHull
from scipy.stats import truncnorm
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt


def generate_convex_with_hull_of_clusters(range_k_clusters=(3, 10), range_std_devs=(0.2, 0.6),
                                          range_n_per_cluster=(5, 50), dims=2):
  k = np.random.randint(range_k_clusters[0], range_k_clusters[1] + 1)
  points = []
  for cluster in range(k):
    center = np.random.rand(dims) - 0.5
    std_dev = np.random.uniform(range_std_devs[0], range_std_devs[1], size=(dims,))
    n = np.random.randint(range_n_per_cluster[0], range_n_per_cluster[1] + 1)
    new_points = center + std_dev * truncnorm.rvs(a=-2 * std_dev, b=2 * std_dev, size=(n, dims,))
    points.append(new_points)

  return np.array(list(chain.from_iterable(points)))

def get_convex_hull_dataset_from_name(hull_name, num_points, dims=2):
  """ Return convex hull given the name of the hull string
  """
  if hull_name in ["square", "diamond", "quad"]:
    # Generate a few fixed sets of points
    hull_points_dict = {"square": np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]),
                        "diamond": np.array([[1, 0], [0, 1], [-1, 0], [0, -1]]),
                        "quad": np.array([[1, 0], [0, 2], [-1, 0], [0, -1.5]])
                        }
    hull_points = hull_points_dict[hull_name]
    xs, ys, hull = generate_convex_dataset_from_hull(num_points, hull_points, symmetric=False, scale_factor=1,
                                                     dims=dims)

  elif "hull" in hull_name:
    # Arbitrary random convex hull
    if "asym" in hull_name:
      xs, ys, hull = generate_convex_dataset(num_points, symmetric=False, scale_factor=1, dims=dims)
    elif "sym" in hull_name:
      xs, ys, hull = generate_convex_dataset(num_points, symmetric=True, scale_factor=1, dims=dims)
    else:
      raise ValueError("Need to specify whether it is symmetric or not with sym or asym")

  return xs, ys, hull

def generate_convex_dataset_from_hull(n, hull_points, symmetric=False, scale_factor=1, dims=2):
  """Generated asymmetric convex dataset with n points"""

  # MAKE TARGET FN BY GENERATING CONVEX HULL + ADDL CONTOURS

  if symmetric:
    ps = hull_points[hull_points[:, 1] > 0.]

    hull_points = np.array(list(ps) + list(-1 * ps))

  hull = ConvexHull(hull_points)

  # TAKE FINITE SAMPLES FROM TARGET FN

  # xs = np.random.uniform(size=n) * np.pi * 2
  # xs = np.stack((np.cos(xs), np.sin(xs)), axis=1)

  xs = np.random.normal(size=(n, dims))
  xs_length = np.expand_dims(np.sqrt(np.sum(xs ** 2, axis=1)), 1)
  xs = xs / xs_length

  ys = []
  for i, x in enumerate(xs):
    y = get_norm_of_unit_vector(x, hull)
    ys.append(y)

  ys = np.array(ys)
  return xs, ys, hull

def intersect_lines(a, b):
  """a and b are (2, 2) numpy arrays with 2 endpoints of line segment. This returns their intersection.
  https://stackoverflow.com/questions/44631259/line-line-intersection-in-python-with-numpy"""
  t, s = np.linalg.solve(np.array([a[1] - a[0], b[0] - b[1]]).T, b[0] - a[0])
  i1 = ((1 - t) * a[0] + t * a[1])
  i2 = ((1 - s) * b[0] + s * b[1])
  assert np.allclose(i1, i2), "{} {} {} {}".format(a, b, i1, i2)
  return i1


def get_norm_of_unit_vector(unit_vector, hull):
  """hull is a scipy.ConvexHull (containing origin)
  unit_vector is a vector with norm of 1 (from origin).
  this computes alpha in R+, s.t. alpha * unitvector lies on the convex hull. """
  assert np.allclose(np.linalg.norm(unit_vector), 1.)

  # No for loops
  # Based on: https://stackoverflow.com/questions/30486312/intersection-of-nd-line-with-convex-hull-in-python
  eq = hull.equations.T
  V, b = eq[:-1], eq[-1]
  alpha = -b / np.dot(V.T, unit_vector)
  hull_distance = np.min(alpha[alpha > 0])

  norm_of_unit_vector = 1. / hull_distance

  return norm_of_unit_vector


def generate_convex_dataset(n, symmetric=False, scale_factor=1, dims=2):
  """Generated asymmetric convex dataset with n points"""

  # MAKE TARGET FN BY GENERATING CONVEX HULL + ADDL CONTOURS
  hull_points = generate_convex_with_hull_of_clusters(dims=dims) * scale_factor

  # Subtract the mean of the hull points, so that the origin will be inside the convex hull
  hull_points = hull_points - np.mean(hull_points, axis=0)

  if symmetric:
    ps = hull_points[hull_points[:, 1] > 0.]

    hull_points = np.array(list(ps) + list(-1 * ps))

  print("Getting convex hull...for {}".format(hull_points.shape))
  hull = ConvexHull(hull_points)
  print("Done getting convex hull")

  # TAKE FINITE SAMPLES FROM TARGET FN
  # Generate random unit vectors in N-D by sampling gaussian N(0,1) for each dim/sample
  # Then normalizing the vector
  xs = np.random.normal(size=(n, dims))
  xs_length = np.expand_dims(np.sqrt(np.sum(xs ** 2, axis=1)), 1)
  xs = xs / xs_length

  #   xs = np.random.uniform(size=n) * np.pi * 2
  #   xs = np.stack((np.cos(xs), np.sin(xs)), axis=1)

  ys = []
  for i, x in enumerate(xs):
    y = get_norm_of_unit_vector(x, hull)
    ys.append(y)

  ys = np.array(ys)
  return xs, ys, hull


def plot_convex_dataset(xs, norm_xs, hull, extra_contours=[], name=None):
  """Here norm_xs are the targets (norm of unit vectors xs)"""

  fig = plt.figure(figsize=(4, 4))
  points = hull.points

  extra_hulls = [ConvexHull(points * c) for c in extra_contours]

  for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-', linewidth=1.5)

  for h in extra_hulls:
    for simplex in h.simplices:
      plt.plot(h.points[simplex, 0], h.points[simplex, 1], 'k:', linewidth=0.5)

  for x, norm in zip(xs, norm_xs):
    plt.plot(x[0], x[1], 'go', markersize=4)
    y_ = x / norm
    plt.plot((0, y_[0]), (0, y_[1]), 'r-')

  plt.axis('equal')
  if name is not None:
    fig.tight_layout()

    # Dump the raw data pickle file
    data = [xs, norm_xs, hull, extra_contours, name]
    pickle.dump(data, open('{}_raw.pickle'.format(name), 'wb'))
    # Save PDF of figure as well
    plt.savefig("{}.pdf".format(name), bbox_inches='tight', pad_inches=0)


  plt.close()
  return

def _plot_contour(ys, plotmarker='r-', zorder=None):
  hull = ConvexHull(ys)
  for simplex in hull.simplices:
    if zorder is not None:
      plt.plot(ys[simplex, 0], ys[simplex, 1], '{}'.format(plotmarker), linewidth=1., zorder=zorder)
    else:
      plt.plot(ys[simplex, 0], ys[simplex, 1], '{}'.format(plotmarker), linewidth=1.)

def _plot_contour_notebook(axs, ys, plotmarker='r-', zorder=None):
  hull = ConvexHull(ys)
  for simplex in hull.simplices:
    if zorder is not None:
      axs.plot(ys[simplex, 0], ys[simplex, 1], '{}'.format(plotmarker), linewidth=1., zorder=zorder)
    else:
      axs.plot(ys[simplex, 0], ys[simplex, 1], '{}'.format(plotmarker), linewidth=1.)

def _plot_points(ps, plotmarker='ro'):
  for p in ps:
    plt.plot(p[0], p[1], '{}'.format(plotmarker), markersize=.5)


def plot_contours(axs, xs, pred_points_list, hull, contour_list=[0.7, 0.85, 1., 1.15, 1.3], name=None):
  # We assume that the vector xs has a norm = 1 under that convex hull given by hull
  norm_xs = np.zeros(xs.shape)
  for i in range(xs.shape[0]):
    norm_xs[i] = get_norm_of_unit_vector(xs[i], hull)

  # Scale the vector x to have length = c under that hull
  unit_xs = xs / norm_xs  # unit_xs has a unit length under that norm
  epsilon = 1e-7

  for c, pred_points in zip(contour_list, pred_points_list):
    target_points = hull.points * c
    _plot_contour(axs, target_points, 'b-')
#     _plot_points(axs, pred_points)
    plotmarker='ro'
    axs.plot(pred_points[:,0], pred_points[:,1], '{}'.format(plotmarker), markersize=.5)

  axs.axis('equal')


def plot_contours_heatmap(xs, norm_func, hull, contour_list=[0.7, 0.85, 1., 1.15, 1.3], name=None):
  fig = plt.figure(figsize=(4, 4))

  # Plot also the groundtruth contour
  for c in contour_list:
    target_points = hull.points * c
    _plot_contour(target_points, 'b-', zorder=0.9)

  # Get a contour plot via matplotlib
  x = np.linspace(-1.75, 1.75, 100)
  y = np.linspace(-1.75, 1.75, 100)

  X, Y = np.meshgrid(x, y)
  X_flat = np.reshape(X, (np.prod(X.shape),))
  Y_flat = np.reshape(Y, (np.prod(Y.shape),))
  vectors = np.stack([X_flat,Y_flat], axis=1)
  Z_flat = norm_func(vectors)
  Z = np.reshape(Z_flat, X.shape)

  contours = plt.contour(X, Y, Z, colors='black', levels=contour_list)
  plt.clabel(contours, inline=True, fontsize=10)

  plt.imshow(Z, extent=[-1.5, 1.5, -1.5, 1.5], origin='lower',
             cmap='RdGy', alpha=0.5)
  plt.colorbar()

  fig.axes[0].set_aspect('equal', 'box') # The fig.axes[1] is for the colorbar
  fig.tight_layout()

  if name is not None:
    # Dump the raw data pickle file
    data = [X, Y, Z, hull, contour_list, name]
    pickle.dump(data, open('{}_raw.pickle'.format(name), 'wb'))
    # Save PDF of figure as well
    plt.savefig("{}.pdf".format(name), bbox_inches='tight', pad_inches=0)

  plt.close()

  return

def plot_contours_heatmap_notebook(axs, X, Y, Z, hull, contour_list=[0.7, 0.85, 1., 1.15, 1.3], name=None, xlim=[-1.75,1.75], ylim=[-1.75,1.75]):

  # Plot Contours of prediction
  contours = axs.contour(X, Y, Z, colors='black', levels=contour_list)
  axs.clabel(contours, inline=True, fontsize=10)

  # Plot also the groundtruth contour
  for c in contour_list:
    target_points = hull.points * c
    _plot_contour_notebook(axs, target_points, 'b-', zorder=0.9)

  im = axs.imshow(Z, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], origin='lower',
             cmap='RdBu', alpha=0.5, vmin=0, vmax=4)

  axs.set_aspect('equal', 'box') # The fig.axes[1] is for the colorbar  axs.tight_layout()

  return im

def plot_full_heatmap(input_folder, arches, datasets, train_size, xlim=[-1.5,1.5], ylim=[-1.5,1.5]):
    num_rows = len(datasets)
    num_cols = len(arches[0])
    fig, axs = plt.subplots(num_rows,num_cols, figsize=(3*num_cols,3*num_rows))

    for r, dataset in enumerate(datasets):
        for c, arch in enumerate(arches[r]):
            if 'train' in arch:
                # Plot the training data
                filename = '{}/hull-{}_train-{}_raw.pickle'.format(input_folder, dataset, train_size)
                # Plot the training set data
                xs, norm_xs, hull, extra_contours, name = pickle.load(open(filename,'rb'))
                points = hull.points
                extra_hulls = [ConvexHull(points * c) for c in extra_contours]
                for simplex in hull.simplices:
                    axs[r,c].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.5)
                for h in extra_hulls:
                    for simplex in h.simplices:
                        axs[r,c].plot(h.points[simplex, 0], h.points[simplex, 1], 'b:', linewidth=0.5)
                for x, norm in zip(xs, norm_xs):
                    axs[r,c].plot(x[0], x[1], 'go', markersize=2)

                axs[r,c].axis('equal')
                axs[r,c].set_aspect('equal','box') 
                axs[r,c].set_xlim(xlim[0],xlim[1])
                axs[r,c].set_ylim(ylim[0],ylim[1])
            else:
                # Plot the model predictions
                filename = '{}/hull-{}_train-{}_arch-{}_raw.pickle'.format(input_folder, dataset, train_size, arch)
                X, Y, Z, hull, contour_list, name = pickle.load(open(filename, 'rb'))
                im = plot_contours_heatmap_notebook(axs[r,c], X, Y, Z, hull, contour_list=contour_list, name=name, xlim=xlim, ylim=ylim)

            if r == 0:
                title = 'MLP' if 'mlp' in arch else 'Deep Norm' if 'deepnorm' in arch else \
                        'Wide Norm' if 'widenorm' in arch else 'Mahalanobis' if 'maha' in arch \
                        else 'Training Data' if 'train' in arch else ''
                axs[r,c].set_title(title, fontsize=18)
            # Turn off y-axis if not in the first column 
            if c != 0:
                axs[r,c].yaxis.set_visible(False)

            # Turn off x-axis if not in the last first column 
            if r < len(datasets) - 1:
                axs[r,c].xaxis.set_visible(False)

    fig.tight_layout(w_pad=0.0)
    fig.colorbar(im, ax=axs.ravel().tolist())
    
    return fig

# Code to get the architectures for plotting
def get_arches_to_plot(input_folder, training_sizes, hull_names, arches = ['mahalanobis', 'deepnorm', 'widenorm', 'mlp'], metric='mean_squared_error'):
    result_name = os.path.join(input_folder, 'results.txt')
    with open(result_name, 'r') as f:
        results = json.load(f)
    
    idx = 4 # 0 = test c=1, 1=train, 2=test c=2, 3=test c=0.5, 4=arch name
    for training_size in training_sizes:
        arch_list = []
        for hull_name in hull_names:
            # First column is the training set
            curr_arch_list = ['train'] 
            
            resdict = defaultdict(list)
            for result in results:
                arch_name = result[0]['post_emb_type']
                curr_train_size = result[0]['num_train']
                curr_hull_name = result[0]['hull_name']
                if curr_train_size == training_size and curr_hull_name == hull_name:
                    # Get the testing curve
                    arch = ''
                    for a in arches:
                        if a in arch_name:
                            arch = a
                    # Note: We are looking at the performance on the LAST epoch (which is where we save the plot data)
                    resdict[arch].append((result[2][metric][-1], result[1][metric][-1],
                                          result[3]['test_contour=2'][metric], 
                                          result[3]['test_contour=0.5'][metric], arch_name, result[2]['max_squared_error'][np.argmin(result[2][metric])],
                                          result[3]['test_contour=2']['max_squared_error']))

            for k,v in resdict.items():
                resdict[k] = list(sorted(v))[:1]

            min_tests = [resdict[k][0][idx] for k in arches]
            bests = np.zeros((4,))
            bests[np.argsort(min_tests)[:1]] = 1

            for k, best in zip(arches, bests):
                min_test = resdict[k][0][idx]

                curr_arch_list.append(resdict[k][0][idx])
            arch_list.append(curr_arch_list)
    return arch_list

if __name__ == '__main__':
  xs, ys, hull = generate_convex_dataset(200, symmetric=False, scale_factor=2)
  plot_convex_dataset(xs, ys, hull)