import sys
import math
import numpy as np


def fit_gmm(t, k=4):
    """
    Expectation Maximization algorithm for estimating k-cluster GMM on dataset t
    :param t: n_obs-by-n_feat data matrix, each row is an observation, each column is a feature
    :param k: number of clusters
    :return: result dictionary. quality is the highest log-likelihood achieved. clusters is the classification result
    under the best estimated model. parameters are the mus, sigmas and probabilities of each cluster
    """
    # convergence bound
    eps = 0.0001

    def gauss(x, m, s):
        """
        Calculate the multivariate gaussian distribution with values matrix x, mean vector m, and variance/covariance
        matrix s
        :param x: n_obs-by-n_feat data matrix, each row is a data
        :param m: 1-by-n_feat mean vector
        :param s: n_feat-by-n_feat variance/covariance matrix
        :return: n_obs-by-1 vector representing gaussian pdf of each data
        """
        for i in range(len(s)):
            if s[i, i] <= sys.float_info[3]:
                s[i, i] = sys.float_info[3]
        s_inv = np.linalg.inv(s)
        xm = np.matrix(x - m)
        return (2.0 * np.pi) ** (-len(x[1]) / 2.0) * (1.0 / (np.linalg.det(s) ** 0.5)) *\
            np.exp(-0.5 * np.sum(np.multiply(xm * s_inv, xm), axis=1))

    def init_params():
        mu = np.array([1.0 * t[np.random.choice(t.shape[0], 1, False), :]], np.float64)
        return {'mu': mu, 'sigma': np.matrix(np.diag([(min_max[f][1]-min_max[f][0])/2.0 for f in range(n_feat)])),
                'prob': 1.0 / k}

    n_obs = t.shape[0]  # number of observations
    n_feat = t.shape[1]  # number of features
    min_max = []

    # use the range of each feature to initialize the covariance matrix
    for f in range(n_feat):
        min_max.append((np.amin(t[:, f]), np.amax(t[:, f])))

    result = {}
    p_clust = np.ndarray([n_obs, k], np.float64)  # P(cluster|observation)
    p_x = np.ndarray([n_obs, k], np.float64)  # P(observation|cluster)

    for i in range(3):
        params = [init_params() for c in range(k)]
        old_log_est = sys.maxsize  # initialization
        log_est = sys.maxsize / 2 + eps  # initialization
        est_round = 0
        while abs(log_est - old_log_est) > eps and est_round < 1000:
            restart = False
            old_log_est = log_est
            # E-step
            for c in range(k):
                p_x[:, c: c + 1] = gauss(t, params[c]['mu'], params[c]['sigma'])
                p_clust[:, c: c + 1] = p_x[:, c: c + 1] * params[c]['prob']
            p_clust = np.divide(p_clust, np.tile(np.sum(p_clust, axis=1), (k, 1)).transpose())
            # M-step
            print(('iter:' + str(i) + ' est#: ' + str(est_round)))
            for c in range(k):
                sum_temp = math.fsum(p_clust[:, c])
                params[c]['prob'] = sum_temp / n_obs
                if params[c]['prob'] <= 1.0 / n_obs:
                    restart = True
                    print(('Restarting, p:' + str(params[c]['prob'])))
                    break
                m = np.sum(np.multiply(t, np.tile(p_clust[:, c: c + 1], (1, n_feat))), axis=0)
                params[c]['mu'] = m/sum_temp
                s = np.matrix(np.diag(np.zeros(n_feat, np.float64)))
                tm = t - params[c]['mu']
                p_c = p_clust[:, c: c + 1]
                p = np.tile(p_c.reshape(p_c.shape + (1,)), (1, 3, 3))
                s = np.sum(np.multiply(p, np.multiply(np.tile(tm.reshape(tm.shape + (1,)), (1, 1, 3)),
                                                      np.transpose(np.tile(tm.reshape(tm.shape + (1,)), (1, 1, 3)),
                                                                   (0, 2, 1)))), axis=0)
                params[c]['sigma'] = s / sum_temp

            # test for bound conditions
            if not restart:
                restart = True
                for c in range(1, k):
                    if not np.allclose(params[c]['mu'], params[c - 1]['mu'])\
                            or not np.allclose(params[c]['sigma'], params[c - 1]['sigma']):
                        restart = False
                        break

            if restart:
                old_log_est = sys.maxsize
                log_est = sys.maxsize / 2 + eps
                params = [init_params() for c in range(k)]
                continue

            # log-likelihood
            log_est = math.fsum([math.log(math.fsum([p_x[o, c] * params[c]['prob'] for c in range(k)]))
                                 for o in range(n_obs)])
            print(('(EM) old and new log likelihood: ' + str(old_log_est) + ' ' + str(log_est)))
            est_round += 1

        # evaluate quality
        quality = -log_est
        if quality not in result or quality > result['quality']:
            result['quality'] = quality
            result['params'] = params
            result['cluster'] = [[o for o in range(n_obs) if p_x[o, c] == max(p_x[o, :])] for c in range(k)]

    return result

# save parameters for prediction
red_barrel = np.loadtxt('red_barrel_sample.tar.gz')
red_barrel_param = fit_gmm(red_barrel)
np.save('red_barrel_param.npy', red_barrel_param['params'])

red_other = np.loadtxt('red_other_sample.tar.gz')
red_other_param = fit_gmm(red_other, 6)
np.save('red_other_param.npy', red_other_param['params'])

sky = np.loadtxt('sky_sample.tar.gz')
sky_param = fit_gmm(sky, 6)
np.save('sky_param.npy', sky_param['params'])

ground = np.loadtxt('ground_sample.tar.gz')
ground_param = fit_gmm(ground, 6)
np.save('ground_param.npy', ground_param['params'])
