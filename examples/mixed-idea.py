from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

n_samples = 10000
n_features = 20
n_boost_round = 1000

X,y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)

print("------------------------------")
print("n_samples:     %d" % (n_samples))
print("n_features:    %d" % (n_features))
print("n_boost_round: %d" % (n_boost_round))
print("------------------------------")

for n_part in [1,2,4,8,16,32,64,128]:
    for n_infer in [1,2,4,8,16,32,64,128]:
        if n_infer > n_part:
            continue
        
        ind_train = np.arange(0, X_train.shape[0])
        part_train = np.array_split(ind_train, n_part)
    
        state = np.random.RandomState(42)
    
        depth_range = range(1,11)
        eta_range = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
     
        depth_best = None
        eta_best = None
        mse_best = 1e30
    
        for depth in depth_range:
            for eta in eta_range:
                z_train = np.zeros(X_train.shape[0])
                z_test = np.zeros(X_test.shape[0])
                for i in range(0, n_boost_round):
           
                    u_train = np.zeros(X_train.shape[0])
                    u_test = np.zeros(X_test.shape[0])
                    target = y_train - z_train
                    for p_idx in range(0, n_part):
                        p = part_train[p_idx]
                        bl = DecisionTreeRegressor(max_depth=depth, max_features='sqrt', random_state=state)
                        bl.fit(X_train[p,:], target[p])
                        u_test += bl.predict(X_test)

                        for pp_idx in range(0, n_infer):
#                            print("p_idx={} pp_idx={}".format(p_idx, (p_idx + pp_idx) % n_part))
                            pp = part_train[(p_idx + pp_idx) % n_part]
                            u_train[pp] += bl.predict(X_train[pp,:])

                    z_train = z_train + eta*u_train/n_infer
                    z_test = z_test + eta*u_test/n_part
    
                mse = mean_squared_error(y_test, z_test)
                print(">> n_part: %3d, n_infer: %3d, eta: %.2f, depth: %2d, mse_test: %e" % (n_part, n_infer, eta, depth, mse))
    
                if mse < mse_best:
                    mse_best = mse
                    eta_best = eta
                    depth_best = depth
    
        print(">> n_part: %3d, n_infer: %3d, eta-opt: %.2f, depth-opt: %2d, mse_test: %e" % (n_part, n_infer, eta_best, depth_best, mse_best))

