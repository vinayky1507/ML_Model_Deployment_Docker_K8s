

from pycaret.datasets import get_data
data = get_data('insurance')
from pycaret.regression import *
s = setup(data, target = 'charges', session_id = 123)
lr = create_model('lr')
# plot_model(lr)
s2 = setup(data, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True, feature_interaction=True,
           bin_numeric_features= ['age', 'bmi'])
# s2[0].columns
lr = create_model('lr')
# plot_model(lr)
save_model(lr, 'deployment_28082022')
# deployment_28042020 = load_model('deployment_28082022')
